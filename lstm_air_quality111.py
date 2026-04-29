# 0. 环境设置

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, LSTM, Dropout, Layer, Permute,
    Multiply, Flatten, RepeatVector
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import keras.backend as K  # noqa: kept for get_config compatibility


# 1. 基础参数设置

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "LSTM-Multivariate_pollution.csv"
TEST_FILE  = BASE_DIR / "pollution_test_data1.csv"

LOSS_PLOT_FILE       = BASE_DIR / "loss_plot_improved.png"
PREDICTION_PLOT_FILE = BASE_DIR / "prediction_comparison_improved.png"
RESULT_FILE          = BASE_DIR / "prediction_results_improved.csv"
MODEL_FILE           = BASE_DIR / "lstm_air_quality_model_improved.keras"

# 模型超参数
N_HOURS       = 24    # 回看窗口（小时）
N_OUT         = 1     # 预测步长
EPOCHS        = 80
BATCH_SIZE    = 64
LEARNING_RATE = 0.0005
N_CV_SPLITS   = 5     # TimeSeriesSplit 折数


# 2. 工具函数：滑动窗口转监督学习

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # 输入：t-n_in, ..., t-1
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"var{j+1}(t-{i})" for j in range(n_vars)]

    # 输出：t, t+1, ..., t+n_out-1（仅第 0 列 = pollution）
    for i in range(0, n_out):
        cols.append(df.shift(-i).iloc[:, [0]])   # 只取第 0 列（pollution）
        label = f"pollution(t)" if i == 0 else f"pollution(t+{i})"
        names.append(label)

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)

    return agg


# 3. 数据列名标准化

def standardize_columns(df):
    rename_map = {
        "pm2.5": "pollution", "PM2.5": "pollution",
        "DEWP":  "dew",  "TEMP": "temp",  "PRES": "press",
        "cbwd":  "wnd_dir", "Iws": "wnd_spd",
        "Is":    "snow", "Ir":  "rain"
    }
    df = df.rename(columns=rename_map)

    required = ["pollution", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少以下必要列：{missing}")

    return df[required].copy()


# 4. 风向 One-Hot 编码

def encode_wind_direction(df, categories):
    df = df.copy()
    df["wnd_dir"] = pd.Categorical(df["wnd_dir"].astype(str), categories=categories)
    return pd.get_dummies(df, columns=["wnd_dir"], prefix="wnd_dir")


# 5. 自定义 Attention 层

class TemporalAttention(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, T, F)
        e = tf.tanh(tf.matmul(x, self.W) + self.b)   # (batch, T, 1)
        a = tf.nn.softmax(e, axis=1)                   # (batch, T, 1)
        output = x * a                                 # (batch, T, F)
        return tf.reduce_sum(output, axis=1)           # (batch, F)

    def compute_output_shape(self, input_shape):
        # 告知 Keras 输出 shape，避免推断失败
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return super().get_config()


# 6. 构建带 Attention 的双层 LSTM 模型
def build_model(n_hours, n_features, n_out, lr):

    inp = Input(shape=(n_hours, n_features), name="input")

    x = LSTM(128, return_sequences=True, name="lstm_1")(inp)
    x = Dropout(0.2, name="drop_1")(x)

    x = LSTM(64, return_sequences=True, name="lstm_2")(x)
    x = Dropout(0.2, name="drop_2")(x)

    x = TemporalAttention(name="attention")(x)   # (batch, 64)

    x = Dense(32, activation="relu", name="dense_1")(x)
    out = Dense(n_out, name="output")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss="mae", optimizer=Adam(learning_rate=lr))
    return model


# 7. 反归一化辅助函数

def inverse_pollution(scaler, scaled_values):

    min_val   = scaler.data_min_[0]
    scale_val = scaler.data_range_[0]          # data_max - data_min
    return scaled_values * scale_val + min_val


# 8. 检查文件
for f in [TRAIN_FILE, TEST_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"找不到文件：{f}")


# 9. 读取 & 预处理数据
print("正在读取数据...")

train_raw = pd.read_csv(TRAIN_FILE, header=0)
test_raw  = pd.read_csv(TEST_FILE,  header=0)

train_df = standardize_columns(train_raw).dropna()
test_df  = standardize_columns(test_raw).dropna()

print("训练集原始形状:", train_df.shape)
print("测试集原始形状:", test_df.shape)

# ---- One-Hot 编码 ----
print("\n正在进行风向 One-Hot 编码...")
wind_categories = sorted(train_df["wnd_dir"].astype(str).unique())
train_encoded = encode_wind_direction(train_df, wind_categories).astype("float32")
test_encoded  = encode_wind_direction(test_df,  wind_categories).astype("float32")
test_encoded  = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

print("编码后训练集形状:", train_encoded.shape)
print("编码后测试集形状:", test_encoded.shape)
print("特征列:", list(train_encoded.columns))

# ---- 归一化（只 fit 训练集） ----

print("\n正在归一化...")
feature_scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = feature_scaler.fit_transform(train_encoded)
test_scaled  = feature_scaler.transform(test_encoded)

n_features = train_scaled.shape[1]
print("输入特征数:", n_features)


# 10. 构造监督学习数据

print(f"\n构造监督学习数据：回看 {N_HOURS}h，预测 {N_OUT} 步...")

train_reframed = series_to_supervised(train_scaled, N_HOURS, N_OUT)
test_reframed  = series_to_supervised(test_scaled,  N_HOURS, N_OUT)

n_obs = N_HOURS * n_features

print("监督学习训练集形状:", train_reframed.shape)
print("监督学习测试集形状:", test_reframed.shape)

# ---- 提取 X / y ----
def split_xy(reframed, n_obs, n_out, n_hours, n_features):
    values = reframed.values
    X = values[:, :n_obs].reshape(-1, n_hours, n_features)
    y = values[:, n_obs:]          # shape: (samples, n_out)
    if n_out == 1:
        y = y[:, 0]                # shape: (samples,)
    return X, y

train_X, train_y = split_xy(train_reframed, n_obs, N_OUT, N_HOURS, n_features)
test_X,  test_y  = split_xy(test_reframed,  n_obs, N_OUT, N_HOURS, n_features)

print("\n训练集 X:", train_X.shape, "y:", train_y.shape)
print("测试集  X:", test_X.shape,  "y:", test_y.shape)



# 11. TimeSeriesSplit 交叉验证

print(f"\n开始 TimeSeriesSplit {N_CV_SPLITS} 折交叉验证（评估用）...")

tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
cv_rmse_list, cv_mae_list = [], []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_X), start=1):
    cv_X_tr, cv_y_tr = train_X[tr_idx], train_y[tr_idx]
    cv_X_val, cv_y_val = train_X[val_idx], train_y[val_idx]

    cv_model = build_model(N_HOURS, n_features, N_OUT, LEARNING_RATE)

    cv_early = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    cv_lr    = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                 min_lr=1e-6, verbose=0)

    cv_model.fit(
        cv_X_tr, cv_y_tr,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(cv_X_val, cv_y_val),
        verbose=0, shuffle=False,
        callbacks=[cv_early, cv_lr]
    )

    yhat_cv = cv_model.predict(cv_X_val, verbose=0)

    # 反归一化
    if N_OUT == 1:
        pred_inv = inverse_pollution(feature_scaler, yhat_cv[:, 0] if yhat_cv.ndim == 2 else yhat_cv)
        true_inv = inverse_pollution(feature_scaler, cv_y_val)
    else:
        pred_inv = inverse_pollution(feature_scaler, yhat_cv[:, 0])
        true_inv = inverse_pollution(feature_scaler, cv_y_val[:, 0])

    fold_rmse = np.sqrt(np.mean((pred_inv - true_inv) ** 2))
    fold_mae  = np.mean(np.abs(pred_inv - true_inv))
    cv_rmse_list.append(fold_rmse)
    cv_mae_list.append(fold_mae)

    print(f"  Fold {fold}: RMSE={fold_rmse:.3f}, MAE={fold_mae:.3f}")

print(f"\n交叉验证结果汇总：")
print(f"  RMSE: {np.mean(cv_rmse_list):.3f} ± {np.std(cv_rmse_list):.3f}")
print(f"  MAE : {np.mean(cv_mae_list):.3f} ± {np.std(cv_mae_list):.3f}")


# 12. 在完整训练集上训练最终模型

print("\n在完整训练集上训练最终模型...")

# 取最后 20% 作为验证集（监控 early stopping，不参与 CV 评估）
n_valid = int(len(train_X) * 0.2)
final_X_tr,  final_y_tr  = train_X[:-n_valid], train_y[:-n_valid]
final_X_val, final_y_val = train_X[-n_valid:], train_y[-n_valid:]

model = build_model(N_HOURS, n_features, N_OUT, LEARNING_RATE)
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                min_lr=1e-6, verbose=1)

history = model.fit(
    final_X_tr, final_y_tr,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(final_X_val, final_y_val),
    verbose=2, shuffle=False,
    callbacks=[early_stop, reduce_lr]
)


# 13. 绘制训练损失曲线

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Improved LSTM Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PLOT_FILE, dpi=300)
plt.close()
print(f"\n训练损失曲线已保存：{LOSS_PLOT_FILE}")


# 14. 独立测试集预测

print("\n对独立测试集进行预测...")

yhat_scaled = model.predict(test_X, verbose=0)   # (samples, n_out) 或 (samples, 1)

# 反归一化（统一取第 0 步，即 t 时刻）
if N_OUT == 1:
    pred_flat = yhat_scaled[:, 0] if yhat_scaled.ndim == 2 else yhat_scaled
    true_flat = test_y
else:
    pred_flat = yhat_scaled[:, 0]
    true_flat = test_y[:, 0]

inv_yhat = inverse_pollution(feature_scaler, pred_flat)
inv_y    = inverse_pollution(feature_scaler, true_flat)


# 15. 计算评价指标

rmse = np.sqrt(np.mean((inv_yhat - inv_y) ** 2))
mae  = np.mean(np.abs(inv_yhat - inv_y))

print("\n========== 测试集最终评估结果 ==========")
print(f"测试集 RMSE : {rmse:.3f}")
print(f"测试集 MAE  : {mae:.3f}")
print(f"\n（供参考）CV 均值 RMSE: {np.mean(cv_rmse_list):.3f}, MAE: {np.mean(cv_mae_list):.3f}")


# 16. 保存预测结果

results = pd.DataFrame({
    "Actual_PM2.5":    inv_y,
    "Predicted_PM2.5": inv_yhat,
    "Absolute_Error":  np.abs(inv_yhat - inv_y)
})
results.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
print(f"\n预测结果已保存：{RESULT_FILE}")



# 17. 绘制真实值与预测值对比图

fig, axes = plt.subplots(1, 1, figsize=(12, 8))


axes[0].plot(inv_y,    label="Actual PM2.5",    alpha=0.8)
axes[0].plot(inv_yhat, label="Predicted PM2.5", alpha=0.8)
axes[0].set_title("Actual vs Predicted PM2.5 — Improved LSTM (Full Test Set)")
axes[0].set_xlabel("Time Steps / Hours")
axes[0].set_ylabel("PM2.5 Concentration")
axes[0].legend()



plt.tight_layout()
plt.savefig(PREDICTION_PLOT_FILE, dpi=300)
plt.close()
print(f"预测对比图已保存：{PREDICTION_PLOT_FILE}")


# 18. 保存模型

model.save(MODEL_FILE)
print(f"模型已保存：{MODEL_FILE}")

print("\n 全部流程运行完成。")