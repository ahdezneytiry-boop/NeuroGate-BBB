import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import joblib
import os

# -------------------------- 配置路径（需修改为你的特征工程数据路径） --------------------------
# 特征工程生成的数据集路径（之前运行feature_engineering.py生成的）
DATA_PATH = r"E:\jqxx\机器学习大作业\bbb_feature_engineering_result.csv"
# 生成的模型保存路径（直接保存到BBB_Deploy目录）
MODEL_SAVE_PATH = r"E:\BBB_Deploy\xgboost_core_only.bin"
# 核心特征列名文件路径（之前生成的）
FEATURE_COLS_PATH = r"E:\jqxx\机器学习大作业\core_feature_cols.txt"
# ------------------------------------------------------------------------------

# 1. 加载数据
df = pd.read_csv(DATA_PATH)
# 读取核心特征列名
with open(FEATURE_COLS_PATH, "r") as f:
    core_cols = f.read().splitlines()

# 2. 拆分训练/测试集
X = df[core_cols].values
y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 训练XGBoost模型（仅生成核心模型）
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    scale_pos_weight=0.45,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 4. 保存模型（生成xgboost_core_only.bin）
xgb_model.save_model(MODEL_SAVE_PATH)
print(f"✅ 模型已生成：{MODEL_SAVE_PATH}")

# 5. 验证模型
y_pred = xgb_model.predict(X_test)
recall = recall_score(y_test, y_pred)
print(f"✅ 模型验证通过，测试集召回率：{recall:.4f}")