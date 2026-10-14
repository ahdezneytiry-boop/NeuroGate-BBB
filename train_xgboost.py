# 导入依赖库 + 容错处理
try:
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import os
    import sys
except ImportError as e:
    print(f"缺失依赖库：{e}")
    print("安装命令：")
    print('"C:/Users/ahdezneytiry-boop1/AppData/Local/Programs/Python/Python311/python.exe" -m pip install numpy xgboost scikit-learn --force-reinstall')
    sys.exit(1)

# -------------------------- 核心配置 --------------------------
MODEL_SAVE_PATH = "data/xgboost_core_only.bin"  # 模型保存路径
# 核心特征列名（与后端特征提取顺序严格一致）
core_cols = [
    "mol_wt", "logp", "solubility", "h_donor", "h_acceptor",
    "tpsa", "rot_bonds", "csp3_frac", "ring_count", "aromatic_rings",
    "pca_1", "pca_2", "pca_3", "pca_4", "pca_5",
    "pca_6", "pca_7", "pca_8", "pca_9", "pca_10"
]

# -------------------------- 工具函数 --------------------------
def fix_windows_path(path):
    """修复Windows路径编码问题"""
    if sys.platform == "win32":
        try:
            return path.encode("gbk").decode("utf-8", errors="ignore")
        except:
            return path
    return path

def ensure_dir_exists(dir_path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录：{dir_path}")

# -------------------------- 主训练逻辑 --------------------------
def main():
    # 预处理路径
    model_path = fix_windows_path(MODEL_SAVE_PATH)
    deploy_dir = os.path.dirname(model_path)
    ensure_dir_exists(deploy_dir)

    # 生成模拟训练数据（20维特征，匹配core_cols）
    print("\n生成模拟训练数据...")
    np.random.seed(42)
    X = np.random.rand(1000, len(core_cols))  # 1000条样本，20维特征
    y = np.random.randint(0, 2, 1000)         # 二分类标签（0/1）
    
    # 拆分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"数据生成完成：训练集{len(X_train)}条，测试集{len(X_test)}条")

    # 转换为XGBoost原生DMatrix格式
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=core_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=core_cols)

    # 原生XGBoost参数（轻量化，快速训练）
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "learning_rate": 0.1,
        "eval_metric": "auc",
        "random_state": 42,
        "verbosity": 0  # 静默模式
    }

    # 训练原生Booster模型
    print("\n训练XGBoost模型...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # 保存原生格式模型（兼容Booster.load_model）
    booster.save_model(model_path)

    # 验证模型生成
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / 1024
        print(f"模型生成成功！")
        print(f"路径：{model_path}")
        print(f"大小：{file_size:.2f} KB")

        # 验证模型加载
        test_booster = xgb.Booster()
        test_booster.load_model(model_path)
        print("模型加载验证通过！")
    else:
        print(f"模型生成失败：{model_path}")
        sys.exit(1)

if __name__ == "__main__":
    print("="*50)
    print("开始生成XGBoost原生模型")
    print("="*50)
    main()
    print("\n模型生成完成！")