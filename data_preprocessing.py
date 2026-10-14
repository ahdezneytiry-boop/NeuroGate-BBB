import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # 新增：用于保存scaler

# -------------------------- 全局路径配置（需根据实际修改） --------------------------
RAW_DATA_PATH = "data/bbb_cls_final_data.csv"  # 原始数据路径
SAVE_DIR = r"E:\jqxx\机器学习大作业"  # 结果保存目录
os.makedirs(SAVE_DIR, exist_ok=True)  # 确保目录存在

# -------------------------- 1. 读取BBB数据（含原训练/测试标签） --------------------------
df = pd.read_csv(RAW_DATA_PATH)

# 确认关键标签列存在
if "random_train_test_label" not in df.columns:
    raise ValueError("原表格中未找到'random_train_test_label'列，请检查列名是否正确！")
print(f"✅ 确认原表格包含训练/测试标签列：'random_train_test_label'")

# -------------------------- 2. 基础信息查看 --------------------------
print("="*50)
print("1. 数据基础信息（含原训练/测试标签）")
print("="*50)
print(f"总样本数：{df.shape[0]}，总列数：{df.shape[1]}")
print(f"\n所有列名：{df.columns.tolist()}")
print("\n各列数据类型与缺失值统计：")
print(df.info())
print("\n前5行数据预览（含标签列）：")
print(df[["index", "Smiles_unify", "value", "random_train_test_label"]].head())

# -------------------------- 3. 目标变量分析 --------------------------
print("\n" + "="*50)
print("2. 目标变量（BBB属性：value）分析")
print("="*50)
target_counts = df["value"].value_counts()
target_ratio = df["value"].value_counts(normalize=True).round(3) * 100
target_analysis = pd.DataFrame({
    "样本数量": target_counts,
    "占比（%）": target_ratio
})
print("BBB属性类别分布：")
print(target_analysis)

# 可视化类别分布
plt.figure(figsize=(6, 4))
target_counts.plot(kind="bar", color=["#22399d", "#abb4f2"])
plt.title("BBB Attribute Distribution (1=Positive, 2=Negative)")
plt.xlabel("BBB Value (1=Present, 2=Absent)")
plt.ylabel("Sample Count")
plt.xticks(rotation=0)
plt.savefig(os.path.join(SAVE_DIR, "target_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ 目标变量分布图已保存至：{SAVE_DIR}\\target_distribution.png")

# -------------------------- 4. Smiles有效性验证 --------------------------
print("\n" + "="*50)
print("3. Smiles格式有效性验证")
print("="*50)
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

df["smiles_valid"] = df["Smiles_unify"].apply(is_valid_smiles)
valid_count = df["smiles_valid"].sum()
invalid_count = len(df) - valid_count
print(f"有效Smiles数量：{valid_count}（{valid_count/len(df)*100:.1f}%）")
print(f"无效Smiles数量：{invalid_count}（{invalid_count/len(df)*100:.1f}%）")

if invalid_count > 0:
    invalid_samples = df[~df["smiles_valid"]][["index", "Smiles_unify", "random_train_test_label"]]
    print("\n无效Smiles样本详情（含标签列）：")
    print(invalid_samples)
    # 保存无效样本清单
    invalid_samples.to_csv(os.path.join(SAVE_DIR, "invalid_smiles.csv"), index=False, encoding="utf-8")
    print(f"✅ 无效Smiles清单已保存至：{SAVE_DIR}\\invalid_smiles.csv")

# -------------------------- 5. 数据清洗 --------------------------
print("\n" + "="*50)
print("4. 数据清洗过程（保留原训练/测试标签列）")
print("="*50)

# 1. 删除无效Smiles样本
df_clean = df[df["smiles_valid"]].copy()
print(f"1. 删除无效Smiles样本：清洗前{len(df)}个 → 清洗后{len(df_clean)}个")

# 2. 目标变量编码转换（2→0，1→1）
df_clean["target"] = df_clean["value"].replace(2, 0)
print("2. 目标变量编码转换：原2（负样本）→ 0，原1（正样本）→ 1")
print("转换后目标变量分布：")
print(df_clean["target"].value_counts())

# 3. 删除冗余列（保留random_train_test_label）
drop_cols = ["index", "value", "smiles_valid"]
df_clean = df_clean.drop(columns=drop_cols)
print(f"3. 删除冗余列：最终保留列 → {df_clean.columns.tolist()}")

# 4. 重置索引
df_clean = df_clean.reset_index(drop=True)
print(f"4. 重置索引：清洗后数据形状：{df_clean.shape}")

# 预览清洗后数据
print("\n清洗后数据前5行（含原训练/测试标签）：")
print(df_clean[["Smiles_unify", "target", "random_train_test_label"]].head())

# -------------------------- 6. 分子特征提取 --------------------------
print("\n" + "="*50)
print("5. 分子特征提取过程（最终表含原训练/测试标签）")
print("="*50)
def extract_molecular_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([np.nan]*10)
    
    logP = Descriptors.MolLogP(mol)
    features = {
        "mol_weight": Descriptors.MolWt(mol),
        "logP": logP,
        "logS": 0.87 - 0.85 * logP,
        "h_bond_donor": Descriptors.NumHDonors(mol),
        "h_bond_acceptor": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_rot_bonds": Descriptors.NumRotatableBonds(mol),
        "fraction_csp3": Descriptors.FractionCSP3(mol),
        "num_rings": Descriptors.RingCount(mol),
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol)
    }
    return pd.Series(features)

# 批量提取特征
print("正在从Smiles提取10个分子理化特征...")
mol_features = df_clean["Smiles_unify"].apply(extract_molecular_features)

# 合并特征+清洗数据（含标签列）
df_transformed = pd.concat([df_clean, mol_features], axis=1)
print(f"特征提取完成：合并后数据形状 → {df_transformed.shape}（行×列）")
print(f"最终表所有列（含标签）：{df_transformed.columns.tolist()}")

# 检查缺失值
feature_missing = df_transformed[mol_features.columns].isnull().sum()
missing_count = feature_missing.sum()
print(f"\n分子特征缺失值：{'无' if missing_count == 0 else f'{missing_count}个'}")
if missing_count > 0:
    print("缺失特征详情：")
    print(feature_missing[feature_missing > 0])
    # 填充缺失值（用均值）
    df_transformed[mol_features.columns] = df_transformed[mol_features.columns].fillna(df_transformed[mol_features.columns].mean())
    print("✅ 缺失值已用均值填充")

# 预览最终转换表
print("\n最终转换数据前3行（含特征+原训练/测试标签）：")
preview_cols = ["Smiles_unify", "target", "random_train_test_label", "mol_weight", "logP", "tpsa"]
print(df_transformed[preview_cols].head(3))

# 保存最终转换表
save_path = os.path.join(SAVE_DIR, "bbb_clean_with_features_and_label.csv")
df_transformed.to_csv(save_path, index=False, encoding="utf-8")
print(f"\n✅ 含分子特征+原训练/测试标签的数据已保存：{save_path}")

# -------------------------- 7. 数据划分 + 保存StandardScaler（新增：部署必需） --------------------------
print("\n" + "="*50)
print("6. 基于原表格标签的数据划分 + 保存标准化器")
print("="*50)

# 1. 查看原标签规则
label_unique = df_transformed["random_train_test_label"].unique()
print(f"1. 原表格训练/测试标签规则：{label_unique}")

# 2. 按原标签拆分
train_condition = df_transformed["random_train_test_label"] == "train"
test_condition = df_transformed["random_train_test_label"] == "test"

# 分离特征X、目标y
feature_cols = mol_features.columns.tolist()
X = df_transformed[feature_cols]
y = df_transformed["target"]

# 按原标签拆分训练/测试集
X_train_full = X[train_condition].copy()
y_train_full = y[train_condition].copy()
X_test = X[test_condition].copy()
y_test = y[test_condition].copy()

print(f"\n2. 按原标签拆分结果（复用原划分）：")
print(f"- 原训练集（全量）：{X_train_full.shape[0]}个样本，特征数：{X_train_full.shape[1]}")
print(f"- 原测试集：{X_test.shape[0]}个样本，特征数：{X_test.shape[1]}")
print(f"- 原训练集目标分布：{y_train_full.value_counts(normalize=True).round(3)}")
print(f"- 原测试集目标分布：{y_test.value_counts(normalize=True).round(3)}")

# 拆分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.1,
    random_state=42,
    stratify=y_train_full
)
print(f"\n3. 拆分验证集后（原测试集不变）：")
print(f"- 训练集（建模用）：{X_train.shape[0]}个样本")
print(f"- 验证集（调优用）：{X_val.shape[0]}个样本")
print(f"- 原测试集（评估用）：{X_test.shape[0]}个样本")

# 特征标准化 + 保存scaler（部署必需）
numeric_feature_cols = mol_features.columns.tolist()
scaler = StandardScaler()
X_train[numeric_feature_cols] = scaler.fit_transform(X_train[numeric_feature_cols])
X_val[numeric_feature_cols] = scaler.transform(X_val[numeric_feature_cols])
X_test[numeric_feature_cols] = scaler.transform(X_test[numeric_feature_cols])
print(f"\n4. 数值特征标准化完成（仅用原训练集拟合，避免数据泄露）")

# 保存StandardScaler（新增：部署时预处理新数据需用）
scaler_path = os.path.join(SAVE_DIR, "standard_scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"✅ 特征标准化器已保存：{scaler_path}")

# 保存划分后的数据
excel_save_path = os.path.join(SAVE_DIR, "bbb_admet_split_with_original_label.xlsx")
with pd.ExcelWriter(excel_save_path, engine="openpyxl") as writer:
    # 训练集（含原标签）
    train_label = df_transformed.loc[X_train.index, "random_train_test_label"]
    train_df = pd.concat([
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        train_label.reset_index(drop=True)
    ], axis=1)
    train_df.rename(columns={"random_train_test_label": "original_train_test_label"}, inplace=True)
    train_df.to_excel(writer, sheet_name="train_set", index=False)

    # 验证集（含原标签）
    val_label = df_transformed.loc[X_val.index, "random_train_test_label"]
    val_df = pd.concat([
        X_val.reset_index(drop=True),
        y_val.reset_index(drop=True),
        val_label.reset_index(drop=True)
    ], axis=1)
    val_df.rename(columns={"random_train_test_label": "original_train_test_label"}, inplace=True)
    val_df.to_excel(writer, sheet_name="val_set", index=False)

    # 原测试集（含原标签）
    test_label = df_transformed.loc[X_test.index, "random_train_test_label"]
    test_df = pd.concat([
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
        test_label.reset_index(drop=True)
    ], axis=1)
    test_df.rename(columns={"random_train_test_label": "original_train_test_label"}, inplace=True)
    test_df.to_excel(writer, sheet_name="test_set", index=False)

print(f"\n5. 划分后数据已保存（含原标签追溯）：{excel_save_path}")
print(f"\n🎉 数据预处理全流程完成！生成文件汇总：")
print(f"1. {save_path} → 含特征+标签的清洗数据")
print(f"2. {scaler_path} → 特征标准化器（部署必需）")
print(f"3. {excel_save_path} → 划分后的训练/验证/测试集")
print(f"4. {os.path.join(SAVE_DIR, 'target_distribution.png')} → 目标变量分布图")