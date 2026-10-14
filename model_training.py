import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem, MolStandardize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from warnings import filterwarnings

filterwarnings("ignore")

# -------------------------- 全局配置（与前两个脚本一致） --------------------------
# 结果保存根目录（当前脚本所在目录，或指定绝对路径）
BASE_DIR = r"E:\jqxx\机器学习大作业"
# 数据路径（特征工程生成的文件）
DATA_FILE = os.path.join(BASE_DIR, "bbb_feature_engineering_result.csv")
# 模型保存路径（部署必需）
GNN_MODEL_SAVE = os.path.join(BASE_DIR, "best_gnn_model_optimized.pth")
XGB_CORE_SAVE = os.path.join(BASE_DIR, "xgboost_core_only.bin")
XGB_FUSION_SAVE = os.path.join(BASE_DIR, "xgboost_with_gnn.bin")
# 其他结果保存路径
SUMMARY_SAVE = os.path.join(BASE_DIR, "model_optimization_summary.csv")
TEST_PRED_SAVE = os.path.join(BASE_DIR, "test_set_predictions.csv")

# 确保目录存在
os.makedirs(BASE_DIR, exist_ok=True)

# 随机种子（完全可复现）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------- 工具函数 --------------------------
def check_and_make_dir(file_path):
    """检查并创建文件保存目录"""
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return file_path

def smiles_augmentation(smiles, n_aug=1):
    """Smiles增强（鲁棒版，避免无效Smiles）"""
    aug_smiles = [smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or n_aug <= 0:
        return aug_smiles

    # 标准化分子（降级处理兼容旧版rdkit）
    try:
        normalizer = MolStandardize.normalize.Normalizer()
        mol = normalizer.normalize(mol)
    except:
        pass

    valid_aug = []
    for _ in range(n_aug * 2):
        try:
            # 随机原子顺序扰动
            shuffled_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            mol_shuffled = Chem.MolFromSmiles(shuffled_smiles)
            if mol_shuffled is None:
                continue
            # 键/氢原子扰动
            if random.random() > 0.5:
                aug1 = Chem.MolToSmiles(mol_shuffled, allBondsExplicit=random.choice([True, False]))
                if Chem.MolFromSmiles(aug1) is not None:
                    valid_aug.append(aug1)
            if random.random() > 0.5:
                aug2 = Chem.MolToSmiles(mol_shuffled, allHsExplicit=random.choice([True, False]))
                if Chem.MolFromSmiles(aug2) is not None:
                    valid_aug.append(aug2)
        except:
            continue

    # 去重+限制数量
    aug_smiles += list(set(valid_aug))
    aug_smiles = list(set(aug_smiles))[:n_aug+1]
    return aug_smiles

def truncate_extreme_values(df, cols, lower_quantile=0.01, upper_quantile=0.99):
    """极端值截断（处理NaN）"""
    df_processed = df.copy()
    cols = [col for col in cols if col in df_processed.columns and df_processed[col].notna().sum() > 0]
    if not cols:
        return df_processed

    for col in cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        q1 = df_processed[col].quantile(lower_quantile)
        q99 = df_processed[col].quantile(upper_quantile)
        df_processed[col] = df_processed[col].clip(lower=q1, upper=q99)
        # TPSA强制非负
        if col.lower() == "tpsa":
            df_processed[col] = df_processed[col].clip(lower=0)
    return df_processed

# -------------------------- 1. 数据加载与预处理 --------------------------
print("="*60)
print("步骤1：数据加载 + 数据层面优化")
print("="*60)

# 加载特征工程生成的数据
try:
    df = pd.read_csv(DATA_FILE)
    # 填充全局NaN
    df = df.fillna(df.median(numeric_only=True))
    print(f"✅ 成功加载数据：{len(df)} 条样本")
except FileNotFoundError:
    raise FileNotFoundError(f"\n❌ 数据文件不存在：{DATA_FILE}\n提示：需先运行feature_engineering.py生成该文件")

# 1.1 异常值处理
phys_chem_cols = [
    "mol_weight", "logP", "logS", "h_bond_donor", "h_bond_acceptor", "tpsa",
    "num_rot_bonds", "fraction_csp3", "num_rings", "num_aromatic_rings"
]
df = truncate_extreme_values(df, phys_chem_cols)
print(f"✅ 异常值处理完成：处理 {len([c for c in phys_chem_cols if c in df.columns])} 个理化特征")

# 1.2 拆分训练/测试集（按原标签）
train_df = df[df["random_train_test_label"] == "train"].reset_index(drop=True)
test_df = df[df["random_train_test_label"] == "test"].reset_index(drop=True)
print(f"✅ 原始数据拆分：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")

# 1.3 Smiles增强（仅训练集，提升泛化性）
print("🔄 生成Smiles增强样本...")
aug_train_data = []
n_aug_per_sample = 1  # 每个样本增强1条
for idx, row in train_df.iterrows():
    smiles = row.get("Smiles_unify", "")
    if pd.isna(smiles) or smiles == "":
        aug_train_data.append(row)
        continue
    # 生成增强样本
    aug_smiles_list = smiles_augmentation(smiles, n_aug=n_aug_per_sample)
    for aug_smiles in aug_smiles_list:
        new_row = row.copy()
        new_row["Smiles_unify"] = aug_smiles
        aug_train_data.append(new_row)

# 去重后构建增强训练集
train_df_aug = pd.DataFrame(aug_train_data).drop_duplicates(subset=["Smiles_unify", "target"]).reset_index(drop=True)
aug_ratio = (len(train_df_aug) - len(train_df)) / len(train_df) * 100
print(f"✅ Smiles增强完成：原始 {len(train_df)} → 增强后 {len(train_df_aug)}（+{aug_ratio:.1f}%）")

# 1.4 SMOTE过采样（平衡训练集样本）
print("🔄 执行SMOTE过采样平衡样本...")
# 定义特征列（仅数值特征，排除非特征列）
gnn_node_attr_cols = ["logP", "tpsa", "h_bond_donor"]  # GNN节点属性列
xgb_core_feature_cols = [
    col for col in df.columns
    if col not in ["Smiles_unify", "property", "target", "random_train_test_label"] + gnn_node_attr_cols
    and df[col].dtype in [np.float64, np.int64]
]
all_feature_cols = gnn_node_attr_cols + xgb_core_feature_cols
all_feature_cols = [col for col in all_feature_cols if col in train_df_aug.columns]

# 准备过采样输入
X_train = train_df_aug[all_feature_cols].values
y_train = train_df_aug["target"].values

# 处理单类别情况
if len(np.unique(y_train)) < 2:
    print("⚠️  训练集仅单类别，跳过SMOTE过采样")
    train_df_resampled = train_df_aug.copy()
else:
    # 使用SMOTE平衡样本（兼容所有imblearn版本）
    smote = SMOTE(
        sampling_strategy="auto",
        random_state=SEED,
        k_neighbors=min(5, len(X_train)-1)  # 避免k值超过样本数
    )
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 重构过采样训练集
    train_df_resampled = pd.DataFrame(X_train_resampled, columns=all_feature_cols)
    train_df_resampled["target"] = y_train_resampled
    # 补充非特征列（Smiles等）
    train_df_resampled["Smiles_unify"] = train_df_aug["Smiles_unify"].sample(
        n=len(train_df_resampled), replace=True, random_state=SEED
    ).values
    train_df_resampled["property"] = train_df_aug["property"].sample(
        n=len(train_df_resampled), replace=True, random_state=SEED
    ).values
    train_df_resampled["random_train_test_label"] = "train"

print(f"✅ SMOTE过采样完成：正样本 {sum(train_df_resampled['target'])} 条，负样本 {len(train_df_resampled)-sum(train_df_resampled['target'])} 条")

# 拆分训练/验证集（分层采样）
train_df_final, val_df = train_test_split(
    train_df_resampled,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df_resampled["target"] if len(np.unique(train_df_resampled["target"])) > 1 else None
)
test_df = test_df.reset_index(drop=True)

# 打印最终数据划分
print(f"\n📊 最终数据划分：")
print(f"- 训练集（增强+过采样）：{len(train_df_final)} 条，正样本占比 {train_df_final['target'].mean():.2%}")
print(f"- 验证集：{len(val_df)} 条，正样本占比 {val_df['target'].mean():.2%}")
print(f"- 测试集：{len(test_df)} 条，正样本占比 {test_df['target'].mean():.2%}")

# -------------------------- 2. GNN建模（生成分子嵌入 + 保存模型） --------------------------
print("\n" + "="*60)
print("步骤2：GNN建模（类别权重+批量处理）")
print("="*60)

# 2.1 分子图生成函数
def smiles_to_graph(smiles, node_attr):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 节点特征：原子序数 + 3个理化属性（logP、tpsa、h_bond_donor）
    atom_features = []
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        node_attr_vec = np.array(node_attr, dtype=np.float32)
        atom_feat = np.concatenate([[atom_num], node_attr_vec])
        atom_features.append(atom_feat)
    x = torch.tensor(atom_features, dtype=torch.float32)
    # 边索引（无向图，双向添加）
    edges = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.tensor([[], []], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# 2.2 构建DataLoader（验证/测试集不shuffle）
def create_gnn_dataloader(data_df, batch_size=32, shuffle=False):
    graphs = []
    for idx, row in data_df.iterrows():
        # 填充缺失的节点属性
        node_attr = [row.get(col, 0.0) for col in gnn_node_attr_cols]
        graph = smiles_to_graph(row["Smiles_unify"], node_attr)
        if graph is not None:
            graph.y = torch.tensor([row["target"]], dtype=torch.long)
            graphs.append(graph)
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)

# 创建DataLoader
train_loader = create_gnn_dataloader(train_df_final, batch_size=32, shuffle=True)
val_loader = create_gnn_dataloader(val_df, batch_size=32, shuffle=False)
test_loader = create_gnn_dataloader(test_df, batch_size=32, shuffle=False)

# 2.3 GNN模型定义（拆分嵌入层和分类层，支持输出128维嵌入）
class GNN_MolecularEmbed(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=128, embed_dim=128, num_classes=2):
        super().__init__()
        # 嵌入层（输出128维特征，用于XGBoost融合）
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc_embed = torch.nn.Linear(hidden_channels, embed_dim)
        # 分类层（输出2维，用于二分类）
        self.fc_classify = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x, edge_index, batch=None, return_embed=False):
        """
        return_embed: True返回128维嵌入特征，False返回2维分类结果
        """
        # 图卷积提取特征
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 批量全局平均池化（处理多图批次）
        if batch is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = global_mean_pool(x, batch)
        # 生成嵌入特征
        embed = self.fc_embed(x)
        # 返回嵌入或分类结果
        return embed if return_embed else self.fc_classify(embed)

# 2.4 GNN训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 GNN训练设备：{device}")

# 初始化模型（二分类，num_classes=2）
gnn_model = GNN_MolecularEmbed(in_channels=4, hidden_channels=128, embed_dim=128, num_classes=2).to(device)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001, weight_decay=1e-5)

# 类别权重（平衡正负样本：负样本1.8，正样本1.0）
class_weights = torch.tensor([1.8, 1.0], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
best_val_recall = 0  # 以验证集召回率为指标保存最优模型

# 2.5 训练/评估函数
def train_gnn(epoch):
    gnn_model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # 输出分类结果，计算损失
        out = gnn_model(data.x, data.edge_index, data.batch, return_embed=False)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def eval_gnn(loader):
    gnn_model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # 输出分类结果，预测类别
            out = gnn_model(data.x, data.edge_index, data.batch, return_embed=False)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return recall, f1, y_true, y_pred

# 2.6 训练GNN（共100轮）
print("🔄 开始GNN训练（共100轮）...")
for epoch in range(1, 101):
    train_loss = train_gnn(epoch)
    val_recall, val_f1, _, _ = eval_gnn(val_loader)
    # 保存最优模型（验证集召回率最高）
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        torch.save(gnn_model.state_dict(), GNN_MODEL_SAVE)
    # 每10轮打印日志
    if epoch % 10 == 0:
        print(f"轮次 {epoch:3d} | 训练损失 {train_loss:.4f} | 验证召回率 {val_recall:.4f} | 验证F1 {val_f1:.4f}")

# 加载最优模型并评估测试集
gnn_model.load_state_dict(torch.load(GNN_MODEL_SAVE))
test_recall, test_f1, test_true, test_pred = eval_gnn(test_loader)
print(f"\n✅ GNN测试集评估：召回率 {test_recall:.4f}，F1-score {test_f1:.4f}")
print(f"✅ GNN最优模型已保存：{GNN_MODEL_SAVE}")

# 2.7 提取GNN嵌入（用于XGBoost融合）
def extract_gnn_embeddings(data_df):
    """提取GNN的128维嵌入特征"""
    gnn_model.eval()
    embeddings = []
    loader = create_gnn_dataloader(data_df, batch_size=32, shuffle=False)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # 返回128维嵌入特征
            embed = gnn_model(data.x, data.edge_index, data.batch, return_embed=True)
            embeddings.extend(embed.cpu().numpy())
    # 补充缺失嵌入（极端情况：部分Smiles生成图失败）
    if len(embeddings) < len(data_df):
        embeddings += [np.zeros(128)] * (len(data_df) - len(embeddings))
    return np.array(embeddings[:len(data_df)])

# 提取各数据集的GNN嵌入
train_gnn_embed = extract_gnn_embeddings(train_df_final)
val_gnn_embed = extract_gnn_embeddings(val_df)
test_gnn_embed = extract_gnn_embeddings(test_df)
print(f"✅ GNN嵌入提取完成：128维/样本，覆盖训练/验证/测试集")

# -------------------------- 3. XGBoost建模（纯特征+融合GNN，保存模型） --------------------------
print("\n" + "="*60)
print("步骤3：XGBoost建模（scale_pos_weight优化）")
print("="*60)

# 3.1 准备特征矩阵（两种方案）
# 方案1：仅核心特征（轻量，优先部署）
X_train_1 = train_df_final[xgb_core_feature_cols].values
X_val_1 = val_df[xgb_core_feature_cols].values
X_test_1 = test_df[xgb_core_feature_cols].values

# 方案2：核心特征 + GNN嵌入（高精度）
X_train_2 = np.hstack([X_train_1, train_gnn_embed])
X_val_2 = np.hstack([X_val_1, val_gnn_embed])
X_test_2 = np.hstack([X_test_1, test_gnn_embed])

# 标签（三种数据集共用）
y_train = train_df_final["target"].values
y_val = val_df["target"].values
y_test = test_df["target"].values

# 3.2 XGBoost超参调优（兼容旧版本，移除早停）
def tune_xgboost_optimized(X_train, y_train):
    """XGBoost超参调优：基于交叉验证筛选最优参数"""
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
        "reg_lambda": [0.1, 1],
        "scale_pos_weight": [0.45]  # 正负样本比≈2.2:1，平衡损失
    }

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",  # 替换recall，兼容旧版XGBoost
        random_state=SEED,
        use_label_encoder=False,
        n_jobs=1  # Windows兼容，避免多线程报错
    )

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        cv=3,
        scoring="recall",  # 以召回率为优化目标（避免漏判可渗透分子）
        n_jobs=1,
        verbose=1
    )

    # 仅用训练集拟合，避免数据泄露
    grid_search.fit(X_train, y_train)

    print(f"✅ XGBoost最优超参：{grid_search.best_params_}")
    print(f"✅ XGBoost交叉验证召回率：{grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

# 训练两种XGBoost模型
print("\n【XGBoost方案1：仅核心特征（轻量版）】")
xgb_model1 = tune_xgboost_optimized(X_train_1, y_train)

print("\n【XGBoost方案2：核心特征+GNN嵌入（高精度版）】")
xgb_model2 = tune_xgboost_optimized(X_train_2, y_train)

# 3.3 评估XGBoost模型并保存
def evaluate_xgboost(model, X_test, y_test, model_name):
    """评估XGBoost模型，生成混淆矩阵和报告"""
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n📋 {model_name} 测试集评估报告：")
    print(classification_report(y_test, y_pred, target_names=["不可渗透BBB", "可渗透BBB"]))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["不可渗透", "可渗透"],
                yticklabels=["不可渗透", "可渗透"])
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title(f"{model_name} 混淆矩阵（优化后）")
    plt.tight_layout()
    cm_path = os.path.join(BASE_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 混淆矩阵已保存：{cm_path}")
    return recall, f1

# 评估并保存模型
print("\n【评估方案1：XGBoost_仅核心特征】")
recall1, f11 = evaluate_xgboost(xgb_model1, X_test_1, y_test, "XGBoost_仅核心特征")
xgb_model1.save_model(XGB_CORE_SAVE)
print(f"✅ XGBoost轻量模型已保存：{XGB_CORE_SAVE}")

print("\n【评估方案2：XGBoost_融合GNN】")
recall2, f12 = evaluate_xgboost(xgb_model2, X_test_2, y_test, "XGBoost_融合GNN")
xgb_model2.save_model(XGB_FUSION_SAVE)
print(f"✅ XGBoost高精度模型已保存：{XGB_FUSION_SAVE}")

# -------------------------- 4. 结果汇总与保存 --------------------------
print("\n" + "="*60)
print("步骤4：优化结果汇总")
print("="*60)

# 结果汇总表
result_summary = pd.DataFrame({
    "模型": ["GNN_优化版", "XGBoost_仅核心特征", "XGBoost_融合GNN"],
    "可渗透BBB召回率": [test_recall, recall1, recall2],
    "F1-score": [test_f1, f11, f12],
    "核心优化措施": [
        "损失函数+类别权重+Smiles增强+异常值截断",
        "SMOTE过采样+scale_pos_weight=0.45+异常值截断",
        "GNN嵌入融合+SMOTE过采样+全流程优化"
    ],
    "优化目标": [
        "提升正样本召回率，减少漏判",
        "提升负样本召回率（从0.65→0.75+）",
        "兼顾正负样本，最大化整体F1"
    ]
})

# 保存所有结果文件
# 1. 汇总表
result_summary.to_csv(SUMMARY_SAVE, index=False, encoding="utf-8-sig")

# 2. 测试集预测结果（含所有模型预测）
test_pred_df = test_df[["Smiles_unify", "target"]].copy()
test_pred_df.columns = ["Smiles_unify", "真实标签（1=可渗透）"]
test_pred_df["GNN_预测"] = test_pred
test_pred_df["XGBoost_仅核心特征_预测"] = xgb_model1.predict(X_test_1)
test_pred_df["XGBoost_融合GNN_预测"] = xgb_model2.predict(X_test_2)
test_pred_df.to_csv(TEST_PRED_SAVE, index=False, encoding="utf-8-sig")

# 打印最终结果
print("\n📊 最终优化结果汇总：")
print(result_summary.round(4))

print("\n" + "="*60)
print("✅ 全流程执行完成！所有结果已保存至：")
print(f"   根目录：{BASE_DIR}")
print(f"   1. 模型汇总表：{SUMMARY_SAVE}")
print(f"   2. 测试集预测：{TEST_PRED_SAVE}")
print(f"   3. XGBoost模型：{XGB_CORE_SAVE} / {XGB_FUSION_SAVE}")
print(f"   4. GNN最优模型：{GNN_MODEL_SAVE}")
print("="*60)
print("\n部署建议：")
print(f"- 轻量场景（如网站实时预测）：使用 {XGB_CORE_SAVE}（无需加载GNN，速度快）")
print(f"- 高精度场景：使用 {XGB_FUSION_SAVE}（需加载GNN生成嵌入，精度高）")