# NeuroGate-BBB · 血脑屏障渗透预测平台

基于 **GNN + XGBoost** 的血脑屏障（Blood-Brain Barrier, BBB）渗透性预测项目，附 FastAPI 推理服务与网页界面。

血脑屏障是药物能否进入中枢神经系统的"守门员"。本项目对 8301 个分子样本进行特征工程（理化性质 + Morgan 指纹 + PCA 降维），构建图神经网络与 XGBoost 集成模型，并封装为可上传 SMILES 的网页预测服务。

## 目录结构

| 文件 | 说明 |
|---|---|
| `data_preprocessing.py` | 数据清洗与预处理 |
| `feature_engineering.py` | 特征工程（RDKit 描述符 + Morgan 指纹 + 相关性过滤 + PCA） |
| `model_training.py` | GNN 模型训练 |
| `train_xgboost.py` | XGBoost 模型训练 |
| `backend_min.py` | FastAPI 推理服务（自动使用脚本所在目录的模型文件） |
| `index.html` | 网页预测界面 |
| `*.pth` / `*.bin` / `*.joblib` | 已训练的模型与预处理组件 |
| `bbb_feature_engineering_report.md` | 特征工程报告（流程与筛选结果） |

## 快速开始

```bash
pip install fastapi uvicorn pandas numpy xgboost joblib rdkit scikit-learn torch
uvicorn backend_min:app --host 0.0.0.0 --port 8000
# 浏览器打开 index.html 或 http://127.0.0.1:8000
```

## 说明

- 模型文件路径已改为可移植写法（脚本所在目录）。
- 原始训练数据未包含在本仓库。

## 许可证

MIT © 2026 [ahdezneytiry-boop](https://github.com/ahdezneytiry-boop)
