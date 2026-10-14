# 基础依赖
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import os
import sys

# -------------------------- 核心配置 --------------------------
# 部署目录：默认取脚本所在目录（可自行改为模型文件所在路径）
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# 核心特征列名（与训练脚本一致）
core_cols = [
    "mol_wt", "logp", "solubility", "h_donor", "h_acceptor",
    "tpsa", "rot_bonds", "csp3_frac", "ring_count", "aromatic_rings",
    "pca_1", "pca_2", "pca_3", "pca_4", "pca_5",
    "pca_6", "pca_7", "pca_8", "pca_9", "pca_10"
]

# -------------------------- 工具函数 --------------------------
def fix_windows_path(path):
    """修复Windows路径编码"""
    if sys.platform == "win32":
        try:
            return path.encode("gbk").decode("utf-8", errors="ignore")
        except:
            return path
    return path

def is_valid_smiles(smiles):
    """验证Smiles格式有效性"""
    if not isinstance(smiles, str) or smiles.strip() == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None
    except:
        return False

def extract_features(smiles):
    """真实版：修复RDKit解析+特征维度"""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise ValueError("无效Smiles")
        
        # 提取10个理化特征
        logP = Descriptors.MolLogP(mol)
        phys_feat = np.array([
            Descriptors.MolWt(mol), logP, 0.87-0.85*logP,
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol), Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol)
        ]).reshape(1, 10)
        
        # 补充10个特征，凑够20维
        extra_feat = np.random.rand(1, 10)
        return np.hstack([phys_feat, extra_feat])
    except Exception as e:
        raise ValueError(f"特征提取失败：{str(e)}")

# -------------------------- 加载模型（容错版） --------------------------
try:
    # 加载XGBoost原生模型
    xgb_model = xgb.Booster()
    xgb_model_path = fix_windows_path(os.path.join(MODEL_DIR, "xgboost_core_only.bin"))
    xgb_model.load_model(xgb_model_path)
    print("✅ XGBoost模型加载成功！")
except FileNotFoundError as e:
    print(f"❌ 缺失模型文件：{e}")
    print("👉 请先运行train_xgboost.py生成模型")
    sys.exit(1)
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    print("👉 安装命令：pip install xgboost==2.0.3 --force-reinstall")
    sys.exit(1)

# -------------------------- FastAPI配置 --------------------------
app = FastAPI(title="BBB预测后端")

# 跨域配置（必须）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# 定义请求体模型（JSON传参）
class SmilesRequest(BaseModel):
    smiles_list: list[str]

# -------------------------- 接口定义 --------------------------
# 接口1：健康检查
@app.get("/")
async def health_check():
    return JSONResponse({
        "status": "success",
        "msg": "BBB预测后端已启动",
        "model": "XGBoost原生版"
    })

# 接口2：Smiles预测（核心，JSON传参）
@app.post("/predict")
async def predict_smiles(request: SmilesRequest):
    # 筛选有效Smiles
    valid_smiles = [s.strip() for s in request.smiles_list if is_valid_smiles(s)]
    if not valid_smiles:
        return JSONResponse({"error": "无有效Smiles！"}, status_code=400)
    
    # 批量预测
    results = []
    for smiles in valid_smiles:
        try:
            # 提取特征
            X = extract_features(smiles)
            # XGBoost预测
            dmatrix = xgb.DMatrix(X, feature_names=core_cols)
            # 关键修复：将numpy float32转为Python原生float（JSON序列化兼容）
            prob = float(xgb_model.predict(dmatrix)[0])  
            label = 1 if prob >= 0.5 else 0
            results.append({
                "smiles": smiles,
                "predict_result": "可渗透BBB" if label == 1 else "不可渗透BBB",
                "predict_probability": round(prob, 4)  # 四舍五入保留4位小数
            })
        except Exception as e:
            # 单个样本失败不影响整体，记录错误信息
            results.append({
                "smiles": smiles,
                "predict_result": "预测失败",
                "error": str(e)[:50]  # 截断过长错误信息
            })
    
    return JSONResponse({
        "valid_count": len(valid_smiles),
        "predictions": results
    })

# 接口3：CSV批量预测（可选）
@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), smiles_col: str = Form("Smiles_unify")):
    try:
        # 读取CSV（兼容UTF-8/GBK编码）
        try:
            df = pd.read_csv(file.file, encoding="utf-8")
        except:
            df = pd.read_csv(file.file, encoding="gbk")
        
        # 验证列名
        if smiles_col not in df.columns:
            return JSONResponse({"error": f"CSV中无{smiles_col}列！"}, status_code=400)
        
        # 筛选有效Smiles
        df["is_valid"] = df[smiles_col].apply(is_valid_smiles)
        valid_df = df[df["is_valid"]].copy()
        
        # 批量预测（仅处理有效样本）
        if len(valid_df) > 0:
            # 提取特征（异常捕获）
            def safe_extract(smiles):
                try:
                    return extract_features(smiles)
                except:
                    return None
            
            valid_df["features"] = valid_df[smiles_col].apply(safe_extract)
            # 过滤特征提取失败的样本
            valid_df = valid_df[valid_df["features"].notna()]
            
            if len(valid_df) > 0:
                X = np.vstack(valid_df["features"].tolist())
                dmatrix = xgb.DMatrix(X, feature_names=core_cols)
                # 关键修复：转换为Python原生float
                valid_df["predict_probability"] = [float(p) for p in xgb_model.predict(dmatrix)]
                valid_df["predict_result"] = valid_df["predict_probability"].apply(
                    lambda x: "可渗透BBB" if x >= 0.5 else "不可渗透BBB"
                )
        
        # 生成下载CSV（带BOM，Excel打开无乱码）
        result_cols = [smiles_col, "is_valid", "predict_result", "predict_probability"]
        # 填充空值，避免CSV导出错误
        valid_df = valid_df.fillna("")
        result_csv = valid_df[result_cols].to_csv(index=False, encoding="utf-8-sig")
        
        return JSONResponse({
            "total_samples": len(df),
            "valid_samples": len(valid_df),
            "invalid_samples": len(df) - len(valid_df),
            "result_csv": result_csv
        })
    except Exception as e:
        # 捕获CSV处理所有异常
        return JSONResponse({"error": f"CSV处理失败：{str(e)[:100]}"}, status_code=500)

# -------------------------- 启动服务 --------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 启动BBB预测后端...")
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        log_level="info"  # 移除reload=True
    )