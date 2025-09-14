import pandas as pd
import numpy as np
import re
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# -----------------------------
# 配置
# -----------------------------
DATA_PATH = "supercon_cleaned.csv"   # 你的 CSV 文件路径
MODEL_PATH = "supercon_model.pkl"
ENCODER_PATH = "supercon_encoder.pkl"

categorical_cols = ["doping", "materialClass", "shape", "fabrication", "substrate"]

# -----------------------------
# 辅助函数
# -----------------------------
def clean_formula(f):
    """清洗化学式"""
    if pd.isna(f):
        return None
    f = re.sub(r"[^A-Za-z0-9\(\)]", "", f.replace(" ", ""))
    return f if f else None

def parse_composition(formula):
    """将化学式转为 Composition 对象"""
    if formula is None:
        return None
    try:
        return Composition(formula)
    except:
        return None

# -----------------------------
# 读取数据
# -----------------------------
df = pd.read_csv(DATA_PATH)

# 清理 Tc 列
df["criticalTemperature"] = pd.to_numeric(df["criticalTemperature"], errors="coerce")
df = df[df["criticalTemperature"] > 0].reset_index(drop=True)

# 清理化学式
df["formula_clean"] = df["formula"].apply(clean_formula)
df["composition"] = df["formula_clean"].apply(parse_composition)
df = df[~df["composition"].isna()].reset_index(drop=True)

print(f"有效数据条数: {len(df)}")

# -----------------------------
# 公式特征
# -----------------------------
featurizer = ElementProperty.from_preset("magpie")
X_formula = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=True)

# -----------------------------
# 类别特征
# -----------------------------
df_cat = df[categorical_cols].fillna("Unknown").astype(str)
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_cat = encoder.fit_transform(df_cat)

# 保存 encoder
joblib.dump(encoder, ENCODER_PATH)

# -----------------------------
# 特征拼接
# -----------------------------
X = np.hstack([X_formula.values, X_cat])
y = df["criticalTemperature"].values

# -----------------------------
# 划分训练集和测试集
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 训练模型
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -----------------------------
# 保存模型
# -----------------------------
joblib.dump(model, MODEL_PATH)

# -----------------------------
# 测试模型
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
print("训练完成，模型已保存！")
