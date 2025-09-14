import pandas as pd
import numpy as np

df = pd.read_csv("supercon2.csv")

# 去掉空格，并尝试去掉 K 单位
df["criticalTemperature"] = df["criticalTemperature"].astype(str).str.strip().str.replace("K","", regex=False)

# 转为浮点数，无法转换的设为 NaN
df["criticalTemperature"] = pd.to_numeric(df["criticalTemperature"], errors='coerce')

# 去掉 NaN 和 Tc <=0 的记录
df = df.dropna(subset=["criticalTemperature"])
df = df[df["criticalTemperature"] > 0]

print(df["criticalTemperature"].head())
df.to_csv("supercon_cleaned.csv", index=False)

import pandas as pd
import re

df = pd.read_csv("supercon_cleaned.csv")
from pymatgen.core import Composition

def safe_composition(formula):
    try:
        return Composition(formula)
    except:
        return None  # 无法解析的返回 None

# 清洗 Tc
df["criticalTemperature"] = df["criticalTemperature"].astype(str).str.strip().str.replace("K","", regex=False)
df["criticalTemperature"] = pd.to_numeric(df["criticalTemperature"], errors='coerce')
df = df.dropna(subset=["criticalTemperature"])
df = df[df["criticalTemperature"] > 0]

# 清洗化学式
def clean_formula(f):
    if pd.isna(f):
        return None
    f = f.replace(" ", "")
    f = re.sub(r"[^A-Za-z0-9\(\)]", "", f)  # 保留字母、数字、括号
    return f if f != "" else None

df["formula_clean"] = df["formula"].apply(clean_formula)

# 构建 composition，并去掉无法解析的行
df["composition"] = df["formula_clean"].apply(safe_composition)
df = df.dropna(subset=["composition"])

print("剩余样本数量:", len(df))
print("无法解析的化学式示例:", df[df["composition"].isna()]["formula"].unique())

df.to_csv("supercon_cleaned.csv", index=False)