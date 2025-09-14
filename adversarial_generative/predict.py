# tools_supercon.py
import re
import numpy as np
import joblib
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from mcp.types import TextContent

# 载入模型和编码器
model = joblib.load("supercon_model.pkl")
encoder = joblib.load("supercon_encoder.pkl")
featurizer = ElementProperty.from_preset("magpie")

# 支持的类别特征
categorical_cols = ["doping", "materialClass", "shape", "fabrication", "substrate"]

def clean_formula(f):
    if f is None:
        return None
    f = re.sub(r"[^A-Za-z0-9\(\)]", "", f.replace(" ", ""))
    return f if f else None

def predict_supercon(material_info: dict) -> TextContent:
    """
    使用训练好的模型预测材料的临界温度 Tc。
    :param material_info: dict, 包含 keys:
        formula, doping, materialClass, shape, fabrication, substrate
    :return: TextContent, 返回预测结果
    """
    formula = clean_formula(material_info.get("formula"))
    if not formula:
        return TextContent(type="text", text="错误: 无效的化学式。")
    
    try:
        composition = Composition(formula)
    except:
        return TextContent(type="text", text=f"错误: 不能解析化学式 {formula}")

    # Magpie 特征
    X_formula = featurizer.featurize(composition)
    X_formula = np.array([X_formula], dtype=float)

    # 类别特征 one-hot
    cat_values = [material_info.get(col, "Unknown") for col in categorical_cols]
    X_cat = encoder.transform([cat_values])

    # 合并特征
    X = np.hstack([X_formula, X_cat])

    # 预测 Tc
    Tc_pred = model.predict(X)[0]

    result_text = (
        f"材料: {formula}\n"
        f"预测临界温度 Tc: {Tc_pred:.2f} K"
    )
    return TextContent(type="text", text=result_text)
