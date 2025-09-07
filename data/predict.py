# predict.py
import joblib
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

def load_model(model_path="baseline_rf.pkl"):
    """加载训练好的模型"""
    return joblib.load(model_path)

def load_feature_columns(columns_path="feature_columns.pkl"):
    """加载训练时保存的特征列顺序"""
    return joblib.load(columns_path)

def featurize_formula(formula, efermi_value, feature_columns):
    """
    将化学式转为特征向量，并加入给定的 efermi
    """
    comp = Composition(formula)
    df = pd.DataFrame({"composition": [comp]})

    # 提取 Magpie 特征
    featurizer = ElementProperty.from_preset("magpie")
    X = featurizer.featurize_dataframe(df, "composition", ignore_errors=True)
    X = X.drop(columns=["composition"])

    # 加入 efermi 列
    if "efermi" in feature_columns:
        X["efermi"] = efermi_value

    # 补齐训练集的其他缺失列
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0

    # 按训练集列顺序排列
    X = X[feature_columns]
    return X

def predict_band_gap(formula, efermi_value, model, feature_columns):
    """预测材料带隙"""
    X = featurize_formula(formula, efermi_value, feature_columns)
    prediction = model.predict(X)[0]
    return prediction

if __name__ == "__main__":
    # 加载模型和训练特征列
    model = load_model("baseline_rf.pkl")
    feature_columns = load_feature_columns("feature_columns.pkl")

    # 示例：预测 LiFePO4 的带隙，假设 efermi 已知为 5.0 eV
    test_formula = "LiFePO4"
    test_efermi = 1.6
    band_gap = predict_band_gap(test_formula, test_efermi, model, feature_columns)
    print(f"🔮 材料 {test_formula} 的预测带隙为 {band_gap:.3f} eV")





