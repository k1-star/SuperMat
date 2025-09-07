import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

def feature_engineering():
    df = pd.read_pickle("clean_materials.pkl")

    # 用 Composition 转换 formula
    df["composition"] = df["formula"].map(Composition)

    # 使用 matminer 提取特征（元素属性）
    featurizer = ElementProperty.from_preset(preset_name="magpie")
    features = featurizer.featurize_dataframe(df, "composition")

    features.to_pickle("features.pkl")
    print(f"特征工程完成，得到 {features.shape[1]} 维特征，保存为 features.pkl")

if __name__ == "__main__":
    feature_engineering()
"""这里的特征化是从化学式出发，提取元素属性特征，原理比较原始，
真正涉及到前沿材料比如热电材料的特征化，可能需要从材料的结构、电子态密度、声子谱等多方面入手，提取更有意义的特征。
这部分内容可以后续再补充。
另外，matminer 还提供了很多其他的 featurizer，可以尝试不同的方法提取特征。
"""