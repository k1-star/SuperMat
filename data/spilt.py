import pandas as pd
from sklearn.model_selection import train_test_split
import joblib



def split_dataset():
    df = pd.read_pickle("features.pkl")

    # 目标是 band_gap
    y = df["band_gap"]

    # 丢掉一些非特征列，但保留 efermi
    drop_cols = ["band_gap", "material_id", "formula", "structure", "composition"]
    X = df.drop(columns=drop_cols)

    # 划分训练、验证、测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    datasets = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }

    for name, data in datasets.items():
        data.to_pickle(f"{name}.pkl")
    # 保存训练特征列顺序
    joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")
    # 保存训练时的特征列顺序，预测时需要用到
    print(" 数据集划分完成：训练/验证/测试集已保存")

if __name__ == "__main__":
    split_dataset()

