# clean_data.py
import pandas as pd

def clean_data():
    df = pd.read_pickle("raw_materials.pkl")

    # 删除缺失 band_gap 的行
    df = df.dropna(subset=["band_gap"])

    # 去掉 band_gap < 0 的异常数据
    df = df[df["band_gap"] >= 0]

    df.to_pickle("clean_materials.pkl")
    print(f"✅ 清洗完成，剩余 {len(df)} 条数据")

if __name__ == "__main__":
    clean_data()
