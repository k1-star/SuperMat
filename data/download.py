from mp_api.client import MPRester
import pandas as pd

#  Materials Project API Key
API_KEY = "57pQ9t5LvE1C4eKzaTBOfFxJbnm3AKqz"

def download_materials_data():
    with MPRester(API_KEY) as mpr:
        # 下载一部分材料
        docs = mpr.materials.summary.search(
            fields=["material_id", "formula_pretty", "structure", 
                    "band_gap","efermi"],#这里可以选择需要的性质，比如热导率，态密度，弹性模量等，但要在split.py中修改对应列
            num_chunks=1,
            chunk_size=100
        )

    # 转换为 DataFrame
    data = []
    for d in docs:
        data.append({
            "material_id": d.material_id,
            "formula": d.formula_pretty,
            "band_gap": d.band_gap,
            "structure": d.structure,
            "efermi": d.efermi
            
        })

    df = pd.DataFrame(data)
    df.to_pickle("raw_materials.pkl")  # 保存为二进制文件
    print(" 数据下载完成，保存为 raw_materials.pkl")

if __name__ == "__main__":
    download_materials_data()
