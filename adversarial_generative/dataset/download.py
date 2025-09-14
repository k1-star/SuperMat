import pandas as pd
import requests

# URL 来自 GitHub release 或 NIMS MDR
url = "https://raw.githubusercontent.com/lfoppiano/supercon/main/data/supercon2_v22.12.03.csv"

def download_supercon2(save_path: str = "supercon2.csv"):
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(resp.content)
        print(f"Downloaded to {save_path}")
    else:
        raise RuntimeError(f"Download failed with status {resp.status_code}")

def load_supercon2(path: str = "supercon2.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    download_supercon2()
    df = load_supercon2()
    print(df.head())
    print("Columns:", df.columns.tolist())
    print("Number of records:", len(df))
