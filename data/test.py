# train_baseline.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # 用于保存模型

def train_baseline():
    # 读取训练、验证、测试集
    X_train = pd.read_pickle("X_train.pkl")
    y_train = pd.read_pickle("y_train.pkl")
    X_val = pd.read_pickle("X_val.pkl")
    y_val = pd.read_pickle("y_val.pkl")
    X_test = pd.read_pickle("X_test.pkl")
    y_test = pd.read_pickle("y_test.pkl")

    # 定义随机森林模型
    model = RandomForestRegressor(
        n_estimators=200,   # 树的数量
        max_depth=20,      # 树的最大深度
        random_state=42,
        n_jobs=-1          # 并行加速
    )

    # 训练
    print("开始训练模型...")
    model.fit(X_train, y_train)

    # 在验证集上评估
    y_val_pred = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    print(f"✅ 验证集结果: MAE={mae_val:.3f}, R²={r2_val:.3f}")

    # 在测试集上评估
    y_test_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"✅ 测试集结果: MAE={mae_test:.3f}, R²={r2_test:.3f}")

    # 保存模型
    joblib.dump(model, "baseline_rf.pkl")
    print("💾模型已保存为 baseline_rf.pkl")

if __name__ == "__main__":
    train_baseline()
##这里我的结果是 验证集结果: MAE=0.476, R²=0.431 测试集结果: MAE=0.513, R²=0.338 模型已保存为 baseline_rf.pkl
##说明预测值和真实值差了0.5eV左右，R²也不高，说明模型还有很大提升空间，特征工程的缺陷比较大，模型也可以继续优化