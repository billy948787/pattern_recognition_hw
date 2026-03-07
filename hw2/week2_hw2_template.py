import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# 1. 生成數據
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**3 + X**2 - 2 * X + 1 + np.random.normal(0, 3, 100).reshape(-1, 1)

# 2. 切分數據 (60% 訓練, 20% 驗證, 20% 測試)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
# 2.5 視覺化資料分佈（依 Train / Val / Test 標示）
plt.figure(figsize=(8, 5))

plt.scatter(
    X_train, y_train, color="steelblue", alpha=0.7, label=f"Train ({len(X_train)} pts)"
)
plt.scatter(
    X_val, y_val, color="orange", alpha=0.7, label=f"Validation ({len(X_val)} pts)"
)
plt.scatter(X_test, y_test, color="green", alpha=0.7, label=f"Test ({len(X_test)} pts)")

# 畫出真實的底層函數（無噪聲）
X_true = np.linspace(-3, 3, 300).reshape(-1, 1)
y_true = 0.5 * X_true**3 + X_true**2 - 2 * X_true + 1
plt.plot(
    X_true, y_true, color="red", linewidth=2, linestyle="--", label="True function"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Data Distribution (Train / Validation / Test)")
plt.legend()
plt.tight_layout()
plt.savefig("data_distribution.png")
plt.show()


# 3. 迭代不同次數，計算訓練與驗證誤差
degrees = [1, 2, 3, 5, 9, 15]

train_errors = []
val_errors = []

for degree in degrees:
    # 創建多項式特徵
    poly_features = PolynomialFeatures(degree)

    # 訓練線性迴歸模型
    model = LinearRegression()

    # 預測並計算誤差 mean_squared_error
    pipeline = make_pipeline(poly_features, model)
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)

    train_error = mean_squared_error(y_train, y_train_pred)
    val_error = mean_squared_error(y_val, y_val_pred)

    train_errors.append(train_error)
    val_errors.append(val_error)

    print(
        f"Degree: {degree}, Train Error: {train_error:.4f}, Validation Error: {val_error:.4f}"
    )


# 4. 繪製誤差曲線
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, marker="o", label="Train MSE")
plt.plot(degrees, val_errors, marker="s", label="Validation MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Training vs Validation Error")
plt.legend()
plt.xticks(degrees)
plt.yscale("log")  # 用 log scale 讓差距更清楚
plt.tight_layout()
plt.savefig("error_curve.png")
plt.show()

# 4.5 畫出每個 degree 的預測曲線
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
y_true = 0.5 * X_plot**3 + X_plot**2 - 2 * X_plot + 1

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, degree in enumerate(degrees):
    pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    pipeline.fit(X_train, y_train)

    y_plot_pred = pipeline.predict(X_plot)
    train_mse = train_errors[i]
    val_mse = val_errors[i]

    ax = axes[i]

    # 畫資料點
    ax.scatter(X_train, y_train, color="steelblue", alpha=0.5, s=20, label="Train")
    ax.scatter(X_val, y_val, color="orange", alpha=0.5, s=20, label="Val")

    # 畫真實函數
    ax.plot(X_plot, y_true, color="red", linewidth=1.5, linestyle="--", label="True")

    # 畫預測曲線，限制 y 軸避免 degree 高時爆炸
    ax.plot(X_plot, y_plot_pred, color="purple", linewidth=2, label=f"Pred")

    ax.set_title(f"Degree {degree}\nTrain MSE={train_mse:.1f}  Val MSE={val_mse:.1f}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-15, 20)  # 固定 y 軸範圍，高 degree 才不會爆出去
    ax.legend(fontsize=7)
    ax.set_xlabel("X")
    ax.set_ylabel("y")

plt.suptitle("Polynomial Fit for Each Degree", fontsize=14)
plt.tight_layout()
plt.savefig("fit_curves.png")
plt.show()


# 5. 找出最佳次數
best_degree = degrees[np.argmin(val_errors)]
print(f"\n最佳多項式次數（依驗證集 MSE）: degree = {best_degree}")
# =======================================================

# 5-fold cross validation

print("\n--- 5-Fold Cross Validation ---")
cv_scores = {}

for degree in degrees:
    pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # cross_val_score 預設用 R²，加上 neg_mean_squared_error 取得 MSE
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    cv_mse = -scores.mean()
    cv_scores[degree] = cv_mse
    print(f"Degree {degree:2d} | CV MSE: {cv_mse:.4f} ± {scores.std():.4f}")

# 5-fold cross validation 的最佳次數
best_degree_cv = min(cv_scores, key=cv_scores.get)
print(f"\n最佳多項式次數（依 5-fold CV MSE）: degree = {best_degree_cv}")
