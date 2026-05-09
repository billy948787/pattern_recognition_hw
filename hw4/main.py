import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

np.random.seed(42)


# True function


def f_true(x):

    return np.sin(2 * np.pi * x)


x_test = np.linspace(0, 1, 200)

y_test_true = f_true(x_test)


# ============================================================

# (a) Generate data

# ============================================================


def generate_data(n=20, noise=0.3):

    x = np.sort(np.random.uniform(0, 1, n))

    y = f_true(x) + np.random.randn(n) * noise

    return x, y


x_train, y_train = generate_data()


# ============================================================

# (b) Polynomial degree comparison (no regularization)

# ============================================================

degrees = [1, 3, 5, 9]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_test, y_test_true, "k-", linewidth=2, label="True f(x)")
ax.scatter(x_train, y_train, color="black", s=30, zorder=5, label="Training points")

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for d, c in zip(degrees, colors):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0)),
    ])
    pipe.fit(x_train.reshape(-1, 1), y_train)
    y_pred = pipe.predict(x_test.reshape(-1, 1))
    ax.plot(x_test, y_pred, color=c, linewidth=1.5, label=f"degree={d}")

ax.set_ylim(-3, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.suptitle("(b) Polynomial Degree Comparison (no regularization)", fontsize=14)

plt.tight_layout()

# plt.savefig("week4_degree.png", dpi=150)

plt.show()


# ============================================================

# (c) Regularization comparison (degree=9)

# ============================================================

alphas = [0, 1e-6, 1e-2, 1.0]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_test, y_test_true, "k-", linewidth=2, label="True f(x)")
ax.scatter(x_train, y_train, color="black", s=30, zorder=5, label="Training points")

colors_c = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for alpha, c in zip(alphas, colors_c):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=9, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    pipe.fit(x_train.reshape(-1, 1), y_train)
    y_pred = pipe.predict(x_test.reshape(-1, 1))
    ax.plot(x_test, y_pred, color=c, linewidth=1.5, label=f"λ={alpha}")

ax.set_ylim(-3, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title("(c) Ridge Regression Comparison (degree=9)", fontsize=14)

plt.tight_layout()
# plt.savefig("week4_ridge.png", dpi=150)
plt.show()


# ============================================================

# (d) Bias-variance simulation

# ============================================================

n_runs = 100

preds_lam0 = np.zeros((n_runs, len(x_test)))
preds_lam1e2 = np.zeros((n_runs, len(x_test)))

for i in range(n_runs):
    x_tr, y_tr = generate_data()

    for lam, store in [(0, preds_lam0), (1e-2, preds_lam1e2)]:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=9, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=lam)),
        ])
        pipe.fit(x_tr.reshape(-1, 1), y_tr)
        store[i] = pipe.predict(x_test.reshape(-1, 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, preds, lam_label in zip(
    axes,
    [preds_lam0, preds_lam1e2],
    ["λ=0", "λ=1e-2"],
):
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    ax.plot(x_test, y_test_true, "k-", linewidth=2, label="True f(x)")
    ax.plot(x_test, mean_pred, "b-", linewidth=2, label="Mean prediction (bias)")
    ax.fill_between(
        x_test,
        mean_pred - std_pred,
        mean_pred + std_pred,
        alpha=0.3,
        color="blue",
        label="±1 std (variance)",
    )
    ax.set_ylim(-3, 3)
    ax.set_title(f"(d) Bias-Variance: {lam_label}", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

plt.tight_layout()
# plt.savefig("week4_bias_variance.png", dpi=150)
plt.show()

print("Interpretation:")
print(
    "λ=0 (no regularization): 高 variance（每次訓練集不同，預測曲線大幅震盪），"
    "bias 較低（平均預測接近真實函數），但過擬合嚴重，曲線間差異極大。"
)
print(
    "λ=1e-2 (Ridge): variance 明顯降低（各次預測曲線更穩定、陰影帶變窄），"
    "bias 略微上升（平均曲線與真實函數稍有偏差），整體泛化能力更好。"
)
print(
    "這體現了 bias-variance tradeoff：正則化壓縮模型複雜度，"
    "以少量 bias 換取大幅 variance 的減少，使測試誤差更低。"
)
