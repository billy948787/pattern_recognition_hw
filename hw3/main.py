import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# (a) 資料切分 & 模型訓練
# ─────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

clf = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=10_000, random_state=42)),
    ]
)
clf.fit(X_train, y_train)

# ─────────────────────────────────────────
# (b) Posterior P̂(C=1 | x)
# ─────────────────────────────────────────

p_hat = clf.predict_proba(X_test)[:, 1]  # shape: (n_test,)
print(p_hat)

# ─────────────────────────────────────────
# (c) 兩種決策規則
# ─────────────────────────────────────────

# 規則 1：0/1 loss  →  ŷ = 1 iff p̂ > 0.5
y_rule1 = (p_hat > 0.5).astype(int)

# 規則 2：成本敏感
# 損失矩陣：λ10=5（把良性誤判為惡性，FP）, λ01=1（把惡性漏掉，FN）
# R(α1|x) = λ10 · (1-p̂)   →  decide C=1 的 risk
# R(α0|x) = λ01 · p̂       →  decide C=0 的 risk
# 決策邊界：R(α1) < R(α0)  ↔  p̂ > λ10/(λ10+λ01)
lam10, lam01 = 5, 1
# 如果P̂ > threshold2，則決定 C=1；反之則決定 C=0
threshold2 = lam10 / (lam10 + lam01)  # = 5/6 ≈ 0.8333

R_alpha1 = lam10 * (1 - p_hat)
R_alpha0 = lam01 * p_hat
# 如果 R(α1) < R(α0)，則決定 C=1；反之則決定 C=0
y_rule2 = (R_alpha1 < R_alpha0).astype(int)

cm1 = confusion_matrix(y_test, y_rule1)
cm2 = confusion_matrix(y_test, y_rule2)


def _metrics(cm, n):
    tn, fp, fn, tp = cm.ravel()
    return dict(
        acc=(tn + tp) / n, prec=tp / (tp + fp), rec=tp / (tp + fn), fp=fp, fn=fn
    )


n = len(y_test)
m1 = _metrics(cm1, n)
m2 = _metrics(cm2, n)

print("=" * 55)
print(f"Test set: {n} samples  |  positives (malignant): {y_test.sum()}")
print("=" * 55)

print(f"\n【Rule 1】0/1 loss  (threshold = 0.50)")
print(f"  Confusion matrix:\n{cm1}")
print(f"  Accuracy={m1['acc']:.4f}  Precision={m1['prec']:.4f}  Recall={m1['rec']:.4f}")
print(f"  FP={m1['fp']}  FN={m1['fn']}")

print(f"\n【Rule 2】Cost-sensitive  λ10={lam10}, λ01={lam01}")
print(f"  threshold = λ10/(λ10+λ01) = {threshold2:.4f}")
print(f"  Confusion matrix:\n{cm2}")
print(f"  Accuracy={m2['acc']:.4f}  Precision={m2['prec']:.4f}  Recall={m2['rec']:.4f}")
print(f"  FP={m2['fp']}  FN={m2['fn']}")

# ─────────────────────────────────────────
# (d) Reject option
# ─────────────────────────────────────────
# 拒絕條件：min(R(α0|x), R(α1|x)) > λR
# 等價於：λ01·p̂ > λR  且  λ10·(1-p̂) > λR
# → p̂ ∈ ( λR/λ01/(λR/λ01+1) , 1 - λR/λ10/(λR/λ10+1) )
# 化簡後：p̂ ∈ (λR/(λR+λ10), λ10/(λ10+λR))   [此例即 (0.074, 0.926)]
lam_R = 0.4
min_risk = np.minimum(R_alpha0, R_alpha1)
reject_mask = min_risk > lam_R

reject_rate = reject_mask.mean()
n_reject = reject_mask.sum()
n_accept = (~reject_mask).sum()

# Non-reject subset：沿用 Rule 2 的決策
y_test_nr = y_test[~reject_mask]
y_pred_nr = y_rule2[~reject_mask]
cm_nr = confusion_matrix(y_test_nr, y_pred_nr)
m_nr = _metrics(cm_nr, n_accept)

rj_lo = lam_R / (lam01 + lam_R)  # = 0.0741
rj_hi = lam10 / (lam10 + lam_R)  # = 0.9259

print(f"\n【Rule 2 + Reject】λR={lam_R}")
print(f"  Reject zone: p̂ ∈ ({rj_lo:.4f}, {rj_hi:.4f})")
print(f"  Rejected: {n_reject}/{n} = {reject_rate:.4f} ({reject_rate * 100:.1f}%)")
print(f"  Accepted: {n_accept}/{n}")
print(f"\n  Confusion matrix (non-reject, n={n_accept}):\n{cm_nr}")
print(
    f"  Accuracy={m_nr['acc']:.4f}  Precision={m_nr['prec']:.4f}  Recall={m_nr['rec']:.4f}"
)
print(f"  FP={m_nr['fp']}  FN={m_nr['fn']}")
print("=" * 55)
