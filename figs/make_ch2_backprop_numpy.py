# -*- coding: utf-8 -*-
"""그림 2.2 밑바닥 NumPy 역전파: 학습 손실 곡선 + 결정경계.
본문 코드와 동일한 구현으로 실제 학습시켜 결과를 그린다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for cand in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False


def make_moons_np(n, noise, seed=0):
    rng = np.random.default_rng(seed)
    n_a = n // 2
    n_b = n - n_a
    t_a = np.pi * rng.random(n_a)
    outer = np.c_[np.cos(t_a), np.sin(t_a)]
    t_b = np.pi * rng.random(n_b)
    inner = np.c_[1 - np.cos(t_b), 1 - np.sin(t_b) - 0.5]
    X = np.vstack([outer, inner]) + noise * rng.standard_normal((n, 2))
    y = np.r_[np.zeros(n_a), np.ones(n_b)].reshape(-1, 1)
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# 데이터
X, y = make_moons_np(400, noise=0.20, seed=1)
mu, sd = X.mean(0), X.std(0)
Xs = (X - mu) / sd

# 파라미터 초기화 (He/tanh 스케일)
rng = np.random.default_rng(0)
H = 16
W1 = rng.standard_normal((2, H)) * np.sqrt(1.0 / 2)
b1 = np.zeros((1, H))
W2 = rng.standard_normal((H, 1)) * np.sqrt(1.0 / H)
b2 = np.zeros((1, 1))

lr, epochs = 0.5, 400
N = len(Xs)
losses = []
for ep in range(epochs):
    # 순전파
    z1 = Xs @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    p = sigmoid(z2)
    # 손실 (이진 교차엔트로피)
    eps = 1e-8
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    losses.append(loss)
    # 역전파 (연쇄법칙 수동)
    dz2 = (p - y) / N
    dW2 = a1.T @ dz2
    db2 = dz2.sum(0, keepdims=True)
    da1 = dz2 @ W2.T
    dz1 = da1 * (1 - a1 ** 2)
    dW1 = Xs.T @ dz1
    db1 = dz1.sum(0, keepdims=True)
    # 갱신
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

# 정확도
pred = (p > 0.5).astype(int)
acc = (pred == y).mean()
print(f"final loss={losses[-1]:.4f}  acc={acc*100:.1f}%")

# ── 그림: (a) 손실 곡선  (b) 결정경계 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=120)

ax1.plot(losses, color="#1AB18B", lw=2)
ax1.set_title("(a) 학습 손실", fontsize=12, fontweight="bold")
ax1.set_xlabel("에폭"); ax1.set_ylabel("BCE 손실")
ax1.grid(alpha=0.25)

# 결정경계
gx, gy = np.meshgrid(np.linspace(Xs[:, 0].min()-0.5, Xs[:, 0].max()+0.5, 300),
                     np.linspace(Xs[:, 1].min()-0.5, Xs[:, 1].max()+0.5, 300))
grid = np.c_[gx.ravel(), gy.ravel()]
gp = sigmoid(np.tanh(grid @ W1 + b1) @ W2 + b2).reshape(gx.shape)
ax2.contourf(gx, gy, gp, levels=[0, 0.5, 1], colors=["#eaf3fb", "#fdeeea"], alpha=0.9)
ax2.contour(gx, gy, gp, levels=[0.5], colors="#333", linewidths=1.4)
m0 = (y.ravel() == 0)
ax2.scatter(Xs[m0, 0], Xs[m0, 1], s=14, c="#2c7be5", label="클래스 0", edgecolors="none")
ax2.scatter(Xs[~m0, 0], Xs[~m0, 1], s=14, c="#E8402E", label="클래스 1", edgecolors="none")
ax2.set_title(f"(b) 학습된 결정경계 (정확도 {acc*100:.1f}%)", fontsize=12, fontweight="bold")
ax2.set_xlabel("$x_1$ (표준화)"); ax2.set_ylabel("$x_2$ (표준화)")
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

plt.tight_layout()
out = "ch2_backprop_numpy.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved:", out)
