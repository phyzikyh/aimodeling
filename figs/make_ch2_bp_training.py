# -*- coding: utf-8 -*-
"""2장 역전파 절: 2-2-2 망을 NumPy 스크래치로 학습(XOR) → 손실 곡선.
본문 코드와 동일 구현으로 실제 학습시켜 검증한다."""
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# XOR: 입력 2 → 출력 2(원-핫: [XOR아님, XOR])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
D = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], float)

rng = np.random.default_rng(0)
W1 = rng.normal(0, 1, (2, 2)); b1 = np.zeros(2)   # W1[j,i]: h_j <- x_i
W2 = rng.normal(0, 1, (2, 2)); b2 = np.zeros(2)   # W2[k,j]: o_k <- h_j
eta = 0.5

losses = []
for epoch in range(3000):
    tot = 0.0
    for x, d in zip(X, D):
        # ── 순전파 ──
        v = W1 @ x + b1;  h = sigmoid(v)
        y = W2 @ h + b2;  o = sigmoid(y)
        tot += 0.5 * np.sum((d - o) ** 2)
        # ── 역전파(일반화 델타 규칙) ──
        delta2 = (d - o) * o * (1 - o)            # 출력층 δ
        delta1 = (W2.T @ delta2) * h * (1 - h)    # 은닉층 δ (오차 역전파)
        # ── 갱신 ──
        W2 += eta * np.outer(delta2, h); b2 += eta * delta2
        W1 += eta * np.outer(delta1, x); b1 += eta * delta1
    losses.append(tot / len(X))

# 예측 확인
print("최종 손실:", round(losses[-1], 5))
for x, d in zip(X, D):
    h = sigmoid(W1 @ x + b1); o = sigmoid(W2 @ h + b2)
    print(f"  x={x.astype(int)}  o={o.round(3)}  예측={int(np.argmax(o))}  정답={int(np.argmax(d))}")

fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=130)
ax.plot(losses, color="#1AB18B", lw=2)
ax.set_title("2-2-2 망의 XOR 학습 손실", fontsize=12, fontweight="bold")
ax.set_xlabel("에폭"); ax.set_ylabel(r"평균 손실  $\frac{1}{2}\sum(d-o)^2$")
ax.grid(alpha=0.25)
ax.set_xlim(0, len(losses))
fig.savefig("ch2_bp_training.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved: ch2_bp_training.png")
