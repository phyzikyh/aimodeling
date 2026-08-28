# -*- coding: utf-8 -*-
"""4장(Transformer) 개념 그림."""
import numpy as np
import matplotlib.pyplot as plt
from _style import save, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_GRAY

rng = np.random.default_rng(4)


# 4.1 사인·코사인 위치 인코딩 히트맵
def fig_positional():
    T, d = 60, 64
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    div = np.exp((i // 2) * (-np.log(10000.0) / (d // 2)))
    pe = np.where(i % 2 == 0, np.sin(pos * div), np.cos(pos * div))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.grid(False)
    im = ax.imshow(pe, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xlabel("임베딩 차원 인덱스")
    ax.set_ylabel("시퀀스 위치 (pos)")
    ax.set_title("사인·코사인 위치 인코딩")
    fig.colorbar(im, ax=ax, shrink=0.85, label="PE 값")
    save(fig, "ch4_positional.png")


# 4.2 어텐션 가중치 히트맵 (예시)
def fig_attention():
    L = 12
    idx = np.arange(L)
    # 대각선 근처 + 특정 위치에 주목하는 구조를 합성
    base = np.exp(-0.25 * (idx[:, None] - idx[None, :])**2)
    base[:, 2] += 0.6            # 2번 위치(예: 중요한 토큰)에 집중
    base[:, 9] += 0.4
    scores = base + 0.05 * rng.random((L, L))
    A = scores / scores.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    ax.grid(False)
    im = ax.imshow(A, cmap="viridis")
    ax.set_xlabel("키 위치 (참고 대상)")
    ax.set_ylabel("쿼리 위치 (질문)")
    ax.set_title("어텐션 가중치 행렬 (예시)")
    ax.set_xticks(idx); ax.set_yticks(idx)
    fig.colorbar(im, ax=ax, shrink=0.85, label="가중치 α")
    save(fig, "ch4_attention.png")


# 4.3 스케일링 효과: sqrt(d_k)로 나누기 전/후 소프트맥스
def fig_scaling():
    dk = 128
    q = rng.standard_normal(dk); K = rng.standard_normal((10, dk))
    scores = K @ q
    def softmax(s): e = np.exp(s - s.max()); return e / e.sum()
    w_raw = softmax(scores)
    w_scaled = softmax(scores / np.sqrt(dk))
    pos = np.arange(10)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True, sharey=True)
    axes[0].bar(pos, w_raw, color=C_RED, alpha=0.85)
    axes[0].set_title("(a) 스케일링 없음\n한 위치에 쏠림(포화)")
    axes[1].bar(pos, w_scaled, color=C_BLUE, alpha=0.85)
    axes[1].set_title("(b) $\\sqrt{d_k}$로 스케일링\n적절히 분산")
    for ax in axes:
        ax.set_xlabel("키 위치"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("어텐션 가중치")
    save(fig, "ch4_scaling.png")


# 4.4 계산 복잡도: RNN O(L) vs 셀프 어텐션 O(L^2)
def fig_complexity():
    L = np.arange(1, 65)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(L, L, color=C_BLUE, lw=2.4, label="순환 신경망  $O(L)$")
    ax.plot(L, L**2, color=C_RED, lw=2.4, label="셀프 어텐션  $O(L^2)$")
    ax.set_xlabel("시퀀스 길이 $L$")
    ax.set_ylabel("연산량 (상대)")
    ax.set_title("시퀀스 길이에 따른 연산량")
    ax.legend(fontsize=10, loc="upper left")
    save(fig, "ch4_complexity.png")


if __name__ == "__main__":
    fig_positional()
    fig_attention()
    fig_scaling()
    fig_complexity()
    print("done ch4")
