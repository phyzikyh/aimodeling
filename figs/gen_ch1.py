# -*- coding: utf-8 -*-
"""1장(AI 모델 학습 개요) 개념 그림."""
import numpy as np
import matplotlib.pyplot as plt
from _style import save, C_BLUE, C_RED, C_GREEN, C_PURPLE, C_GRAY

rng = np.random.default_rng(1)


# 1.1 활성화 함수
def fig_activations():
    x = np.linspace(-5, 5, 400)
    relu = np.maximum(0, x)
    sig = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(x, relu, color=C_BLUE, lw=2.4, label="ReLU  $\\max(0,x)$")
    ax.plot(x, sig, color=C_RED, lw=2.4, label="Sigmoid  $\\sigma(x)$")
    ax.plot(x, tanh, color=C_GREEN, lw=2.4, label="Tanh")
    ax.axhline(0, color="#9aa5b1", lw=0.8); ax.axvline(0, color="#9aa5b1", lw=0.8)
    ax.set_xlabel("입력 $x$"); ax.set_ylabel("출력 $\\phi(x)$")
    ax.set_title("대표적 비선형 활성화 함수")
    ax.set_ylim(-1.4, 3.0)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "ch1_activations.png")


# 1.2 make_moons 데이터와 선형 vs 비선형 결정경계
def make_moons(n=400, noise=0.18):
    n1 = n // 2; n2 = n - n1
    t1 = np.linspace(0, np.pi, n1)
    m1 = np.c_[np.cos(t1), np.sin(t1)]
    t2 = np.linspace(0, np.pi, n2)
    m2 = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5]
    X = np.vstack([m1, m2]) + noise * rng.standard_normal((n, 2))
    y = np.r_[np.zeros(n1), np.ones(n2)].astype(int)
    return X, y


def fig_moons_boundary():
    X, y = make_moons()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), constrained_layout=True)
    for ax in axes:
        ax.scatter(X[y==0,0], X[y==0,1], s=12, color=C_BLUE, alpha=0.7, label="클래스 0")
        ax.scatter(X[y==1,0], X[y==1,1], s=12, color=C_RED, alpha=0.7, label="클래스 1")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
    # (a) 선형 경계: 직선 하나로는 못 나눔
    xs = np.linspace(-1.6, 2.6, 50)
    axes[0].plot(xs, 0.6 - 0.55*xs, color=C_GRAY, lw=2.2, ls="--")
    axes[0].set_title("(a) 선형 경계로는 분리 불가")
    # (b) 비선형(곡선) 경계
    xs2 = np.linspace(-1.6, 2.6, 200)
    axes[1].plot(xs2, 0.55 + 0.9*np.sin(1.6*xs2 - 0.3), color=C_GREEN, lw=2.4)
    axes[1].set_title("(b) 비선형 경계로 분리")
    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.set_xlim(-1.7, 2.7); ax.set_ylim(-1.3, 1.7)
    save(fig, "ch1_moons_boundary.png")


# 1.3 학습 곡선 (손실 / 정확도)
def fig_learning_curve():
    ep = np.arange(1, 51)
    tr_loss = 0.7 * np.exp(-ep/12) + 0.05 + 0.01*rng.standard_normal(50)
    va_loss = 0.7 * np.exp(-ep/14) + 0.12 + 0.02*rng.standard_normal(50)
    va_loss[25:] += np.linspace(0, 0.06, 25)   # 후반 살짝 증가(과적합 기미)
    tr_acc = 1 - 0.5*np.exp(-ep/10) - 0.01*rng.standard_normal(50)
    va_acc = 1 - 0.5*np.exp(-ep/12) - 0.06 - 0.01*rng.standard_normal(50)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    axes[0].plot(ep, tr_loss, color=C_BLUE, lw=2, label="훈련")
    axes[0].plot(ep, va_loss, color=C_RED, lw=2, label="검증")
    axes[0].set_title("손실 곡선"); axes[0].set_xlabel("에폭"); axes[0].set_ylabel("손실")
    axes[1].plot(ep, tr_acc, color=C_BLUE, lw=2, label="훈련")
    axes[1].plot(ep, va_acc, color=C_RED, lw=2, label="검증")
    axes[1].set_title("정확도 곡선"); axes[1].set_xlabel("에폭"); axes[1].set_ylabel("정확도")
    for ax in axes:
        ax.legend(fontsize=9)
    save(fig, "ch1_learning_curve.png")


if __name__ == "__main__":
    fig_activations()
    fig_moons_boundary()
    fig_learning_curve()
    print("done ch1")
