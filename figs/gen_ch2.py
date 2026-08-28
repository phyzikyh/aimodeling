# -*- coding: utf-8 -*-
"""2장(모델 학습 기본 개념) 개념 그림."""
import numpy as np
import matplotlib.pyplot as plt
from _style import save, C_BLUE, C_RED, C_GREEN, C_PURPLE, C_ORANGE, C_GRAY

rng = np.random.default_rng(2)


# 2.1 손실 지형과 경사하강 경로
def fig_gd_landscape():
    def f(x, y): return 0.5 * (0.18*x**2 + y**2)
    xs = np.linspace(-5, 5, 200); ys = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    cs = ax.contour(X, Y, Z, levels=14, cmap="viridis", linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    # GD 경로
    p = np.array([-4.6, 3.6]); lr = 0.35; path = [p.copy()]
    for _ in range(14):
        g = np.array([0.18*p[0], p[1]])
        p = p - lr * g; path.append(p.copy())
    path = np.array(path)
    ax.plot(path[:,0], path[:,1], "-o", color=C_RED, ms=4, lw=1.8, label="경사하강 경로")
    ax.scatter([0], [0], marker="*", s=180, color=C_GREEN, zorder=5, label="최소점")
    ax.set_xlabel("파라미터 $w_1$"); ax.set_ylabel("파라미터 $w_2$")
    ax.set_title("손실 지형과 경사하강")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "ch2_gd_landscape.png")


# 2.2 편향-분산: 다항식 회귀 과소/적정/과대적합
def fig_bias_variance():
    def true_f(x): return np.sin(1.6*x)
    xn = np.sort(rng.uniform(-2.5, 2.5, 16))
    yn = true_f(xn) + 0.18*rng.standard_normal(16)
    xx = np.linspace(-2.7, 2.7, 300)
    degrees = [(1, "(a) 과소적합 (1차)"), (4, "(b) 적정 (4차)"), (15, "(c) 과대적합 (15차)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, (d, title) in zip(axes, degrees):
        coef = np.polyfit(xn, yn, d)
        yy = np.polyval(coef, xx)
        ax.plot(xx, true_f(xx), color=C_GRAY, lw=1.6, ls="--", label="참 함수")
        ax.plot(xx, yy, color=C_RED, lw=2.0, label="학습된 모델")
        ax.scatter(xn, yn, s=22, color=C_BLUE, zorder=5, label="데이터")
        ax.set_title(title, fontsize=11); ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$"); axes[0].legend(fontsize=8, loc="lower center")
    save(fig, "ch2_bias_variance.png")


# 2.3 옵티마이저 경로 비교 (병든 조건수 이차함수)
def fig_optimizers():
    a, b = 1.0, 9.0
    def grad(p): return np.array([a*p[0], b*p[1]])
    def run(kind, steps=60):
        p = np.array([-4.5, 1.1]); path = [p.copy()]
        v = np.zeros(2); m = np.zeros(2); s = np.zeros(2)
        for t in range(1, steps+1):
            g = grad(p)
            if kind == "sgd":
                p = p - 0.09*g
            elif kind == "momentum":
                v = 0.82*v - 0.03*g; p = p + v
            elif kind == "adam":
                m = 0.9*m + 0.1*g; s = 0.999*s + 0.001*g*g
                mh = m/(1-0.9**t); sh = s/(1-0.999**t)
                p = p - 0.22*mh/(np.sqrt(sh)+1e-8)
            path.append(p.copy())
        return np.array(path)
    xs = np.linspace(-5, 5, 200); ys = np.linspace(-1.6, 1.6, 200)
    X, Y = np.meshgrid(xs, ys); Z = 0.5*(a*X**2 + b*Y**2)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax.contour(X, Y, Z, levels=18, cmap="Greys", linewidths=0.6, alpha=0.55)
    for kind, col, lab in [("sgd", C_BLUE, "SGD"), ("momentum", C_ORANGE, "모멘텀"), ("adam", C_RED, "Adam")]:
        pa = run(kind)
        ax.plot(pa[:,0], pa[:,1], "-o", color=col, ms=2.5, lw=1.6, label=lab)
    ax.scatter([0],[0], marker="*", s=160, color=C_GREEN, zorder=5)
    ax.set_title("옵티마이저별 수렴 경로 (병든 조건수)")
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "ch2_optimizers.png")


# 2.4 초기화별 층 통과 시 활성값 표준편차
def fig_init():
    depth, width, n = 30, 256, 512
    def propagate(std_mode):
        x = rng.standard_normal((n, width))
        stds = []
        for _ in range(depth):
            if std_mode == "bad":
                W = rng.standard_normal((width, width)) * 1.0
            elif std_mode == "he":
                W = rng.standard_normal((width, width)) * np.sqrt(2/width)
            elif std_mode == "small":
                W = rng.standard_normal((width, width)) * 0.02
            x = np.maximum(0, x @ W)          # ReLU
            stds.append(x.std())
        return stds
    fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
    layers = np.arange(1, depth+1)
    ax.plot(layers, propagate("bad"),   color=C_RED,   lw=2, marker="o", ms=3, label="분산 과다 (std=1.0)")
    ax.plot(layers, propagate("small"), color=C_PURPLE,lw=2, marker="s", ms=3, label="분산 과소 (std=0.02)")
    ax.plot(layers, propagate("he"),    color=C_GREEN, lw=2, marker="^", ms=3, label="He 초기화")
    ax.set_yscale("log")
    ax.set_xlabel("층 깊이"); ax.set_ylabel("활성값 표준편차 (로그)")
    ax.set_title("가중치 초기화에 따른 신호 전파")
    ax.legend(fontsize=9, loc="center right")
    save(fig, "ch2_init.png")


# 2.5 평평한 최소 vs 뾰족한 최소 (SAM)
def fig_sam():
    x = np.linspace(-6, 6, 700)
    def loss(shift=0.0):
        sharp = 0.95 * np.exp(-((x + 2.6 - shift)**2) / 0.10)
        flat  = 0.90 * np.exp(-((x - 2.6 - shift)**2) / 3.0)
        return 1.05 - sharp - flat
    train = loss(0.0)
    test = loss(0.5)      # 분포가 살짝 이동한 테스트 손실
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.plot(x, train, color=C_BLUE, lw=2.4, label="훈련 손실")
    ax.plot(x, test, color=C_RED, lw=2.0, ls="--", label="테스트 손실(분포 이동)")
    ax.axvline(-2.6, color=C_GRAY, lw=0.8, ls=":")
    ax.axvline(2.6, color=C_GRAY, lw=0.8, ls=":")
    ax.annotate("뾰족한 최소\n(테스트 손실 급증 → 일반화 나쁨)",
                xy=(-2.6, loss(0)[np.argmin(np.abs(x+2.6))]),
                xytext=(-5.9, 0.55), fontsize=9, color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c"))
    ax.annotate("평평한 최소\n(이동에도 안정 → 일반화 좋음)",
                xy=(2.6, loss(0)[np.argmin(np.abs(x-2.6))]),
                xytext=(0.3, 0.62), fontsize=9, color="#166534",
                arrowprops=dict(arrowstyle="->", color="#166534"))
    ax.set_xlabel("파라미터 $\\theta$"); ax.set_ylabel("손실")
    ax.set_title("뾰족한 최소 vs 평평한 최소 (SAM의 동기)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", fontsize=9)
    save(fig, "ch2_sam_minima.png")


if __name__ == "__main__":
    fig_gd_landscape()
    fig_bias_variance()
    fig_optimizers()
    fig_init()
    fig_sam()
    print("done ch2")
