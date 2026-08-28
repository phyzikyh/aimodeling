# -*- coding: utf-8 -*-
"""7장(생성 모델) 개념 그림 생성 스크립트.
numpy로 합성한 '개념 예시' 데이터로 matplotlib 정적 PNG를 만든다.
라벨은 영어/수식으로 두어 폰트 문제를 피하고, 한글 설명은 qmd 캡션에 둔다.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

os.makedirs(os.path.dirname(__file__), exist_ok=True)
OUT = os.path.dirname(__file__)

plt.rcParams.update({
    "font.family": "Malgun Gothic",   # 그림 속 한글
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#4a5568",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
C_REAL = "#2c7be5"   # 실제 데이터/분포
C_GEN  = "#e5533c"   # 생성/가짜
C_MAN  = "#2f9e44"   # 매니폴드
C_ACC  = "#8957e5"   # 강조

rng = np.random.default_rng(7)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("saved", name, os.path.getsize(p), "bytes")


# ---------------------------------------------------------------
# Fig 7.1  매니폴드 가설: 3D swiss roll + 2D 내재 좌표
# ---------------------------------------------------------------
def fig_manifold():
    n = 1500
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))     # 내재 좌표 1
    h = 21 * rng.random(n)                          # 내재 좌표 2
    x = t * np.cos(t)
    z = t * np.sin(t)
    fig = plt.figure(figsize=(9, 3.9), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(x, h, z, c=t, cmap="viridis", s=6, alpha=0.8)
    ax1.set_title("(a) 3D 공간에 놓인 데이터")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.view_init(elev=12, azim=-72)
    ax1.grid(False)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(t, h, c=t, cmap="viridis", s=6, alpha=0.85)
    ax2.set_title("(b) 실제로는 2D 매니폴드 위에 분포")
    ax2.set_xlabel("intrinsic coord 1  (t)")
    ax2.set_ylabel("intrinsic coord 2  (h)")
    save(fig, "ch7_manifold.png")


# ---------------------------------------------------------------
# Fig 7.2  AE vs VAE 잠재공간: 이산적 군집(구멍) vs 연속적 분포
# ---------------------------------------------------------------
def fig_latent_ae_vae():
    K = 6
    centers = np.array([[np.cos(2*np.pi*k/K), np.sin(2*np.pi*k/K)] for k in range(K)]) * 3.2
    n_per = 350
    # (a) AE: 중심에서 멀리 떨어진 촘촘한 군집 -> 사이에 빈 공간(구멍)
    ae = [c + rng.normal(0, 0.35, (n_per, 2)) for c in centers]
    # (b) VAE: 표준정규로 당겨져 서로 겹치며 공간을 연속적으로 채움
    va = [0.9*rng.normal(0, 1.0, (n_per, 2)) + 0.55*c for c in centers]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for k in range(K):
        axes[0].scatter(ae[k][:,0], ae[k][:,1], s=7, color=cmap(k), alpha=0.7)
        axes[1].scatter(va[k][:,0], va[k][:,1], s=7, color=cmap(k), alpha=0.6)
    axes[0].set_title("(a) 일반 오토인코더 잠재공간")
    axes[1].set_title("(b) VAE 잠재공간")
    for ax in axes:
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
        ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
        ax.set_aspect("equal")
    axes[0].annotate("빈 공간(구멍)\n→ 디코딩 시 이상 샘플",
                     xy=(0,0), xytext=(0.0,-4.9), ha="center", fontsize=9,
                     color="#b91c1c",
                     arrowprops=dict(arrowstyle="->", color="#b91c1c"))
    axes[1].annotate("연속적으로 채워짐\n→ 매끄러운 생성",
                     xy=(0,0), xytext=(0.0,-4.9), ha="center", fontsize=9,
                     color="#166534")
    save(fig, "ch7_latent_ae_vae.png")


# ---------------------------------------------------------------
# Fig 7.3  GAN 학습: 생성 분포가 실제 분포로 수렴
# ---------------------------------------------------------------
def fig_gan_training():
    # 실제: 반지름 3의 링 위 8개 가우시안 모드
    K = 8
    ctr = np.array([[np.cos(2*np.pi*k/K), np.sin(2*np.pi*k/K)] for k in range(K)])*3.0
    def real(n=1600):
        idx = rng.integers(0, K, n)
        return ctr[idx] + rng.normal(0, 0.22, (n, 2))
    R = real()
    stages = {
        "iter 0": rng.normal(0, 0.6, (1600, 2)),                       # 중앙 blob
        "iter 3k": 0.5*real() + rng.normal(0, 0.9, (1600, 2)),          # 부분 학습
        "iter 20k": real() + rng.normal(0, 0.05, (1600, 2)),           # 수렴
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), constrained_layout=True)
    for ax, (title, G) in zip(axes, stages.items()):
        ax.scatter(R[:,0], R[:,1], s=6, color=C_REAL, alpha=0.25, label="real")
        ax.scatter(G[:,0], G[:,1], s=6, color=C_GEN, alpha=0.55, label="generated")
        ax.set_title(title); ax.set_xlim(-4.5,4.5); ax.set_ylim(-4.5,4.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    save(fig, "ch7_gan_training.png")


# ---------------------------------------------------------------
# Fig 7.4  모드 붕괴: 실제 다중 모드 vs 몇 개 모드만 생성
# ---------------------------------------------------------------
def fig_mode_collapse():
    K = 8
    ctr = np.array([[np.cos(2*np.pi*k/K), np.sin(2*np.pi*k/K)] for k in range(K)])*3.0
    def real(n=1600):
        idx = rng.integers(0, K, n); return ctr[idx]+rng.normal(0,0.22,(n,2))
    R = real()
    # 붕괴: 2개 모드에만 집중
    idx = rng.choice([1, 5], 1600)
    Gc = ctr[idx] + rng.normal(0, 0.22, (1600, 2))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.1), constrained_layout=True)
    axes[0].scatter(R[:,0], R[:,1], s=7, color=C_REAL, alpha=0.5)
    axes[0].set_title("(a) 실제 데이터 (8개 모드)")
    axes[1].scatter(R[:,0], R[:,1], s=7, color=C_REAL, alpha=0.12)
    axes[1].scatter(Gc[:,0], Gc[:,1], s=7, color=C_GEN, alpha=0.6)
    axes[1].set_title("(b) 모드 붕괴한 생성 결과")
    for ax in axes:
        ax.set_xlim(-4.5,4.5); ax.set_ylim(-4.5,4.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch7_mode_collapse.png")


# ---------------------------------------------------------------
# Fig 7.5  Wasserstein vs JS: 분리된 분포에서의 거리/기울기
# ---------------------------------------------------------------
def fig_wasserstein():
    d = np.linspace(-3, 3, 400)
    js = np.where(np.abs(d) < 1e-6, 0.0, np.log(2))     # 겹치지 않으면 상수 log2
    w = np.abs(d)                                        # EM 거리 = |d|
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(d, js, color=C_GEN, lw=2.4, label="JS divergence")
    ax.plot(d, w,  color=C_REAL, lw=2.4, label="Wasserstein distance")
    ax.set_xlabel("두 분포의 분리 정도  $\\theta$")
    ax.set_ylabel("거리 / 발산 값")
    ax.set_title("분리된 분포에서 거리 비교")
    ax.axhline(np.log(2), color=C_GEN, ls=":", lw=1, alpha=0.6)
    ax.annotate("JS는 평평 → 기울기 0\n(학습 신호 없음)", xy=(1.6, np.log(2)),
                xytext=(0.4, 1.55), fontsize=9, color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c"))
    ax.annotate("W는 선형 → 유용한 기울기", xy=(2.2, 2.2),
                xytext=(-2.9, 2.3), fontsize=9, color="#1d4ed8")
    ax.legend(loc="upper center", fontsize=9)
    save(fig, "ch7_wasserstein.png")


# ---------------------------------------------------------------
# Fig 7.6  디퓨전 forward: 데이터 -> 점진적 노이즈
# ---------------------------------------------------------------
def fig_diffusion():
    # two-moons 형태 데이터
    n = 900
    a = np.pi * rng.random(n)
    m1 = np.c_[np.cos(a), np.sin(a)]
    m2 = np.c_[1 - np.cos(a), 1 - np.sin(a) - 0.5]
    x0 = np.vstack([m1, m2]) + rng.normal(0, 0.04, (2*n, 2))
    x0 = (x0 - x0.mean(0)) / x0.std(0)
    alphas = [1.0, 0.85, 0.55, 0.25, 0.0]     # 남은 신호 비율
    fig, axes = plt.subplots(1, len(alphas), figsize=(12, 2.7), constrained_layout=True)
    noise = rng.normal(0, 1, x0.shape)
    for ax, al in zip(axes, alphas):
        xt = np.sqrt(al) * x0 + np.sqrt(1 - al) * noise
        col = C_REAL if al == 1.0 else (C_ACC if al == 0.0 else "#6b7280")
        ax.scatter(xt[:,0], xt[:,1], s=5, color=col, alpha=0.55)
        ax.set_title(f"t: signal={al:.2f}", fontsize=10)
        ax.set_xlim(-3,3); ax.set_ylim(-3,3)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_ylabel("forward 과정 →", fontsize=10)
    save(fig, "ch7_diffusion.png")


if __name__ == "__main__":
    fig_manifold()
    fig_latent_ae_vae()
    fig_gan_training()
    fig_mode_collapse()
    fig_wasserstein()
    fig_diffusion()
    print("done")
