# -*- coding: utf-8 -*-
"""12·13주차(멀티모달·확산 모델) 교재용 그림 중 도식과 수치 그래프. 모두 직접 작성한 그림이다.
스테이블 디퓨전으로 생성한 실제 이미지 그림은 gen_ch11_real.py가 만든다.
글꼴에 없는 기호(아래첨자, 그리스 문자)는 수식 글꼴($...$)로 쓴다."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from _ours import new_ax, heading, zone, BRAND, CORAL, INDIGO, AMBER, ROSE, INK, SUB, LINE, TINT
from gen_ours_ch9 import rb, tx, arr, blend, save

mpl.rcParams["mathtext.fontset"] = "dejavusans"


# ============ 그림 11-1: 서로 다른 모달리티를 하나의 표현 공간으로 ============
def fig_space():
    fig, ax = new_ax(13.6, 6.4, (0, 27.2), (0, 12.8))
    heading(ax, 0.4, 12.0, "서로 다른 데이터를 하나의 의미 공간에 정렬합니다")
    rb(ax, 0.6, 7.0, 5.6, 3.2, TINT["brand"], BRAND, lw=2.0, shadow=True)
    tx(ax, 3.4, 9.5, "이미지", fs=13, bold=True)
    tx(ax, 3.4, 8.5, "픽셀 격자 (공간 정보)", fs=10, color=SUB)
    tx(ax, 3.4, 7.7, "고양이가 소파에 앉아 있는 사진", fs=9, color=SUB)
    rb(ax, 0.6, 2.2, 5.6, 3.2, TINT["indigo"], INDIGO, lw=2.0, shadow=True)
    tx(ax, 3.4, 4.7, "텍스트", fs=13, bold=True)
    tx(ax, 3.4, 3.7, "토큰 열 (개념 정보)", fs=10, color=SUB)
    tx(ax, 3.4, 2.9, '"a cat sitting on a sofa"', fs=9.5, color=SUB)
    rb(ax, 8.0, 7.2, 4.6, 2.8, "#ffffff", BRAND, lw=2.0, shadow=True)
    tx(ax, 10.3, 8.6, "이미지 인코더", fs=12, bold=True)
    rb(ax, 8.0, 2.4, 4.6, 2.8, "#ffffff", INDIGO, lw=2.0, shadow=True)
    tx(ax, 10.3, 3.8, "텍스트 인코더", fs=12, bold=True)
    arr(ax, (6.3, 8.6), (7.9, 8.6), BRAND, lw=2.4)
    arr(ax, (6.3, 3.8), (7.9, 3.8), INDIGO, lw=2.4)
    rb(ax, 15.0, 1.4, 11.6, 9.8, "#fbfcfd", LINE, lw=2.0, r=0.4)
    tx(ax, 20.8, 10.5, "공통 표현 공간", fs=13, bold=True)
    # 고양이 쌍
    for (x, y), c in [((17.6, 8.0), BRAND), ((18.8, 7.0), INDIGO), ((23.4, 5.0), BRAND), ((22.2, 4.0), INDIGO)]:
        ax.add_patch(Circle((x, y), 0.32, fc=c, ec="white", lw=1.5, zorder=6))
    ax.add_patch(Circle((18.2, 7.5), 1.45, fc="none", ec=AMBER, lw=2.0, ls="--", zorder=4))
    ax.add_patch(Circle((22.8, 4.5), 1.45, fc="none", ec=AMBER, lw=2.0, ls="--", zorder=4))
    tx(ax, 16.4, 9.6, "고양이 사진", fs=9.5, color=BRAND, bold=True)
    tx(ax, 21.4, 7.2, '"a cat on a sofa"', fs=9.5, color=INDIGO, bold=True)
    tx(ax, 24.6, 6.6, "개 사진", fs=9.5, color=BRAND, bold=True)
    tx(ax, 21.2, 2.5, '"a dog running"', fs=9.5, color=INDIGO, bold=True)
    arr(ax, (12.7, 8.6), (16.9, 8.1), BRAND, lw=2.4, rad=-0.1)
    arr(ax, (12.7, 3.8), (21.0, 3.2), INDIGO, lw=2.4, rad=0.1)
    tx(ax, 20.8, 1.9, "뜻이 같은 쌍은 가깝게, 다른 쌍은 멀게 배치합니다", fs=10.5, color=SUB)
    save(fig, "ch11_space.png")


# ============ 그림 11-2: CLIP의 대조 학습 ============
def fig_clip():
    fig, ax = new_ax(13.6, 7.0, (0, 27.2), (0, 14.0))
    heading(ax, 0.4, 13.2, "CLIP의 대조 학습: 배치 안의 정답 쌍만 가깝게 만듭니다")
    n = 4
    cell = 1.9
    x0, y0 = 11.0, 2.4
    S = np.array([[0.92, 0.18, 0.11, 0.21],
                  [0.15, 0.88, 0.23, 0.09],
                  [0.10, 0.20, 0.90, 0.17],
                  [0.22, 0.12, 0.19, 0.85]])
    for i in range(n):
        for j in range(n):
            on = (i == j)
            ax.add_patch(Rectangle((x0 + j * cell, y0 + (n - 1 - i) * cell), cell, cell,
                                   fc=blend(BRAND, 0.78) if on else "#f1f3f5", ec="white", lw=2, zorder=3))
            tx(ax, x0 + j * cell + cell / 2, y0 + (n - 1 - i) * cell + cell / 2, f"{S[i, j]:.2f}",
               fs=11, bold=on, color=INK if on else SUB)
    for j in range(n):
        tx(ax, x0 + j * cell + cell / 2, y0 + n * cell + 0.5, f"$T_{j + 1}$", fs=13, bold=True, color=INDIGO)
    for i in range(n):
        tx(ax, x0 - 0.6, y0 + (n - 1 - i) * cell + cell / 2, f"$I_{i + 1}$", fs=13, bold=True, color=BRAND)
    tx(ax, x0 + n * cell / 2, y0 + n * cell + 1.4, "텍스트 임베딩", fs=11, color=INDIGO, bold=True)
    tx(ax, x0 - 2.5, y0 + n * cell / 2, "이미지\n임베딩", fs=11, color=BRAND, bold=True)
    rb(ax, 0.6, 7.2, 6.2, 4.6, TINT["gray"], LINE, lw=1.5)
    tx(ax, 3.7, 10.9, "한 배치 N개의 (이미지, 텍스트) 쌍", fs=11, bold=True)
    tx(ax, 3.7, 9.8, "대각선 = 정답 쌍 (N개)", fs=10, color=BRAND, bold=True)
    tx(ax, 3.7, 8.9, "나머지 = 오답 쌍 ($N^2 - N$개)", fs=10, color=SUB)
    tx(ax, 3.7, 7.9, "(그림의 값은 설명용입니다)", fs=9, color=SUB)
    rb(ax, 0.6, 1.2, 6.2, 4.6, TINT["amber"], AMBER, lw=1.5)
    tx(ax, 3.7, 4.9, "학습 목표", fs=11, bold=True)
    tx(ax, 3.7, 3.9, "행마다: 정답 텍스트를 고르는 분류", fs=10)
    tx(ax, 3.7, 3.0, "열마다: 정답 이미지를 고르는 분류", fs=10)
    tx(ax, 3.7, 2.1, "두 교차 엔트로피의 평균", fs=10, color=SUB)
    save(fig, "ch11_clip.png")


# ============ 그림 11-3: 전방향 과정의 스케줄 (수치 그래프) ============
def schedule(T=1000, b0=1e-4, b1=0.02):
    beta = np.linspace(b0, b1, T)
    abar = np.cumprod(1 - beta)
    return beta, abar


def fig_schedule():
    beta, abar = schedule()
    t = np.arange(1, 1001)
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.2))
    c = [BRAND, INDIGO, CORAL]
    axs[0].plot(t, beta, color=c[0], lw=2.6)
    axs[0].set_title(r"노이즈 크기 $\beta_t$ (선형 스케줄)", fontsize=12, fontweight="bold", color=INK)
    axs[0].set_xlabel("단계 t"); axs[0].set_ylabel(r"$\beta_t$")
    axs[1].plot(t, abar, color=c[1], lw=2.6)
    axs[1].set_title(r"원본이 남는 비율 $\bar{\alpha}_t$", fontsize=12, fontweight="bold", color=INK)
    axs[1].set_xlabel("단계 t"); axs[1].set_ylabel(r"$\bar{\alpha}_t$")
    axs[2].plot(t, np.sqrt(abar), color=c[1], lw=2.6, label=r"신호  $\sqrt{\bar{\alpha}_t}$")
    axs[2].plot(t, np.sqrt(1 - abar), color=c[2], lw=2.6, label=r"노이즈  $\sqrt{1-\bar{\alpha}_t}$")
    axs[2].set_title(r"$x_t$에 섞인 신호와 노이즈의 크기", fontsize=12, fontweight="bold", color=INK)
    axs[2].set_xlabel("단계 t")
    axs[2].legend(frameon=False, fontsize=11, loc="center right")
    for a in axs:
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(0, 1000)
    plt.tight_layout()
    save(fig, "ch11_schedule.png")
    print("abar at t=100,250,500,750,1000:", [f"{float(abar[k - 1]):.4g}" for k in (100, 250, 500, 750, 1000)])
    print("signal sqrt(abar):", [f"{float(np.sqrt(abar[k - 1])):.3f}" for k in (100, 250, 500, 750, 1000)])


# ============ 그림 11-4: DDPM 학습 한 단계 ============
def fig_ddpm_train():
    fig, ax = new_ax(14.0, 6.2, (0, 28.0), (0, 12.4))
    heading(ax, 0.4, 11.6, "DDPM 학습 한 단계: 섞은 노이즈를 맞히게 합니다")
    steps = [("① 깨끗한 이미지", r"$x_0$를 학습 데이터에서" + "\n뽑습니다", TINT["brand"], BRAND, 0.6),
             ("② 단계와 노이즈", r"$t$는 1~T 중 무작위" + "\n" + r"$\epsilon$은 표준 가우시안", TINT["indigo"], INDIGO, 5.9),
             ("③ 섞기", r"$x_t=\sqrt{\bar{\alpha}_t}\,x_0$" + "\n" + r"$+\sqrt{1-\bar{\alpha}_t}\,\epsilon$", TINT["amber"], AMBER, 11.2),
             ("④ U-Net 예측", r"$\epsilon_\theta(x_t,t)$가" + "\n" + r"$\epsilon$을 추정합니다", TINT["rose"], ROSE, 16.5),
             ("⑤ 손실", r"$\|\epsilon-\epsilon_\theta(x_t,t)\|^2$" + "\n" + "을 줄입니다", TINT["coral"], CORAL, 21.8)]
    for k, (t1, t2, fc, ec, x) in enumerate(steps):
        rb(ax, x, 4.0, 5.0, 5.2, fc, ec, lw=2.0, shadow=True)
        tx(ax, x + 2.5, 8.2, t1, fs=11.5, bold=True)
        tx(ax, x + 2.5, 6.2, t2, fs=10.5, color=INK)
        if k < 4:
            arr(ax, (x + 5.05, 6.6), (x + 5.8, 6.6), SUB, lw=2.4)
    tx(ax, 14.0, 2.4, "손실이 줄수록 U-Net은 '이 노이즈 이미지에 섞인 노이즈가 무엇인지'를 더 잘 맞힙니다", fs=11, color=INK)
    tx(ax, 14.0, 1.4, "(이 한 단계를 수많은 이미지와 수많은 t에 대해 반복해 학습합니다)", fs=9.5, color=SUB)
    save(fig, "ch11_ddpm_train.png")


# ============ 그림 11-5: 역방향 샘플링 루프 ============
def fig_sampling():
    fig, ax = new_ax(14.0, 5.6, (0, 28.0), (0, 11.2))
    heading(ax, 0.4, 10.4, "생성(샘플링): 순수 노이즈에서 시작해 노이즈를 조금씩 덜어 냅니다")
    xs = [1.0, 6.0, 11.0, 16.0, 21.0]
    labs = [r"$x_T$" + "\n순수 노이즈", r"$x_{T-1}$", "···", r"$x_1$", r"$x_0$" + "\n생성된 이미지"]
    cols = [CORAL, AMBER, SUB, INDIGO, BRAND]
    for x, l, c in zip(xs, labs, cols):
        rb(ax, x, 4.0, 4.2, 3.6, blend(c, 0.18) if c != SUB else TINT["gray"], c, lw=2.0, shadow=True)
        tx(ax, x + 2.1, 5.8, l, fs=11.5, bold=True)
    for k in range(4):
        arr(ax, (xs[k] + 4.25, 5.8), (xs[k + 1] - 0.05, 5.8), SUB, lw=2.4)
    for k in (0, 1, 3):
        tx(ax, xs[k] + 4.6, 7.1, "U-Net", fs=9, color=ROSE, bold=True)
    tx(ax, 14.0, 2.6, r"각 단계: $\hat{\epsilon}=\epsilon_\theta(x_t,t)$로 노이즈를 추정하고, 그만큼 덜어 내 $x_{t-1}$을 만듭니다", fs=11)
    tx(ax, 14.0, 1.6, "원 논문은 T=1000단계, 스테이블 디퓨전의 기본 설정은 50단계입니다", fs=10, color=SUB)
    save(fig, "ch11_sampling.png")


# ============ 그림 11-6: LDM / 스테이블 디퓨전 구조 ============
def fig_ldm():
    fig, ax = new_ax(15.0, 7.4, (0, 30.0), (0, 14.8))
    heading(ax, 0.4, 14.0, "잠재 확산 모델(LDM): 압축한 공간에서 확산합니다")
    rb(ax, 0.6, 7.0, 4.2, 3.6, TINT["brand"], BRAND, lw=2.0, shadow=True)
    tx(ax, 2.7, 9.2, "이미지", fs=12, bold=True); tx(ax, 2.7, 8.2, "512×512×3", fs=10, color=SUB)
    tx(ax, 2.7, 7.5, "= 786,432개", fs=9.5, color=SUB)
    rb(ax, 6.0, 7.3, 3.2, 3.0, "#ffffff", BRAND, lw=2.0, shadow=True)
    tx(ax, 7.6, 8.8, "VAE\n인코더", fs=11, bold=True)
    arr(ax, (4.85, 8.8), (5.95, 8.8), BRAND, lw=2.4)
    zone(ax, 10.2, 1.2, 12.6, 11.4, "", color=INDIGO, fc="#f5f7fe", ec="#cdd6fb")
    tx(ax, 16.5, 11.9, "잠재 공간에서의 확산", fs=12, bold=True, color=INDIGO)
    rb(ax, 11.0, 7.0, 3.8, 3.6, TINT["amber"], AMBER, lw=2.0)
    tx(ax, 12.9, 9.2, "잠재 z", fs=12, bold=True); tx(ax, 12.9, 8.2, "64×64×4", fs=10, color=SUB)
    tx(ax, 12.9, 7.5, "= 16,384개", fs=9.5, color=SUB)
    rb(ax, 16.4, 6.6, 5.4, 4.6, "#ffffff", ROSE, lw=2.0, shadow=True)
    tx(ax, 19.1, 10.3, "노이즈 예측 U-Net", fs=11.5, bold=True)
    tx(ax, 19.1, 9.2, r"$\epsilon_\theta(z_t,\,t,\,c)$", fs=11, color=INK)
    tx(ax, 19.1, 8.1, "교차 어텐션으로", fs=9.5, color=SUB)
    tx(ax, 19.1, 7.4, "텍스트 조건 c를 반영", fs=9.5, color=SUB)
    arr(ax, (9.25, 8.8), (10.95, 8.8), BRAND, lw=2.4)
    arr(ax, (14.85, 8.8), (16.35, 8.8), AMBER, lw=2.4)
    ax.annotate("", xy=(14.2, 6.9), xytext=(18.0, 6.55),
                arrowprops=dict(arrowstyle="-|>", color=ROSE, lw=2.0, connectionstyle="arc3,rad=0.25"))
    tx(ax, 15.6, 5.55, "노이즈 제거를 T번 반복", fs=9.5, color=ROSE, bold=True)
    rb(ax, 11.0, 1.7, 3.8, 2.2, TINT["indigo"], INDIGO, lw=2.0, shadow=True)
    tx(ax, 12.9, 2.8, "프롬프트", fs=11, bold=True)
    rb(ax, 15.8, 1.7, 3.6, 2.2, "#ffffff", INDIGO, lw=2.0, shadow=True)
    tx(ax, 17.6, 2.8, "텍스트 인코더", fs=10.5, bold=True)
    arr(ax, (14.85, 2.8), (15.75, 2.8), INDIGO, lw=2.2)
    arr(ax, (19.45, 2.8), (21.0, 2.8), INDIGO, lw=2.2)
    arr(ax, (21.0, 2.8), (21.0, 6.5), INDIGO, lw=2.2)
    tx(ax, 22.2, 4.5, "임베딩\n77×768", fs=9, color=INDIGO)
    rb(ax, 24.0, 7.3, 3.2, 3.0, "#ffffff", BRAND, lw=2.0, shadow=True)
    tx(ax, 25.6, 8.8, "VAE\n디코더", fs=11, bold=True)
    arr(ax, (21.85, 8.8), (23.95, 8.8), BRAND, lw=2.4)
    rb(ax, 25.8, 1.6, 3.8, 3.0, TINT["brand"], BRAND, lw=2.0, shadow=True)
    tx(ax, 27.7, 3.1, "생성된\n이미지", fs=11, bold=True)
    arr(ax, (25.6, 7.25), (27.6, 4.65), BRAND, lw=2.4, rad=-0.15)
    tx(ax, 15.0, 0.5, "이미지 크기의 1/48인 공간에서 확산하므로 계산이 크게 줄어듭니다 (512×512 기준)", fs=10.5, color=INK)
    save(fig, "ch11_ldm.png")


# ============ 그림 11-7: 교차 어텐션 ============
def fig_cross():
    fig, ax = new_ax(14.0, 6.6, (0, 28.0), (0, 13.2))
    heading(ax, 0.4, 12.4, "교차 어텐션: 이미지 위치가 프롬프트의 단어를 참조합니다")
    tx(ax, 4.0, 11.0, "잠재 z의 각 위치 (64×64 = 4,096개)", fs=11, bold=True, color=BRAND)
    tx(ax, 4.0, 10.3, "→ 쿼리 Q", fs=10.5, color=BRAND)
    for i in range(4):
        rb(ax, 1.0 + i * 1.7, 8.2, 1.4, 1.4, TINT["brand"], BRAND, lw=1.8)
    tx(ax, 22.5, 11.0, "프롬프트의 토큰", fs=11.5, bold=True, color=INDIGO)
    tx(ax, 22.5, 10.3, "→ 키 K, 값 V", fs=10.5, color=INDIGO)
    toks = ["a", "red", "apple", "on", "a", "table"]
    for k, w in enumerate(toks):
        rb(ax, 16.6 + k * 1.75, 8.2, 1.55, 1.4, TINT["indigo"], INDIGO, lw=1.8)
        tx(ax, 17.4 + k * 1.75, 8.9, w, fs=10)
    focus = {1: 0.9, 2: 1.0}
    for k in range(len(toks)):
        w = focus.get(k, 0.15)
        arr(ax, (1.7, 8.15), (17.4 + k * 1.75, 8.15), blend(CORAL, 0.25 + 0.75 * w), lw=1.0 + 3.0 * w, rad=0.16, ms=8)
    tx(ax, 14.0, 5.6, "단어와 관련이 큰 위치일수록 굵은 선", fs=9.5, color=CORAL)
    rb(ax, 6.2, 2.0, 15.6, 3.0, TINT["amber"], AMBER, lw=2.0, shadow=True)
    tx(ax, 14.0, 3.9, r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d}}\right)V$", fs=13)
    tx(ax, 14.0, 2.6, "각 위치는 관련이 큰 단어(예: red, apple)의 값을 더 많이 가져옵니다", fs=10, color=SUB)
    tx(ax, 14.0, 0.8, "Q는 이미지 쪽에서, K와 V는 텍스트 쪽에서 옵니다 (셀프 어텐션과 다른 점)", fs=10.5)
    save(fig, "ch11_cross.png")


# ============ 그림 11-8: 분류기 없는 가이던스 ============
def fig_cfg():
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    o = np.array([0.0, 0.0])
    eu = np.array([2.0, 0.9])
    ec = np.array([2.6, 1.6])
    d = ec - eu
    for w, col, lab in [(1, INDIGO, "w=1"), (3, AMBER, "w=3"), (7.5, CORAL, "w=7.5")]:
        e = eu + w * d
        ax.annotate("", xy=e, xytext=o, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6))
        ax.text(e[0] + 0.12, e[1] + 0.05, r"$\tilde{\epsilon}$ (" + lab + ")", color=col, fontsize=12, fontweight="bold")
    ax.annotate("", xy=eu, xytext=o, arrowprops=dict(arrowstyle="-|>", color=SUB, lw=2.6))
    ax.text(eu[0] - 0.1, eu[1] - 0.55, "조건 없음", color=SUB, fontsize=11, fontweight="bold", ha="center")
    ax.annotate("", xy=ec, xytext=eu, arrowprops=dict(arrowstyle="-|>", color=BRAND, lw=2.6, ls="--"))
    ax.text(ec[0] + 0.45, eu[1] + 0.2, "조건이 만든 방향\n(조건 있음 $-$ 조건 없음)", color=BRAND, fontsize=10.5, fontweight="bold")
    ax.set_xlim(-0.3, 8.6); ax.set_ylim(-0.9, 6.4)
    ax.set_aspect("equal")
    ax.set_title(r"$\tilde{\epsilon}=\epsilon_{\mathrm{uncond}}+w\,(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}})$   (설명용 2차원 그림)",
                 fontsize=12.5, fontweight="bold", color=INK)
    ax.axis("off")
    plt.tight_layout()
    save(fig, "ch11_cfg.png")


# ============ 그림 11-9: 확산 모델과 GAN ============
def fig_compare():
    fig, ax = new_ax(14.0, 6.4, (0, 28.0), (0, 12.8))
    heading(ax, 0.4, 12.0, "확산 모델이 GAN과 다른 점")
    cols = [("GAN", CORAL, "생성자와 판별자의 경쟁", ["적대적 학습이라 불안정", "모드 붕괴(일부 유형만 생성)", "한 번에 생성(빠름)"]),
            ("확산 모델", BRAND, "노이즈 예측이라는 회귀 문제", ["안정적인 손실(MSE)로 학습", "다양한 모드를 잘 포착", "여러 단계를 반복(느림)"])]
    for k, (nm, c, sub, items) in enumerate(cols):
        x0 = 1.0 + k * 13.6
        rb(ax, x0, 1.2, 12.2, 9.6, "#ffffff", c, lw=2.2, shadow=True)
        rb(ax, x0, 9.2, 12.2, 1.6, blend(c, 0.18), c, lw=2.2)
        tx(ax, x0 + 6.1, 10.35, nm, fs=14, bold=True)
        tx(ax, x0 + 6.1, 9.5, sub, fs=10, color=SUB)
        for i, it in enumerate(items):
            tx(ax, x0 + 0.8, 7.4 - i * 1.9, "•  " + it, fs=11.5, ha="left")
    save(fig, "ch11_compare.png")


if __name__ == "__main__":
    fig_space(); fig_clip(); fig_schedule(); fig_ddpm_train(); fig_sampling()
    fig_ldm(); fig_cross(); fig_cfg(); fig_compare()
