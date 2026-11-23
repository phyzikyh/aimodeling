# -*- coding: utf-8 -*-
"""14주차(강화학습·RLHF·DPO) 교재용 그림. 모두 직접 작성한 그림이다.
수치 그림(13-2, 13-3, 13-4, 13-8)의 값은 exp14.py(numpy, torch)로 직접 계산한 값이다.
도식은 색을 쓰지 않고, 수치 그래프만 팔레트를 쓴다. 글꼴에 없는 기호는 수식 글꼴($...$)로 쓴다."""
import json, os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from _ours import new_ax, heading, BRAND, CORAL, INDIGO, AMBER, ROSE, INK, SUB, LINE
from gen_ours_ch9 import rb, tx, arr, save

mpl.rcParams["mathtext.fontset"] = "dejavusans"
MONO = "#3a3f45"
WHITE = "#ffffff"
GRAY = "#f1f3f5"
SP = "C:/Users/user/AppData/Local/Temp/claude/34cacf723f7d5199/38411a20-952e-4490-9ed6-67155d38a355/scratchpad/exp14_out.json"
R = json.load(open(SP, encoding="utf-8"))


# ============ 그림 13-1: 에이전트와 환경의 상호작용 ============
def fig_loop():
    fig, ax = new_ax(14.0, 6.6, (0, 28.0), (0, 13.2))
    heading(ax, 0.4, 12.4, "강화학습의 상호작용: 에이전트가 행동하면 환경이 새 상태와 보상을 돌려줍니다")
    rb(ax, 1.5, 6.4, 7.6, 3.8, WHITE, MONO, lw=2.2, shadow=True)
    tx(ax, 5.3, 9.0, "에이전트", fs=14, bold=True)
    tx(ax, 5.3, 7.6, r"정책 $\pi(a\,|\,s)$로 행동을 고름", fs=10, color=SUB)
    rb(ax, 18.9, 6.4, 7.6, 3.8, GRAY, MONO, lw=2.2, shadow=True)
    tx(ax, 22.7, 9.0, "환경", fs=14, bold=True)
    tx(ax, 22.7, 7.6, r"전이 $P(s'\,|\,s,a)$와 보상 $R$", fs=10, color=SUB)
    arr(ax, (9.2, 9.4), (18.8, 9.4), MONO, lw=2.4)
    tx(ax, 14.0, 10.2, r"행동 $a_t$", fs=12)
    arr(ax, (18.8, 7.2), (9.2, 7.2), MONO, lw=2.4)
    tx(ax, 14.0, 6.3, r"다음 상태 $s_{t+1}$, 보상 $r_{t+1}$", fs=12)
    rb(ax, 1.5, 1.4, 25.0, 3.0, WHITE, MONO, lw=1.6)
    tx(ax, 14.0, 3.5, r"궤적: $s_0,\ a_0,\ r_1,\ s_1,\ a_1,\ r_2,\ s_2,\ \ldots$", fs=12.5)
    tx(ax, 14.0, 2.3, r"반환: $G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \cdots$ (이 값의 기댓값을 키우는 것이 목표)", fs=10.5, color=SUB)
    save(fig, "ch13_loop.png")


# ============ 그림 13-2: 10팔 슬롯머신 ============
def fig_bandit():
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
    cols = {"0.0": INDIGO, "0.01": AMBER, "0.1": CORAL}
    labs = {"0.0": r"$\varepsilon=0$ (항상 활용)", "0.01": r"$\varepsilon=0.01$", "0.1": r"$\varepsilon=0.1$"}
    for k in ("0.0", "0.01", "0.1"):
        axs[0].plot(R["bandit"][k]["rew"], color=cols[k], lw=1.6, label=labs[k])
        axs[1].plot(np.array(R["bandit"][k]["opt"]) * 100, color=cols[k], lw=1.6, label=labs[k])
    axs[0].set_ylabel("평균 보상", fontsize=11); axs[1].set_ylabel("최적 행동을 고른 비율 (%)", fontsize=11)
    for a in axs:
        a.set_xlabel("스텝", fontsize=11); a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.25)
    axs[0].legend(fontsize=10, frameon=False, loc="lower right")
    fig.suptitle("10팔 슬롯머신에서 $\\varepsilon$-탐욕의 효과 (2000번 반복 평균)", fontsize=13, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "ch13_bandit.png")


# ============ 그림 13-3: 격자 세계의 가치 함수 ============
def fig_grid():
    V = np.array(R["grid"]["V"]); pol = R["grid"]["pol"]
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    ax.set_xlim(0, 4); ax.set_ylim(4, 0); ax.set_aspect("equal"); ax.axis("off")
    cmap = plt.get_cmap("YlGn")
    for r in range(4):
        for c in range(4):
            if (r, c) == (1, 1): fc = "#f6c6bf"
            elif (r, c) == (3, 3): fc = "#bfe8d8"
            else: fc = cmap(0.12 + 0.62 * V[r, c] / 10)
            ax.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec="white", lw=3))
            if (r, c) == (1, 1):
                ax.text(c + .5, r + .5, "함정\n-10", ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
            elif (r, c) == (3, 3):
                ax.text(c + .5, r + .5, "목표\n+10", ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
            else:
                ax.text(c + .5, r + .66, f"{V[r, c]:.2f}", ha="center", va="center", fontsize=11.5, color=INK)
                dirs = pol[f"{r},{c}"]
                d = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
                for a in dirs:
                    dx, dy = d[a]
                    ax.add_patch(FancyArrowPatch((c + .5, r + .28), (c + .5 + dx * .22, r + .28 + dy * .22),
                                 arrowstyle="-|>", mutation_scale=12, lw=1.6, color=MONO))
    ax.set_title(r"$V^*(s)$와 최적 행동 ($\gamma=0.9$, 값 반복 6회 갱신 후 수렴)", fontsize=12, fontweight="bold", color=INK)
    ax = axs[1]
    gs = [0.5, 0.9, 0.99]; vs = [0.312, 5.905, 9.510]
    ax.bar([str(g) for g in gs], vs, color=[INDIGO, BRAND, CORAL], alpha=0.88, width=0.55)
    for i, v in enumerate(vs):
        ax.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_ylim(0, 11); ax.set_xlabel(r"할인율 $\gamma$", fontsize=11); ax.set_ylabel(r"시작 칸 (0,0)의 $V^*$", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("할인율이 클수록 먼 목표가 더 크게 보입니다", fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ch13_grid.png")


# ============ 그림 13-4: 절벽 걷기, Q-러닝 대 SARSA ============
def fig_cliff():
    fig, axs = plt.subplots(1, 2, figsize=(13.6, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    ax.set_xlim(0, 12); ax.set_ylim(4, 0); ax.set_aspect("equal"); ax.axis("off")
    for r in range(4):
        for c in range(12):
            fc = "#f6c6bf" if (r == 3 and 1 <= c <= 10) else "#f1f3f5"
            ax.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec="white", lw=2.5))
    ax.text(0.5, 3.5, "S", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(11.5, 3.5, "G", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(5.5, 3.5, "절벽 (-100)", ha="center", va="center", fontsize=11, color=INK)
    for algo, col, off in (("q", INDIGO, 0.12), ("sarsa", CORAL, -0.12)):
        p = np.array(R["cliff"][algo]["path"], dtype=float)
        ax.plot(p[:, 1] + .5 + off, p[:, 0] + .5 + off, color=col, lw=3.0, alpha=0.92, solid_capstyle="round")
    ax.plot([], [], color=INDIGO, lw=3, label="Q-러닝의 탐욕 경로 (13걸음)")
    ax.plot([], [], color=CORAL, lw=3, label="SARSA의 탐욕 경로 (17걸음)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=10.5)
    ax.set_title("학습이 끝난 뒤 탐욕 정책이 가는 길", fontsize=12, fontweight="bold", color=INK)
    ax = axs[1]
    k = 10
    for algo, col, lab in (("q", INDIGO, "Q-러닝"), ("sarsa", CORAL, "SARSA")):
        m = np.array(R["cliff"][algo]["mean"])
        sm = np.convolve(m, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + k, sm, color=col, lw=2, label=lab)
    ax.set_ylim(-110, 0); ax.set_xlabel("에피소드", fontsize=11); ax.set_ylabel("에피소드 반환 (탐험 포함)", fontsize=11)
    ax.legend(frameon=False, fontsize=11, loc="lower right"); ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(r"학습 중 받은 반환 ($\varepsilon=0.1$, 100회 평균, 10개씩 이동평균)", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ch13_cliff.png")


# ============ 그림 13-5: DQN의 학습 구조 ============
def fig_dqn():
    fig, ax = new_ax(14.0, 7.2, (0, 28.0), (0, 14.4))
    heading(ax, 0.4, 13.6, "DQN의 학습 구조: 경험 재생 버퍼와 타깃 네트워크가 학습을 안정시킵니다")
    rb(ax, 0.6, 8.4, 5.6, 3.2, GRAY, MONO, lw=2.2, shadow=True)
    tx(ax, 3.4, 10.5, "환경", fs=13, bold=True); tx(ax, 3.4, 9.3, "CartPole 등", fs=9.5, color=SUB)
    rb(ax, 9.6, 8.4, 7.0, 3.2, WHITE, MONO, lw=2.2, shadow=True)
    tx(ax, 13.1, 10.5, r"온라인 네트워크 $Q_\theta$", fs=12.5, bold=True); tx(ax, 13.1, 9.3, "행동을 고르고, 학습으로 갱신됩니다", fs=9.5, color=SUB)
    rb(ax, 9.6, 2.4, 7.0, 3.2, WHITE, MONO, lw=2.2, shadow=True)
    tx(ax, 13.1, 4.5, "경험 재생 버퍼", fs=12.5, bold=True); tx(ax, 13.1, 3.3, r"$(s,a,r,s',\mathrm{done})$를 쌓아 둡니다", fs=9.5, color=SUB)
    rb(ax, 21.0, 2.4, 6.4, 3.2, WHITE, MONO, lw=2.2, shadow=True)
    tx(ax, 24.2, 4.5, r"타깃 네트워크 $Q_{\bar\theta}$", fs=12.5, bold=True); tx(ax, 24.2, 3.3, "평소에는 고정, 주기적으로 복사", fs=9.5, color=SUB)
    rb(ax, 21.0, 8.4, 6.4, 3.2, GRAY, MONO, lw=2.2, shadow=True)
    tx(ax, 24.2, 10.5, "손실", fs=12.5, bold=True); tx(ax, 24.2, 9.3, r"$(y - Q_\theta(s,a))^2$", fs=10.5, color=SUB)
    arr(ax, (9.5, 10.6), (6.3, 10.6), MONO, lw=2.2); tx(ax, 7.9, 11.4, r"행동 $a$ ($\varepsilon$-탐욕)", fs=9.5)
    arr(ax, (3.4, 8.3), (9.5, 4.6), MONO, lw=2.2, rad=0.15); tx(ax, 4.2, 5.7, "경험 저장", fs=9.5)
    arr(ax, (13.1, 5.7), (13.1, 8.3), MONO, lw=2.2); tx(ax, 15.9, 7.0, "무작위 미니배치", fs=9.5)
    arr(ax, (16.7, 10.6), (20.9, 10.6), MONO, lw=2.2); tx(ax, 18.8, 11.4, r"예측 $Q_\theta(s,a)$", fs=9.5)
    arr(ax, (24.2, 5.7), (24.2, 8.3), MONO, lw=2.2); tx(ax, 27.0, 7.0, r"목표 $y$", fs=9.5)
    arr(ax, (16.7, 8.5), (20.9, 5.5), "#9aa1a9", lw=1.8, rad=0.0); tx(ax, 18.0, 6.2, r"$C$번마다 복사", fs=9.5, color=SUB)
    tx(ax, 14.0, 0.9, r"목표 $y = r + \gamma \max_{a'} Q_{\bar\theta}(s',a')$ (종료 상태이면 $y=r$)", fs=11, color=INK)
    save(fig, "ch13_dqn.png")


# ============ 그림 13-6: PPO 클리핑 ============
def fig_ppo():
    eps = 0.2
    r = np.linspace(0.4, 1.6, 400)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=False)
    for ax, A, ttl in ((axs[0], 1.0, r"$\hat A>0$ (평균보다 좋은 행동)"), (axs[1], -1.0, r"$\hat A<0$ (평균보다 나쁜 행동)")):
        unc = r * A; cl = np.clip(r, 1 - eps, 1 + eps) * A; obj = np.minimum(unc, cl)
        ax.plot(r, unc, color=LINE, lw=2, ls="--", label=r"$r\hat A$ (클리핑 없음)")
        ax.plot(r, cl, color=AMBER, lw=1.8, ls=":", label=r"$\mathrm{clip}(r)\hat A$")
        ax.plot(r, obj, color=INDIGO, lw=3.2, label=r"$\min$ (PPO 목적)")
        ax.axvline(1.0, color=SUB, lw=0.8); ax.axvline(1 - eps, color=SUB, lw=0.8, ls=":"); ax.axvline(1 + eps, color=SUB, lw=0.8, ls=":")
        ax.set_xlabel(r"정책 비율 $r=\pi_\theta/\pi_{\theta_\mathrm{old}}$", fontsize=11); ax.set_title(ttl, fontsize=12, fontweight="bold", color=INK)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.2)
        ax.legend(fontsize=9.5, frameon=False, loc="upper left" if A > 0 else "lower left")
    axs[0].set_ylabel("목적 함수 값", fontsize=11)
    fig.suptitle(r"PPO 클리핑 목적 ($\varepsilon=0.2$): 비율이 구간 밖으로 나가면 더 밀어 줄 이유가 사라집니다", fontsize=12.5, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "ch13_ppo_clip.png")


# ============ 그림 13-7: RLHF와 DPO ============
def fig_rlhf():
    fig, ax = new_ax(14.0, 8.2, (0, 28.0), (0, 16.4))
    heading(ax, 0.4, 15.6, "RLHF는 세 단계, DPO는 한 단계로 선호를 학습합니다")
    tx(ax, 0.6, 13.6, "RLHF", fs=12.5, bold=True, ha="left")
    w, h, y = 7.8, 4.2, 8.6
    xs = [0.6, 10.1, 19.6]
    items = [("① 지도 미세조정(SFT)", "사람이 쓴 모범 응답으로", "초기 정책 $\\pi_{\\mathrm{SFT}}$를 만듭니다"),
             ("② 보상 모델 학습", "응답 쌍의 선호 비교로", "점수 $r_\\phi(x,y)$를 내는 모델을 학습"),
             ("③ PPO로 정책 개선", "보상 점수에서 KL 페널티를 빼고", "정책 $\\pi_\\theta$를 갱신합니다")]
    for k, (a, b, c) in enumerate(items):
        rb(ax, xs[k], y, w, h, WHITE if k < 2 else GRAY, MONO, lw=2.2, shadow=True)
        tx(ax, xs[k] + w / 2, y + 3.0, a, fs=11.5, bold=True)
        tx(ax, xs[k] + w / 2, y + 1.9, b, fs=9.5, color=SUB); tx(ax, xs[k] + w / 2, y + 1.0, c, fs=9.5, color=SUB)
        if k < 2: arr(ax, (xs[k] + w + 0.05, y + h / 2), (xs[k + 1] - 0.05, y + h / 2), MONO, lw=2.2)
    ax.plot([0.6, 27.4], [7.2, 7.2], color=LINE, lw=1.4)
    tx(ax, 0.6, 5.9, "DPO", fs=12.5, bold=True, ha="left")
    y2 = 1.2
    rb(ax, 0.6, y2, 7.8, 3.6, WHITE, MONO, lw=2.2, shadow=True)
    tx(ax, 4.5, y2 + 2.5, "선호 쌍 데이터", fs=11.5, bold=True); tx(ax, 4.5, y2 + 1.3, "$(x,\\ y_w,\\ y_l)$", fs=10, color=SUB)
    arr(ax, (8.5, y2 + 1.8), (10.0, y2 + 1.8), MONO, lw=2.2)
    rb(ax, 10.1, y2, 9.4, 3.6, GRAY, MONO, lw=2.2, shadow=True)
    tx(ax, 14.8, y2 + 2.5, "DPO 손실로 직접 학습", fs=11.5, bold=True)
    tx(ax, 14.8, y2 + 1.3, "기준 정책과 비교한 로그 확률만 사용", fs=9.5, color=SUB)
    arr(ax, (19.6, y2 + 1.8), (21.1, y2 + 1.8), MONO, lw=2.2)
    rb(ax, 21.2, y2, 6.2, 3.6, WHITE, MONO, lw=2.2, shadow=True)
    tx(ax, 24.3, y2 + 2.5, "정렬된 정책", fs=11.5, bold=True); tx(ax, 24.3, y2 + 1.3, "$\\pi_\\theta$", fs=10, color=SUB)
    tx(ax, 14.0, 0.2, "보상 모델과 강화학습 반복이 사라지고, SFT 모델이 기준 정책 $\\pi_{\\mathrm{ref}}$로 쓰입니다", fs=9.8, color=INK)
    save(fig, "ch13_rlhf.png")


# ============ 그림 13-8: DPO 소규모 실험 ============
def fig_dpo():
    d = R["dpo"]; ref = np.array(d["ref"])
    fig, axs = plt.subplots(1, 4, figsize=(14.2, 4.1), sharey=True)
    x = np.arange(4); w = 0.27
    for ax, b in zip(axs, ["0.1", "0.5", "1.0", "2.0"]):
        pi = np.array(d["res"][b]["pi"]); st = np.array(d["res"][b]["star"])
        ax.bar(x - w, ref, w, color=LINE, label=r"기준 $\pi_{\mathrm{ref}}$")
        ax.bar(x, pi, w, color=INDIGO, alpha=0.9, label="DPO로 학습한 정책")
        ax.bar(x + w, st, w, color=AMBER, alpha=0.9, label=r"이론해 $\pi^*$")
        ax.set_xticks(x); ax.set_xticklabels([f"y{i}\n(r={i})" for i in range(4)], fontsize=9.5)
        ax.set_title(rf"$\beta={b}$   KL={d['res'][b]['kl']:.2f}", fontsize=11.5, fontweight="bold", color=INK)
        ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("응답 확률", fontsize=11)
    axs[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(r"응답 4개짜리 소규모 실험: $\beta$가 작을수록 보상이 높은 응답으로 멀리 이동합니다", fontsize=12.5, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "ch13_dpo_toy.png")


if __name__ == "__main__":
    fig_loop(); fig_bandit(); fig_grid(); fig_cliff(); fig_dqn(); fig_ppo(); fig_rlhf(); fig_dpo()
