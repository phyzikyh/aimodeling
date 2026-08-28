# -*- coding: utf-8 -*-
"""3장 아키텍처(Olah 스타일): LSTM 셀, GRU 셀."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np
from _style import save

PINK_F, PINK_E = "#ff8393", "#e2586a"    # σ / tanh 박스
OP_F, OP_E = "#ff5566", "#d63f4e"        # × +
OLIVE_F, OLIVE_E = "#cdcb83", "#9c9b4d"  # C 상태
GRAY_F, GRAY_E = "#d9dee4", "#94a0ac"    # h 상태
TEAL_F, TEAL_E = "#93dccf", "#48b09d"    # x_t
B_F, B_I, B_O = "#d6ece7", "#efe8d5", "#d8eddf"
LW = 2.0


def rbox(ax, cx, cy, w, h, text, fc, ec, fs=12, tc="black"):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.09",
                 fc=fc, ec=ec, lw=1.6, zorder=5))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=6, color=tc)


def opc(ax, cx, cy, sym, r=0.28):
    ax.add_patch(Circle((cx, cy), r, fc=OP_F, ec=OP_E, lw=1.6, zorder=6))
    ax.text(cx, cy, sym, ha="center", va="center", fontsize=15, color="white",
            fontweight="bold", zorder=7)


def wire(ax, pts, head=True, color="black", lw=LW):
    pts = np.array(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")
    if head:
        p0, p1 = pts[-2], pts[-1]
        ax.annotate("", xy=p1, xytext=p1-(p1-p0)*0.001,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16),
                    zorder=4)


def setup(ax, xl, yl):
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")


def fig_lstm_cell():
    fig, ax = plt.subplots(figsize=(10, 5.7), constrained_layout=True)
    setup(ax, (0, 15), (0, 9))
    # 외곽 점선 둥근 박스
    ax.add_patch(FancyBboxPatch((2.5, 1.5), 9.6, 5.4,
                 boxstyle="round,pad=0.02,rounding_size=0.35",
                 fc="none", ec="black", lw=1.8, ls=(0, (6, 4)), zorder=2))
    ax.text(7.3, 6.55, "LSTM 메모리 셀", ha="center", fontsize=12.5, fontweight="bold", zorder=6)
    # 게이트 밴드
    for x0, x1, c, lab, lx in [(2.7, 4.7, B_F, "망각 게이트", 3.7),
                               (4.7, 8.3, B_I, "입력 게이트", 6.5),
                               (8.3, 11.9, B_O, "출력 게이트", 10.1)]:
        ax.add_patch(Rectangle((x0, 1.7), x1-x0, 4.55, fc=c, ec="none", alpha=0.7, zorder=1))
        ax.text(lx, 6.05, lab, ha="center", fontsize=10.5, style="italic", color="#3a4654", zorder=6)
    yC, yH, yG = 5.5, 2.35, 3.55
    # 상태 박스
    rbox(ax, 1.15, yC, 1.5, 0.7, r"$C_{t-1}$", OLIVE_F, OLIVE_E, 13)
    rbox(ax, 13.2, yC, 1.5, 0.7, r"$C_{t}$", OLIVE_F, OLIVE_E, 13)
    rbox(ax, 1.15, yH, 1.5, 0.7, r"$h_{t-1}$", GRAY_F, GRAY_E, 13)
    rbox(ax, 13.2, yH, 1.5, 0.7, r"$h_{t}$", GRAY_F, GRAY_E, 13)
    rbox(ax, 10.1, 8.1, 1.4, 0.7, r"$h_{t}$", GRAY_F, GRAY_E, 13)
    ax.add_patch(Circle((3.6, 0.75), 0.42, fc=TEAL_F, ec=TEAL_E, lw=1.6, zorder=5))
    ax.text(3.6, 0.75, r"$X_t$", ha="center", va="center", fontsize=12, zorder=6)
    # 셀 상태 하이웨이 C
    xf, xp = 3.7, 5.9
    wire(ax, [(1.9, yC), (xf-0.3, yC)], head=False)
    opc(ax, xf, yC, "×")
    wire(ax, [(xf+0.3, yC), (xp-0.3, yC)], head=False)
    opc(ax, xp, yC, "+")
    wire(ax, [(xp+0.3, yC), (12.45, yC)])
    # h 하이웨이 (아래)
    wire(ax, [(1.9, yH), (12.45, yH)])
    # x_t + h_{t-1} 합류 → 게이트로 분기
    wire(ax, [(3.6, 1.17), (3.6, yH)], head=False)
    gate_x = [3.7, 5.3, 6.5, 9.2]      # forget σ, input σ, input tanh, output σ
    for gx in gate_x:
        wire(ax, [(gx, yH), (gx, yG-0.32)], head=False)
    # 게이트 박스
    rbox(ax, 3.7, yG, 0.95, 0.62, r"$\sigma$", PINK_F, PINK_E, 14)     # forget
    rbox(ax, 5.3, yG, 0.95, 0.62, r"$\sigma$", PINK_F, PINK_E, 14)     # input i
    rbox(ax, 6.5, yG, 1.0, 0.62, "tanh", PINK_F, PINK_E, 11)          # candidate
    rbox(ax, 9.2, yG, 0.95, 0.62, r"$\sigma$", PINK_F, PINK_E, 14)     # output
    # forget σ → × (셀선)
    wire(ax, [(3.7, yG+0.31), (3.7, yC-0.28)])
    # input: i×g → +
    xig = 5.9
    opc(ax, xig, 4.5, "×", r=0.26)
    wire(ax, [(5.3, yG+0.31), (5.3, 4.5), (xig-0.24, 4.5)])
    wire(ax, [(6.5, yG+0.31), (6.5, 4.5), (xig+0.24, 4.5)], head=False)
    wire(ax, [(xig, 4.76), (xig, yC-0.28)])
    # output gate: tanh(C_t) × o → h_t
    xo = 9.95
    rbox(ax, xo, 5.02, 1.0, 0.56, "tanh", PINK_F, PINK_E, 11)
    wire(ax, [(11.2, yC), (11.2, 5.02), (xo+0.5, 5.02)], head=True)   # C_t 분기 → tanh
    opc(ax, xo, 4.15, "×", r=0.26)
    wire(ax, [(xo, 4.74), (xo, 4.41)])                               # tanh → ×
    wire(ax, [(9.2, yG+0.31), (9.2, 4.15), (xo-0.24, 4.15)])          # σ → ×
    # × → h_t (위로 + 아래 h선으로)
    wire(ax, [(xo, 3.89), (xo, yH)], head=False)                    # 아래 h선 합류
    wire(ax, [(10.1, yH), (10.1, 7.75)])                            # 위 h_t 로
    ax.set_title("LSTM 셀 아키텍처", fontsize=13, fontweight="bold")
    save(fig, "ch3_lstm_arch.png")


def fig_gru_cell():
    fig, ax = plt.subplots(figsize=(10.4, 5.3), constrained_layout=True)
    setup(ax, (0, 15), (0, 8.2))
    ax.add_patch(FancyBboxPatch((2.5, 1.35), 9.8, 5.0,
                 boxstyle="round,pad=0.02,rounding_size=0.35",
                 fc="none", ec="black", lw=1.8, ls=(0, (6, 4)), zorder=2))
    ax.text(7.4, 6.0, "GRU 셀", ha="center", fontsize=12.5, fontweight="bold", zorder=6)
    for x0, x1, c, lab, lx in [(2.7, 6.2, B_F, "리셋 게이트", 4.3),
                               (6.2, 12.1, B_I, "업데이트 게이트", 9.0)]:
        ax.add_patch(Rectangle((x0, 1.55), x1-x0, 3.95, fc=c, ec="none", alpha=0.75, zorder=1))
        ax.text(lx, 5.25, lab, ha="center", fontsize=10.5, style="italic", color="#3a4654", zorder=6)
    yH, yG, ymid, ybus = 4.8, 2.9, 3.9, 1.7
    rbox(ax, 1.15, yH, 1.5, 0.7, r"$h_{t-1}$", GRAY_F, GRAY_E, 13)
    rbox(ax, 13.2, yH, 1.5, 0.7, r"$h_{t}$", GRAY_F, GRAY_E, 13)
    rbox(ax, 10.8, 7.35, 1.4, 0.7, r"$h_{t}$", GRAY_F, GRAY_E, 13)
    ax.add_patch(Circle((3.5, 0.72), 0.42, fc=TEAL_F, ec=TEAL_E, lw=1.6, zorder=5))
    ax.text(3.5, 0.72, r"$X_t$", ha="center", va="center", fontsize=12, zorder=6)
    x1z, xplus = 7.8, 10.8
    # h 하이웨이
    wire(ax, [(1.9, yH), (x1z-0.3, yH)], head=False)
    opc(ax, x1z, yH, "×")
    wire(ax, [(x1z+0.3, yH), (xplus-0.3, yH)], head=False)
    opc(ax, xplus, yH, "+")
    wire(ax, [(xplus+0.3, yH), (12.45, yH)])
    wire(ax, [(xplus, yH+0.3), (xplus, 7.0)])                       # + → 위 h_t
    # 입력 버스
    wire(ax, [(3.5, 1.14), (3.5, ybus), (9.4, ybus)], head=False)
    for gx in [4.3, 5.7, 9.4]:                                     # rσ, tanh, zσ 로 x_t
        wire(ax, [(gx, ybus), (gx, (yG if gx != 5.7 else ymid)-0.32)], head=False)
    # 게이트 σ
    rbox(ax, 4.3, yG, 0.95, 0.6, r"$\sigma$", PINK_F, PINK_E, 14)   # reset r
    rbox(ax, 9.4, yG, 0.95, 0.6, r"$\sigma$", PINK_F, PINK_E, 14)   # update z
    # reset ×  (r ⊙ h_{t-1})
    opc(ax, 4.3, ymid, "×", r=0.24)
    wire(ax, [(4.3, yG+0.3), (4.3, ymid-0.24)])                    # r → ×
    wire(ax, [(2.3, yH-0.35), (2.3, ymid), (4.3-0.24, ymid)], head=True)  # h_{t-1} → ×
    # candidate tanh
    rbox(ax, 5.7, ymid, 1.0, 0.56, "tanh", PINK_F, PINK_E, 11)
    wire(ax, [(4.54, ymid), (5.7-0.5, ymid)])                     # reset× → tanh
    # z×  (z ⊙ h~)
    opc(ax, xplus, ymid, "×", r=0.24)
    wire(ax, [(6.2, ymid), (xplus-0.24, ymid)])                   # tanh(h~) → z×
    wire(ax, [(9.4, yG+0.3), (9.4, 3.2), (xplus, 3.2), (xplus, ymid-0.24)], head=True)  # z → z× (아래로 입력)
    wire(ax, [(xplus, ymid+0.24), (xplus, yH-0.3)])              # z× → +  (위로 출력)
    # (1-z) → (1-z)×  (위로 우회, 주황)
    wire(ax, [(9.4, 3.2), (9.4, 5.6), (x1z, 5.6), (x1z, yH+0.3)], head=True, color="#b45309")
    ax.text(8.35, 5.72, r"$1-z_t$", fontsize=10, color="#b45309", zorder=6)
    ax.set_title("GRU 셀 아키텍처", fontsize=13, fontweight="bold")
    save(fig, "ch3_gru_arch.png")


if __name__ == "__main__":
    fig_lstm_cell()
    fig_gru_cell()
    print("done arch3")
