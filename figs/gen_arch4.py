# -*- coding: utf-8 -*-
"""4장 아키텍처: 트랜스포머 인코더-디코더, 어텐션(스케일드/멀티헤드)."""
import matplotlib.pyplot as plt
from _style import save
from _arch import (box, opnode, arrow, setup,
                   BOX_BLUE, EDGE_BLUE, BOX_GREEN, EDGE_GREEN,
                   BOX_ORANGE, EDGE_ORANGE, BOX_PURPLE, EDGE_PURPLE,
                   BOX_GRAY, EDGE_GRAY, BOX_RED, EDGE_RED)


def fig_transformer():
    fig, ax = plt.subplots(figsize=(8.6, 6.6), constrained_layout=True)
    setup(ax, (0, 12), (0, 12.5))
    W = 3.4
    # ---------- 인코더 (왼쪽) ----------
    ex = 0.7
    box(ax, ex, 0.6, W, 0.7, "입력 임베딩 + 위치 인코딩", fc=BOX_GRAY, ec=EDGE_GRAY, fs=9.5)
    box(ax, ex, 2.2, W, 0.7, "멀티 헤드 셀프 어텐션", fc=BOX_BLUE, ec=EDGE_BLUE, fs=9.5)
    box(ax, ex, 3.15, W, 0.55, "Add & Norm", fc="#fff7cc", ec="#b7791f", fs=9)
    box(ax, ex, 4.0, W, 0.7, "피드포워드", fc=BOX_GREEN, ec=EDGE_GREEN, fs=9.5)
    box(ax, ex, 4.95, W, 0.55, "Add & Norm", fc="#fff7cc", ec="#b7791f", fs=9)
    ax.add_patch(plt.Rectangle((ex-0.25, 2.0), W+0.5, 3.7, fill=False,
                 ec=EDGE_BLUE, lw=1.4, ls="--"))
    ax.text(ex+W+0.05, 5.55, "N×", fontsize=11, color=EDGE_BLUE, fontweight="bold")
    for y0, y1 in [(1.3, 2.2), (2.9, 3.15), (3.7, 4.0), (4.7, 4.95)]:
        arrow(ax, (ex+W/2, y0), (ex+W/2, y1), color=EDGE_GRAY, lw=1.6)
    ax.text(ex+W/2, 6.0, "인코더", fontsize=12, ha="center", fontweight="bold", color=EDGE_BLUE)
    # ---------- 디코더 (오른쪽) ----------
    dx = 7.2
    box(ax, dx, 0.6, W, 0.7, "출력 임베딩 + 위치 인코딩", fc=BOX_GRAY, ec=EDGE_GRAY, fs=9.5)
    box(ax, dx, 2.2, W, 0.7, "마스크드 셀프 어텐션", fc=BOX_PURPLE, ec=EDGE_PURPLE, fs=9.5)
    box(ax, dx, 3.15, W, 0.55, "Add & Norm", fc="#fff7cc", ec="#b7791f", fs=9)
    box(ax, dx, 4.0, W, 0.7, "인코더-디코더 어텐션", fc=BOX_ORANGE, ec=EDGE_ORANGE, fs=9.5)
    box(ax, dx, 4.95, W, 0.55, "Add & Norm", fc="#fff7cc", ec="#b7791f", fs=9)
    box(ax, dx, 5.8, W, 0.7, "피드포워드", fc=BOX_GREEN, ec=EDGE_GREEN, fs=9.5)
    box(ax, dx, 6.75, W, 0.55, "Add & Norm", fc="#fff7cc", ec="#b7791f", fs=9)
    ax.add_patch(plt.Rectangle((dx-0.25, 2.0), W+0.5, 5.5, fill=False,
                 ec=EDGE_PURPLE, lw=1.4, ls="--"))
    ax.text(dx+W+0.05, 7.35, "N×", fontsize=11, color=EDGE_PURPLE, fontweight="bold")
    for y0, y1 in [(1.3, 2.2), (2.9, 3.15), (3.7, 4.0), (4.7, 4.95), (5.5, 5.8), (6.5, 6.75)]:
        arrow(ax, (dx+W/2, y0), (dx+W/2, y1), color=EDGE_GRAY, lw=1.6)
    box(ax, dx, 8.0, W, 0.62, "선형 + 소프트맥스", fc=BOX_RED, ec=EDGE_RED, fs=9.5)
    arrow(ax, (dx+W/2, 7.3), (dx+W/2, 8.0), color=EDGE_GRAY, lw=1.6)
    arrow(ax, (dx+W/2, 8.62), (dx+W/2, 9.3), color="#111", lw=1.8)
    ax.text(dx+W/2, 9.6, "다음 토큰 확률", fontsize=10, ha="center", fontweight="bold")
    ax.text(dx+W/2, 6.0, "디코더", fontsize=12, ha="center", fontweight="bold", color=EDGE_PURPLE)
    # 인코더 출력 -> 디코더 cross-attention (키·값) : 박스 관통 없이 아래로 완만히
    arrow(ax, (ex+W+0.05, 4.9), (dx-0.05, 4.35), color=EDGE_ORANGE, lw=2.2, rad=0.18)
    ax.text(5.95, 4.95, "키·값", fontsize=9.5, color=EDGE_ORANGE, ha="center", fontweight="bold")
    ax.set_title("트랜스포머 인코더-디코더 아키텍처", fontsize=13, fontweight="bold")
    save(fig, "ch4_transformer_arch.png")


def fig_attention():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.6), constrained_layout=True)
    # (a) 스케일드 닷-프로덕트 어텐션
    ax = axes[0]; setup(ax, (0, 6.5), (0, 10))
    cx = 2.7
    box(ax, 0.7, 0.5, 1.2, 0.6, "Q", fc=BOX_BLUE, ec=EDGE_BLUE)
    box(ax, 3.5, 0.5, 1.2, 0.6, "K", fc=BOX_GREEN, ec=EDGE_GREEN)
    steps = [(r"MatMul  ($QK^{\top}$)", 2.0, BOX_GRAY, EDGE_GRAY),
             (r"Scale  ($\div\sqrt{d_k}$)", 3.1, "#fff7cc", "#b7791f"),
             ("SoftMax", 4.2, BOX_RED, EDGE_RED),
             (r"MatMul  ($\times V$)", 5.6, BOX_GRAY, EDGE_GRAY)]
    for t, y, fc, ec in steps:
        box(ax, cx-1.5, y, 3.0, 0.6, t, fc=fc, ec=ec, fs=9.5)
    arrow(ax, (1.3, 1.1), (2.1, 2.0), color=EDGE_BLUE, lw=1.5)
    arrow(ax, (4.1, 1.1), (3.3, 2.0), color=EDGE_GREEN, lw=1.5)
    for y0, y1 in [(2.6, 3.1), (3.7, 4.2), (4.8, 5.6)]:
        arrow(ax, (cx, y0), (cx, y1), color=EDGE_GRAY, lw=1.5)
    box(ax, 4.9, 5.6, 1.1, 0.6, "V", fc=BOX_ORANGE, ec=EDGE_ORANGE)   # V는 옆에서 입력
    arrow(ax, (4.9, 5.9), (4.2, 5.9), color=EDGE_ORANGE, lw=1.5)
    arrow(ax, (cx, 6.2), (cx, 7.1), color="#111", lw=1.8)
    ax.text(cx, 7.45, "출력", fontsize=10, ha="center", fontweight="bold")
    ax.set_title("(a) 스케일드 닷-프로덕트 어텐션", fontsize=11, fontweight="bold")
    # (b) 멀티 헤드 어텐션
    ax = axes[1]; setup(ax, (0, 6), (0, 10))
    box(ax, 0.6, 0.5, 1.3, 0.6, "Q", fc=BOX_BLUE, ec=EDGE_BLUE)
    box(ax, 2.35, 0.5, 1.3, 0.6, "K", fc=BOX_GREEN, ec=EDGE_GREEN)
    box(ax, 4.1, 0.5, 1.3, 0.6, "V", fc=BOX_ORANGE, ec=EDGE_ORANGE)
    for x in (1.25, 3.0, 4.75):
        box(ax, x-0.9, 1.8, 1.8, 0.55, "Linear", fc=BOX_GRAY, ec=EDGE_GRAY, fs=8.5)
        arrow(ax, (x, 1.1), (x, 1.8), color=EDGE_GRAY, lw=1.3)
    box(ax, 0.7, 3.2, 4.6, 0.75, "스케일드 닷-프로덕트\n어텐션  (헤드 h개 병렬)",
        fc=BOX_RED, ec=EDGE_RED, fs=9)
    for x in (1.25, 3.0, 4.75):
        arrow(ax, (x, 2.35), (x, 3.2), color=EDGE_GRAY, lw=1.3)
    box(ax, 1.4, 5.0, 3.2, 0.6, "Concat", fc=BOX_GRAY, ec=EDGE_GRAY, fs=9.5)
    arrow(ax, (3.0, 3.95), (3.0, 5.0), color=EDGE_GRAY, lw=1.4)
    box(ax, 1.4, 6.4, 3.2, 0.6, r"Linear  ($W^{O}$)", fc=BOX_PURPLE, ec=EDGE_PURPLE, fs=9.5)
    arrow(ax, (3.0, 5.6), (3.0, 6.4), color=EDGE_GRAY, lw=1.4)
    arrow(ax, (3.0, 7.0), (3.0, 7.8), color="#111", lw=1.6)
    ax.text(3.0, 8.15, "출력", fontsize=10, ha="center", fontweight="bold")
    ax.set_title("(b) 멀티 헤드 어텐션", fontsize=11, fontweight="bold")
    save(fig, "ch4_attention_arch.png")


if __name__ == "__main__":
    fig_transformer()
    fig_attention()
    print("done arch4")
