# -*- coding: utf-8 -*-
"""4장 우리-스타일 그림: 시계열 트랜스포머 계보."""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _ours import new_ax, save, heading, BRAND, CORAL, INDIGO, AMBER, INK, SUB, TINT


def branchcard(ax, cx, y, title, members, color, w=3.15, h=2.9):
    ax.add_patch(FancyBboxPatch((cx-w/2+0.06, y-0.09), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", fc="#dfe3e8", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((cx-w/2, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", fc="white", ec="#e4e8ec", lw=1.2, zorder=4))
    tw = 0.285*len(title) + 0.55
    ax.add_patch(FancyBboxPatch((cx-tw/2, y+h-0.28), tw, 0.5,
                 boxstyle="round,pad=0.02,rounding_size=0.24", fc=color, ec="none", zorder=5))
    ax.text(cx, y+h-0.02, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="white", zorder=6)
    for i, (nm, yr) in enumerate(members):
        yy = y+h-0.95-i*0.62
        ax.add_patch(FancyBboxPatch((cx-w/2+0.25, yy-0.24), w-0.5, 0.48,
                     boxstyle="round,pad=0.02,rounding_size=0.12", fc=TINT["gray"], ec=color, lw=1.3, zorder=5))
        ax.text(cx-w/2+0.45, yy, nm, ha="left", va="center", fontsize=11,
                fontweight="bold", color=INK, zorder=6)
        ax.text(cx+w/2-0.42, yy, yr, ha="right", va="center", fontsize=9.5,
                color=SUB, zorder=6)


def ar(ax, p0, p1, color, lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=lw,
                 color=color, zorder=2, shrinkA=3, shrinkB=3,
                 connectionstyle=f"arc3,rad={rad}"))


def fig_ts_transformer_lineage():
    fig, ax = new_ax(11.4, 6.6, (0, 16), (0, 9.4))
    heading(ax, 0.4, 8.9, "시계열 트랜스포머의 계보")
    ax.text(0.8, 8.25, "표준 트랜스포머의 이차 복잡도와 시계열 특성을 겨냥해 네 방향으로 변형이 발전했습니다.",
            fontsize=11, color=SUB, va="center")

    # 루트
    rx, ry, rw, rh = 5.3, 7.0, 5.4, 0.98
    ax.add_patch(FancyBboxPatch((rx+0.06, ry-0.09), rw, rh,
                 boxstyle="round,pad=0.02,rounding_size=0.2", fc="#dfe3e8", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh,
                 boxstyle="round,pad=0.02,rounding_size=0.2", fc=INK, ec="none", zorder=4))
    ax.text(rx+rw/2, ry+rh/2+0.12, "Attention is All You Need", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="white", zorder=5)
    ax.text(rx+rw/2, ry+rh/2-0.24, "Vaswani et al., 2017", ha="center", va="center",
            fontsize=9.5, color="#c9ccd2", zorder=5)

    xs = [2.15, 6.05, 9.95, 13.85]
    colors = [BRAND, INDIGO, AMBER, CORAL]
    branches = [
        ("효율적 어텐션", [("LogSparse", "2019"), ("Informer", "2021"), ("Pyraformer", "2022")]),
        ("시계열 분해 결합", [("Autoformer", "2021"), ("FEDformer", "2022")]),
        ("표현·패치화", [("PatchTST", "2023")]),
        ("해석·불확실성", [("TFT", "2021")]),
    ]
    for (t, ms), cx, c in zip(branches, xs, colors):
        branchcard(ax, cx, 1.2, t, ms, c)
        ar(ax, (rx+rw/2, ry), (cx, 4.15), c, rad=(0.0 if abs(cx-8)<3 else 0.06*(1 if cx>8 else -1)))

    save(fig, "ch4_ts_transformer_ours.png")


if __name__ == "__main__":
    fig_ts_transformer_lineage()
    print("done ours ch4")
