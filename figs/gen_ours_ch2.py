# -*- coding: utf-8 -*-
"""2장 우리-스타일 그림: 옵티마이저 계보(genealogy)."""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _ours import new_ax, save, heading, BRAND, CORAL, INDIGO, AMBER, ROSE, INK, SUB, TINT


def lnode(ax, cx, cy, name, year, color, tint, w=2.15, h=0.92, hi=False):
    ax.add_patch(FancyBboxPatch((cx-w/2+0.06, cy-h/2-0.08), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.16", fc="#dfe3e8", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.16",
                 fc=(color if hi else tint), ec=color, lw=(2.4 if hi else 1.8), zorder=4))
    ax.text(cx, cy+0.12, name, ha="center", va="center", fontsize=13,
            fontweight="bold", color=("white" if hi else INK), zorder=5)
    if year:
        ax.text(cx, cy-0.24, year, ha="center", va="center", fontsize=9.5,
                color=("white" if hi else SUB), zorder=5)


def famcard(ax, cx, y, title, members, color, w=2.72, h=2.05):
    ax.add_patch(FancyBboxPatch((cx-w/2+0.06, y-0.09), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", fc="#dfe3e8", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((cx-w/2, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", fc="white", ec="#e4e8ec", lw=1.2, zorder=4))
    tw = 0.30*len(title) + 0.5
    ax.add_patch(FancyBboxPatch((cx-tw/2, y+h-0.26), tw, 0.48,
                 boxstyle="round,pad=0.02,rounding_size=0.22", fc=color, ec="none", zorder=5))
    ax.text(cx, y+h-0.01, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="white", zorder=6)
    for i, m in enumerate(members):
        ax.text(cx, y+h-0.72-i*0.42, m, ha="center", va="center", fontsize=11,
                color=INK, fontweight="bold", zorder=6)


def ar(ax, p0, p1, color, lw=2.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, lw=lw,
                 color=color, zorder=2, shrinkA=3, shrinkB=3,
                 connectionstyle=f"arc3,rad={rad}"))


def fig_optimizer_genealogy():
    fig, ax = new_ax(11.2, 6.6, (0, 16), (0, 9.4))
    heading(ax, 0.4, 8.9, "옵티마이저의 계보")
    ax.text(0.8, 8.25, "SGD에서 적응적 학습률과 모멘텀이 더해져 Adam으로 수렴하고, 이후 여러 특화 계열로 확장됩니다.",
            fontsize=11, color=SUB, va="center")

    # 상단 계보 스파인
    yL = 6.7
    spine = [(1.9, "SGD", "기본"), (4.9, "AdaGrad", "2011"),
             (8.0, "RMSProp", "2012"), (11.0, "Adam", "2014"), (14.0, "AdamW", "2017")]
    colors = [INDIGO, AMBER, AMBER, BRAND, BRAND]
    tints = [TINT["indigo"], TINT["amber"], TINT["amber"], TINT["brand"], TINT["brand"]]
    for (cx, nm, yr), c, ti in zip(spine, colors, tints):
        lnode(ax, cx, yL, nm, yr, c, ti, hi=(nm == "Adam"))
    for i in range(len(spine)-1):
        ar(ax, (spine[i][0]+1.1, yL), (spine[i+1][0]-1.1, yL), "#9aa2ad")
    # 모멘텀 유입
    ax.text(6.7, 7.72, "+ 모멘텀", ha="center", fontsize=10.5, color=INDIGO, fontweight="bold")
    ar(ax, (6.7, 7.52), (10.6, yL+0.5), INDIGO, lw=2.0, rad=-0.16)
    # Adam 변형 칩
    ax.text(11.0, 5.55, "변형: AMSGrad · NAdam · RAdam · AdaBelief", ha="center",
            fontsize=9.5, color=SUB, style="italic")

    # 하단 현대 계열 밴드 라벨
    ax.text(0.8, 4.35, "현대 계열 (Adam 이후의 확장)", fontsize=12, color=INK, fontweight="bold")
    ar(ax, (11.0, yL-0.55), (8.0, 4.75), BRAND, lw=2.2, rad=0.1)

    fams = [
        ("부호 기반", ["Lion"], CORAL),
        ("대규모 배치", ["LARS", "LAMB"], INDIGO),
        ("메모리 효율", ["Adafactor", "GaLore"], AMBER),
        ("2차·곡률", ["Sophia", "Shampoo", "Muon"], BRAND),
        ("평탄 손실", ["SAM"], ROSE),
    ]
    xs = [1.9, 4.85, 7.8, 10.75, 13.7]
    for (t, ms, c), cx in zip(fams, xs):
        famcard(ax, cx, 1.55, t, ms, c)

    save(fig, "ch2_optim_genealogy_ours.png")


if __name__ == "__main__":
    fig_optimizer_genealogy()
    print("done ours ch2")
