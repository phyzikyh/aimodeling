# -*- coding: utf-8 -*-
"""5장 우리-스타일 그림(CNN 기초): 합성곱 계산·스트라이드/패딩·수용영역."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from _ours import new_ax, save, heading, numgrid, BRAND, CORAL, INDIGO, AMBER, INK, SUB, TINT


def _arrow(ax, p0, p1, color=SUB, lw=2.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, lw=lw,
                 color=color, zorder=3, shrinkA=2, shrinkB=2))


def _label(ax, cx, y, text, color=SUB, fs=10):
    ax.text(cx, y, text, ha="center", va="center", fontsize=fs, color=color)


def _grid(ax, x, y, R, C, u, ec="#c7ccd3", fc="#f4f6f8", hi=None, hicolor=BRAND, lw=1.6):
    """숫자 없는 빈 격자. (i,j)=상단 왼쪽부터."""
    for i in range(R):
        for j in range(C):
            ax.add_patch(Rectangle((x+j*u, y+(R-1-i)*u), u, u, fc=fc, ec=ec, lw=lw, zorder=2))
    if hi:
        for (i, j) in hi:
            ax.add_patch(Rectangle((x+j*u, y+(R-1-i)*u), u, u, fc="none", ec=hicolor, lw=3.0, zorder=4))
    return (x, y, C*u, R*u)


# ---------- 그림 1: 합성곱 계산(숫자로 따라가기) ----------
def fig_conv_calc():
    fig, ax = new_ax(12.0, 6.6, (0, 24), (0, 13.2))
    heading(ax, 0.5, 12.4, "합성곱 계산 — 숫자로 따라가기")
    ax.text(0.9, 11.55, "커널을 입력의 한 자리에 겹쳐 곱한 뒤 모두 더하면 특징 맵의 한 칸이 됩니다.",
            fontsize=11, color=SUB, va="center")

    I = np.array([[3, 0, 1, 2], [1, 5, 2, 0], [2, 1, 4, 3], [0, 2, 1, 1]])
    K = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    O = np.array([[-2, 6], [-6, 2]])

    hi_in = [(r, c) for r in range(3) for c in range(3)]
    x0, y0, w0, h0 = numgrid(ax, 1.2, 6.6, I, u=1.15, tint="#eef1f4", hi=hi_in, hicolor=BRAND)
    _label(ax, x0 + w0 / 2, y0 - 0.5, "입력 I (4×4)", INK, 11)

    ax.text(6.9, 9.2, "*", fontsize=26, ha="center", va="center", color=INK, fontweight="bold")
    _label(ax, 6.9, 8.5, "합성곱", SUB, 9.5)

    xk, yk, wk, hk = numgrid(ax, 8.1, 7.35, K, u=1.05, tint="#e9edfd", ec="#c9d4fb")
    _label(ax, xk + wk / 2, yk - 0.5, "커널 K (3×3)", INDIGO, 11)

    ax.text(12.6, 9.2, "=", fontsize=26, ha="center", va="center", color=INK, fontweight="bold")

    xo, yo, wo, ho = numgrid(ax, 13.6, 7.9, O, u=1.25, tint="#ffe9e5", ec="#f6c3ba",
                             hi=[(0, 0)], hicolor=CORAL)
    _label(ax, xo + wo / 2, yo - 0.5, "특징 맵 I*K (2×2)", CORAL, 11)

    # 계산 과정(강조된 좌상단 칸)
    ax.add_patch(FancyBboxPatch((1.2, 1.2), 21.4, 3.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                 fc="#f7fbf9", ec="#c6eadf", lw=1.4, zorder=1))
    ax.text(2.0, 3.9, "왼쪽 위 3×3 칸의 계산", fontsize=11, color=BRAND, fontweight="bold", va="center")
    ax.text(2.0, 2.9, "O(0,0) = (3·1 + 0·0 + 1·(-1)) + (1·2 + 5·0 + 2·(-2)) + (2·1 + 1·0 + 4·(-1))",
            fontsize=12.5, color=INK, va="center", family="DejaVu Sans")
    ax.text(2.0, 1.95, "         = 2 + (-2) + (-2) = -2",
            fontsize=12.5, color=INK, va="center", family="DejaVu Sans")
    save(fig, "ch5_conv_calc_ours.png")


# ---------- 그림 2: 스트라이드·패딩과 출력 크기 ----------
def fig_stride_pad():
    fig, ax = new_ax(12.6, 5.2, (0, 25.2), (0, 10.4))
    heading(ax, 0.5, 9.7, "스트라이드·패딩과 출력 크기")
    ax.text(0.9, 8.9, "출력 크기 " + r"$O=\lfloor (W-F+2P)/S \rfloor + 1$" +
            ".  아래는 5×5 입력에 3×3 커널을 쓴 세 경우입니다.",
            fontsize=10.5, color=SUB, va="center")

    def panel(cx, title, pad, stride, outR, formula, note_color):
        u = 0.62
        # 입력 5×5 (+패딩 링)
        gx = cx - (5 * u) / 2
        gy = 3.2
        if pad:
            ax.add_patch(Rectangle((gx - u, gy - u), (5 + 2) * u, (5 + 2) * u,
                         fc="#fdf1d8", ec="#e6b24b", lw=1.4, ls="--", zorder=1))
        _grid(ax, gx, gy, 5, 5, u, hi=[(r, c) for r in range(3) for c in range(3)], hicolor=BRAND)
        ax.text(cx, gy + 5 * u + 0.5, title, ha="center", fontsize=10.5, color=INK, fontweight="bold")
        # 출력 크기 표시
        ax.text(cx, gy - (u if pad else 0) - 0.55, formula, ha="center", fontsize=10, color=note_color,
                fontweight="bold")
        ax.text(cx, gy - (u if pad else 0) - 1.15, f"→ 출력 {outR}×{outR}", ha="center", fontsize=10.5,
                color=SUB)

    panel(4.2, "패딩 0 · 스트라이드 1", 0, 1, 3, r"$(5-3+0)/1 + 1 = 3$", BRAND)
    panel(12.6, "패딩 1 · 스트라이드 1", 1, 1, 5, r"$(5-3+2)/1 + 1 = 5$", AMBER)
    panel(21.0, "패딩 0 · 스트라이드 2", 0, 2, 2, r"$\lfloor(5-3)/2\rfloor + 1 = 2$", INDIGO)
    save(fig, "ch5_stride_pad_ours.png")


# ---------- 그림 3: 수용 영역 성장 ----------
def fig_receptive():
    fig, ax = new_ax(11.5, 5.6, (0, 23), (0, 11.2))
    heading(ax, 0.5, 10.4, "수용 영역 — 3×3을 두 번 쌓으면 5×5")
    ax.text(0.9, 9.6, "층을 쌓을수록 위층의 한 뉴런이 원래 입력에서 바라보는 영역이 넓어집니다.",
            fontsize=10.5, color=SUB, va="center")

    u = 0.8
    # 입력 7×7: 5×5 수용영역 강조
    ix = 1.4
    iy = 1.2
    _grid(ax, ix, iy, 7, 7, u, hi=[(r, c) for r in range(1, 6) for c in range(1, 6)], hicolor=CORAL)
    ax.text(ix + 3.5 * u, iy - 0.6, "입력 (7×7)\n붉은 5×5 = 수용 영역", ha="center", fontsize=10, color=SUB)

    # 은닉1 5×5: 3×3 강조
    hx = 10.0
    hy = 2.2
    _grid(ax, hx, hy, 5, 5, u, hi=[(r, c) for r in range(1, 4) for c in range(1, 4)], hicolor=BRAND)
    ax.text(hx + 2.5 * u, hy - 0.6, "은닉1 (5×5)\n3×3 합성곱 뒤", ha="center", fontsize=10, color=SUB)

    # 은닉2 3×3: 가운데 1칸 강조
    ox = 17.6
    oy = 3.0
    _grid(ax, ox, oy, 3, 3, u, hi=[(1, 1)], hicolor=INDIGO)
    ax.text(ox + 1.5 * u, oy - 0.6, "은닉2 (3×3)\n한 뉴런", ha="center", fontsize=10, color=SUB)

    _arrow(ax, (hx - 0.7, hy + 2.5 * u), (ix + 7 * u + 0.5, iy + 3.5 * u), BRAND)
    _arrow(ax, (ox - 0.7, oy + 1.5 * u), (hx + 5 * u + 0.5, hy + 2.5 * u), BRAND)
    ax.text(8.0, 8.2, "3×3", fontsize=10, color=BRAND, ha="center", fontweight="bold")
    ax.text(15.6, 8.2, "3×3", fontsize=10, color=BRAND, ha="center", fontweight="bold")
    save(fig, "ch5_receptive_ours.png")


if __name__ == "__main__":
    fig_conv_calc()
    fig_stride_pad()
    fig_receptive()
    print("done ours ch5")
