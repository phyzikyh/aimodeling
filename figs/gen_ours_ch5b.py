# -*- coding: utf-8 -*-
"""5장 CNN 기초 확장 그림: 전체구조·입력데이터·계산단계·부피변화·풀링·차원."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon
from _ours import new_ax, save, heading, numgrid, volume, chip, SHADE, \
    BRAND, CORAL, INDIGO, AMBER, ROSE, INK, SUB, TINT


def _ar(ax, p0, p1, color=SUB, lw=2.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, lw=lw,
                 color=color, zorder=5, shrinkA=2, shrinkB=2))


def _grid(ax, x, y, R, C, u, ec="#c7ccd3", fc="#f4f6f8", hi=None, hicolor=BRAND, lw=1.5):
    for i in range(R):
        for j in range(C):
            ax.add_patch(Rectangle((x+j*u, y+(R-1-i)*u), u, u, fc=fc, ec=ec, lw=lw, zorder=2))
    if hi:
        for (i, j) in hi:
            ax.add_patch(Rectangle((x+j*u, y+(R-1-i)*u), u, u, fc="none", ec=hicolor, lw=3.0, zorder=4))
    return (x, y, C*u, R*u)


# ---------- 1) CNN 전체 구조 ----------
def fig_cnn_structure():
    fig, ax = new_ax(14.0, 6.0, (0, 33.5), (0, 14.4))
    heading(ax, 0.5, 13.4, "합성곱 신경망의 전체 구조")
    ax.text(1.0, 12.45, "입력층 → (합성곱층 + 풀링층) 반복 → 완전연결층 → 출력층.  공간 크기는 줄고 채널 수는 늘어납니다.",
            fontsize=10.5, color=SUB, va="center")

    cy, TY, SY = 7.6, 10.7, 4.4

    def vb(x, fw, dp, kind, title, shape):
        r = volume(ax, x, cy - fw / 2, fw, fw, dp, kind)
        ax.text(x + fw / 2, TY, title, ha="center", fontsize=11, color=INK, fontweight="bold")
        ax.text(x + fw / 2, SY, shape, ha="center", fontsize=9.5, color=SUB)
        return r

    r = vb(1.0, 2.8, 0.3, "gray", "입력층", "28×28×1")
    _ar(ax, (r + 0.15, cy), (5.0, cy), BRAND)
    r = vb(5.0, 2.4, 1.4, "brand", "합성곱층", "26×26×32")
    _ar(ax, (r + 0.15, cy), (9.7, cy), INDIGO)
    r = vb(9.7, 1.6, 1.4, "indigo", "풀링층", "13×13×32")
    _ar(ax, (r + 0.15, cy), (13.4, cy), BRAND)
    r = vb(13.4, 1.35, 2.3, "brand", "합성곱층", "11×11×64")
    _ar(ax, (r + 0.15, cy), (17.3, cy), INDIGO)
    r = vb(17.3, 0.9, 2.3, "indigo", "풀링층", "5×5×64")

    # Flatten -> FC -> 출력
    _ar(ax, (r + 0.15, cy), (21.2, cy), SUB)
    ax.add_patch(FancyBboxPatch((21.2, cy - 2.6), 0.5, 5.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=SHADE["gray"][0], ec="#4d5766", lw=1.3, zorder=3))
    ax.text(21.45, TY, "Flatten", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(21.45, SY, "1600", ha="center", fontsize=9.5, color=SUB)
    _ar(ax, (21.9, cy), (24.3, cy), CORAL)
    ax.add_patch(FancyBboxPatch((24.3, cy - 1.7), 0.5, 3.4, boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=SHADE["coral"][0], ec="#4d5766", lw=1.3, zorder=3))
    ax.text(24.55, TY, "완전연결층", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(24.55, SY, "128", ha="center", fontsize=9.5, color=SUB)
    _ar(ax, (24.9, cy), (27.4, cy), ROSE)
    for k in range(10):
        ax.add_patch(Rectangle((27.4, cy - 2.0 + k * 0.4), 1.6, 0.32,
                     fc=("#f2a6c9" if k != 3 else ROSE), ec="#b84a86", lw=1.0, zorder=3))
    ax.text(28.2, TY, "출력층", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(28.2, SY, "10 (softmax)", ha="center", fontsize=9.5, color=SUB)

    # 구역 밴드
    ax.annotate("", xy=(19.4, 2.9), xytext=(1.0, 2.9),
                arrowprops=dict(arrowstyle="-", color="#c6eadf", lw=8, alpha=0.6))
    ax.text(10.0, 2.4, "특징 추출부 (합성곱 + 풀링 반복)", ha="center", fontsize=10, color=BRAND, fontweight="bold")
    ax.annotate("", xy=(29.0, 2.9), xytext=(21.2, 2.9),
                arrowprops=dict(arrowstyle="-", color="#fce4ef", lw=8, alpha=0.7))
    ax.text(25.0, 2.4, "분류부", ha="center", fontsize=10, color=ROSE, fontweight="bold")
    save(fig, "ch5_cnn_structure_ours.png")


# ---------- 2) 입력 데이터: 흑백 vs 컬러 ----------
def fig_input_data():
    fig, ax = new_ax(12.0, 5.6, (0, 24), (0, 11.2))
    heading(ax, 0.5, 10.3, "입력층 — 이미지는 숫자 격자")
    ax.text(1.0, 9.45, "흑백은 채널 1개(밝기), 컬러는 빨강·초록·파랑 3개 채널로 이루어집니다.",
            fontsize=10.5, color=SUB, va="center")

    # 흑백 6x6
    G = np.array([[52, 60, 70, 80, 90, 88], [40, 120, 150, 160, 130, 70],
                  [30, 140, 255, 250, 150, 60], [35, 150, 248, 240, 140, 66],
                  [45, 110, 150, 150, 120, 72], [60, 66, 74, 82, 90, 95]])
    x0, y0, w0, h0 = numgrid(ax, 1.0, 2.6, G, u=0.85, tint="#eef1f4", fs=8.5)
    ax.text(x0 + w0 / 2, y0 + h0 + 0.45, "흑백 이미지", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(x0 + w0 / 2, y0 - 0.55, "(H, W, 1) · 픽셀 = 밝기 하나 (0~255)", ha="center", fontsize=9.5, color=SUB)

    # 컬러: R/G/B 세 평면(등축 오프셋)
    cx, cyb = 15.5, 3.0
    planes = [("B", "#9db2f6", 0.0), ("G", "#79cfb8", 0.0), ("R", "#f7a79a", 0.0)]
    off = 0.55
    for idx, (nm, col, _) in enumerate(planes):
        ox = cx + idx * off
        oy = cyb + idx * off
        ax.add_patch(Rectangle((ox, oy), 3.6, 3.6, fc=col, ec="#4d5766", lw=1.4, zorder=3 + idx))
        for g in range(1, 6):
            ax.plot([ox + g * 0.6, ox + g * 0.6], [oy, oy + 3.6], color="#ffffff", lw=0.8, alpha=0.7, zorder=3 + idx)
            ax.plot([ox, ox + 3.6], [oy + g * 0.6, oy + g * 0.6], color="#ffffff", lw=0.8, alpha=0.7, zorder=3 + idx)
        ax.text(ox + 3.75, oy + 3.4, nm, fontsize=12, color="#2b3440", fontweight="bold", zorder=9)
    ax.text(cx + 1.8 + off, cyb + 3.6 + 1.15, "컬러 이미지", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(cx + 1.8 + off, cyb - 0.75, "(H, W, 3) · 픽셀 = (R, G, B) 세 값", ha="center", fontsize=9.5, color=SUB)
    save(fig, "ch5_input_data_ours.png")


# ---------- 3) 합성곱 계산 단계 (몇 칸 -> ... -> 완성) ----------
def fig_conv_steps():
    fig, ax = new_ax(12.5, 7.4, (0, 25), (0, 14.8))
    heading(ax, 0.5, 14.0, "합성곱 계산 — 커널을 밀며 특징 맵 완성하기 (stride=1)")

    I = np.array([[1, 2, 0, 1, 3], [0, 1, 2, 3, 1], [1, 0, 1, 2, 0], [2, 1, 0, 1, 1], [0, 2, 1, 0, 2]])
    K = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    O = np.array([[4, 7, 7], [4, 7, 7], [4, 4, 5]])

    def step(y, hi_cols, title, expr, val, label=True):
        hi = [(r, c) for r in range(3) for c in range(hi_cols, 3 + hi_cols)]
        numgrid(ax, 1.0, y, I, u=0.66, tint="#eef1f4", hi=hi, hicolor=BRAND)
        if label:
            ax.text(1.0 + 2.5 * 0.66, y - 0.42, "입력 (5×5)", ha="center", fontsize=8.5, color=SUB)
        ax.text(5.7, y + 1.7, title, fontsize=10.5, color=BRAND, fontweight="bold")
        ax.text(5.7, y + 1.0, expr, fontsize=11, color=INK)
        ax.text(5.7, y + 0.35, val, fontsize=11, color=INK)

    step(10.2, 0, "① 왼쪽 위 — 커널이 1인 자리만 더함",
         "O(0,0) = 1+0+1 + 0+1+0 + 1+0+1", "         = 4", label=False)
    step(5.0, 1, "② 한 칸 오른쪽",
         "O(0,1) = 2+0+1 + 0+2+0 + 0+0+2", "         = 7", label=True)

    ax.text(4.6, 4.4, "...", fontsize=22, color=SUB, ha="center", fontweight="bold")

    # 커널
    numgrid(ax, 15.6, 10.6, K, u=0.8, tint="#e9edfd", ec="#c9d4fb")
    ax.text(15.6 + 1.5 * 0.8, 10.6 - 0.5, "커널 K (3×3)", ha="center", fontsize=10, color=INDIGO)

    # 완성본
    xo, yo, wo, ho = numgrid(ax, 18.9, 2.2, O, u=1.0, tint="#ffe9e5", ec="#f6c3ba",
                             hi=[(0, 0), (0, 1)], hicolor=CORAL)
    ax.text(xo + wo / 2, yo + ho + 0.5, "완성된 특징 맵 (3×3)", ha="center", fontsize=10.5, color=CORAL, fontweight="bold")
    ax.text(xo + wo / 2, yo - 0.5, "5×5 입력 → 3×3 출력", ha="center", fontsize=9.5, color=SUB)
    _ar(ax, (12.5, 6.5), (18.6, 4.0), CORAL)
    save(fig, "ch5_conv_steps_ours.png")


# ---------- 4) 부피 변화: 흑백/컬러/다필터 ----------
def fig_conv_volume():
    fig, ax = new_ax(12.5, 7.6, (0, 25), (0, 15.2))
    heading(ax, 0.5, 14.4, "합성곱이 부피를 바꾸는 방식")

    def row(y, in_shape, in_dp, in_kind, k_txt, out_shape, out_dp, out_ch, note):
        volume(ax, 1.0, y, 2.4, 2.4, in_dp, in_kind)
        ax.text(2.2, y - 0.55, in_shape, ha="center", fontsize=9.5, color=SUB)
        _ar(ax, (4.4, y + 1.2), (8.6, y + 1.2), BRAND)
        ax.text(6.5, y + 2.0, k_txt, ha="center", fontsize=9.5, color=BRAND, fontweight="bold")
        volume(ax, 9.0, y + 0.3, 1.8, 1.8, out_dp, out_kind_of(out_ch))
        ax.text(9.9, y - 0.55, out_shape, ha="center", fontsize=9.5, color=SUB)
        ax.text(13.2, y + 1.2, note, fontsize=10, color=INK, va="center")

    def out_kind_of(ch):
        return "coral"

    row(11.2, "(8, 8, 1) 흑백", 0.25, "gray", "3×3 커널 1개", "(6, 6, 1)", 0.25, 1,
        "채널 1개 입력 → 커널 1개 → 출력 1채널. 8→6으로 줄어듭니다.")
    row(6.6, "(8, 8, 3) 컬러", 0.9, "brand", "3×3×3 커널 1개", "(6, 6, 1)", 0.25, 1,
        "3채널을 한 번에 곱-합 → 출력은 여전히 1채널.")
    row(1.6, "(8, 8, 3) 컬러", 0.9, "brand", "3×3×3 커널 2개", "(6, 6, 2)", 0.9, 2,
        "필터가 2개면 출력 채널도 2개. 필터 수 = 출력 채널 수.")
    save(fig, "ch5_conv_volume_ours.png")


# ---------- 5) 풀링: 최대 vs 평균 ----------
def fig_pool_maxavg():
    fig, ax = new_ax(12.0, 5.4, (0, 24), (0, 10.8))
    heading(ax, 0.5, 9.9, "풀링층 — 최대 풀링과 평균 풀링")
    ax.text(1.0, 9.05, "겹치지 않는 2×2 영역을 값 하나로 요약해 특징 맵을 절반으로 줄입니다.",
            fontsize=10.5, color=SUB, va="center")

    F = np.array([[1, 3, 2, 4], [5, 6, 1, 2], [0, 2, 3, 1], [4, 1, 0, 5]])
    zones = [(0, 0, BRAND), (0, 2, AMBER), (2, 0, INDIGO), (2, 2, CORAL)]
    hi = []
    for (r, c, _) in zones:
        hi += [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
    numgrid(ax, 1.0, 2.2, F, u=1.05, tint="#eef1f4", hi=hi, hicolor="#9aa3af")
    ax.text(1.0 + 2 * 1.05, 1.6, "입력 특징 맵 (4×4)", ha="center", fontsize=10, color=SUB)

    Mx = np.array([[6, 4], [4, 5]])
    Av = np.array([[3.75, 2.25], [1.75, 2.25]])
    _ar(ax, (5.6, 6.0), (8.4, 7.3), CORAL)
    _ar(ax, (5.6, 4.4), (8.4, 3.1), INDIGO)
    x1, y1, w1, h1 = numgrid(ax, 9.0, 6.3, Mx, u=1.1, tint="#ffe9e5", ec="#f6c3ba", fs=13)
    ax.text(x1 + w1 / 2, y1 - 0.45, "최대 풀링 (2×2)", ha="center", fontsize=10, color=CORAL, fontweight="bold")
    x2, y2, w2, h2 = numgrid(ax, 9.0, 2.0, Av, u=1.1, tint="#e9edfd", ec="#c9d4fb", fs=12)
    ax.text(x2 + w2 / 2, y2 - 0.45, "평균 풀링 (2×2)", ha="center", fontsize=10, color=INDIGO, fontweight="bold")

    ax.text(15.2, 6.85, "각 2×2에서 가장 큰 값", fontsize=9.5, color=SUB, va="center")
    ax.text(15.2, 2.55, "각 2×2의 평균값", fontsize=9.5, color=SUB, va="center")
    ax.text(15.2, 4.7, "예) 왼쪽 위 [[1,3],[5,6]]\n  최대=6,  평균=(1+3+5+6)/4=3.75",
            fontsize=9.5, color=INK, va="center")
    save(fig, "ch5_pool_maxavg_ours.png")


# ---------- 6) 1D / 2D / 3D 합성곱 ----------
def fig_conv_dims():
    fig, ax = new_ax(12.5, 4.8, (0, 25), (0, 9.6))
    heading(ax, 0.5, 8.8, "합성곱의 차원 — 1D · 2D · 3D")

    # 1D
    ax.text(3.6, 7.2, "1D 합성곱", ha="center", fontsize=11, color=INK, fontweight="bold")
    _grid(ax, 0.8, 4.6, 1, 8, 0.7, hi=[(0, 0), (0, 1), (0, 2)], hicolor=BRAND)
    ax.text(3.6, 3.9, "커널이 한 축(시간)으로 이동", ha="center", fontsize=9, color=SUB)
    ax.text(3.6, 3.2, "시계열 · 오디오 · 텍스트", ha="center", fontsize=9.5, color=BRAND, fontweight="bold")

    # 2D
    ax.text(11.5, 7.2, "2D 합성곱", ha="center", fontsize=11, color=INK, fontweight="bold")
    _grid(ax, 9.6, 3.3, 5, 5, 0.62, hi=[(r, c) for r in range(3) for c in range(3)], hicolor=BRAND)
    ax.text(11.5, 2.6, "커널이 두 축(가로·세로)으로 이동", ha="center", fontsize=9, color=SUB)
    ax.text(11.5, 1.9, "이미지", ha="center", fontsize=9.5, color=BRAND, fontweight="bold")

    # 3D
    ax.text(20.0, 7.2, "3D 합성곱", ha="center", fontsize=11, color=INK, fontweight="bold")
    volume(ax, 17.6, 3.2, 3.0, 3.0, 1.6, "brand")
    ax.add_patch(Rectangle((18.2, 4.2), 1.1, 1.1, fc="none", ec=CORAL, lw=2.6, zorder=6))
    ax.text(20.0, 2.4, "커널이 세 축(가로·세로·깊이)으로 이동", ha="center", fontsize=9, color=SUB)
    ax.text(20.0, 1.7, "동영상 · CT · 볼륨 데이터", ha="center", fontsize=9.5, color=BRAND, fontweight="bold")
    save(fig, "ch5_conv_dims_ours.png")


if __name__ == "__main__":
    fig_cnn_structure()
    fig_input_data()
    fig_conv_steps()
    fig_conv_volume()
    fig_pool_maxavg()
    fig_conv_dims()
    print("done ours ch5b")
