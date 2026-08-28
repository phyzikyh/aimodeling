# -*- coding: utf-8 -*-
"""'우리 스타일' 시연: 합성곱 특징 추출 (에디토리얼 카드 시스템)."""
import numpy as np
from _ours import (new_ax, save, heading, card, numgrid, node, connect, chip,
                   BRAND, CORAL, INDIGO, AMBER, ROSE, INK, SUB, TINT)
from matplotlib.patches import Circle


def conv_node(ax, cx, cy, r=0.4, color=BRAND):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=color, lw=2.4, zorder=6))
    for ang in (0, 45, 90, 135):
        a = np.deg2rad(ang); rr = r*0.55
        ax.plot([cx-rr*np.cos(a), cx+rr*np.cos(a)], [cy-rr*np.sin(a), cy+rr*np.sin(a)],
                color=color, lw=3, solid_capstyle="round", zorder=7)


def fig_conv_ours():
    fig, ax = new_ax(11.5, 5.6, (0, 17), (0, 9))
    heading(ax, 0.5, 8.25, "합성곱 — 필터로 특징을 뽑습니다")
    ax.text(0.85, 7.5, "필터를 미끄러뜨리며 비슷한 패턴을 찾고, 곱해서 더한 값(내적)이 특징 맵이 됩니다.",
            fontsize=11.5, color=SUB, va="center")

    # 입력 카드
    card(ax, 0.7, 2.5, 4.35, 3.9, title="입력 이미지", color=BRAND)
    img = [[1,1,1,2,2,2]]*4
    numgrid(ax, 1.0, 2.95, img, 0.6, tint=TINT["brand"], fs=11,
            hi=[(i, j) for i in range(4) for j in (1, 2, 3)], hicolor=BRAND)
    ax.text(3.05, 2.62, "세로 경계가 있는 이미지", fontsize=9.5, color=SUB, ha="center")

    # 합성곱 노드 (직접 그린 별표)
    conv_node(ax, 6.05, 4.35, r=0.4, color=BRAND)
    ax.text(6.05, 3.62, "합성곱", fontsize=9.5, color=BRAND, fontweight="bold", ha="center")
    connect(ax, (5.1, 4.35), (5.6, 4.35))

    # 필터 카드
    card(ax, 4.95, 0.35, 2.25, 2.55, title="필터", color=INDIGO)
    numgrid(ax, 5.35, 0.72, [[-1,0,1],[-1,0,1],[-1,0,1]], 0.55, tint=TINT["indigo"], fs=11)
    connect(ax, (6.05, 2.95), (6.05, 3.9))

    # 특징 맵 카드
    card(ax, 7.55, 2.9, 3.5, 2.9, title="특징 맵", color=CORAL)
    numgrid(ax, 8.35, 3.75, [[0,3,3,0]], 0.62, tint=TINT["coral"], fs=13)
    connect(ax, (6.45, 4.35), (7.65, 4.35))

    # 강조: 인라인 색칠 대신 태그 칩 (오른쪽 세로 정렬, 카드와 겹치지 않게)
    ax.text(14.0, 6.0, "요점", fontsize=11, color=INK, fontweight="bold", ha="center")
    chip(ax, 14.0, 5.25, "합성곱 = 곱하고 더하기(내적)", color=BRAND, fs=10.5)
    chip(ax, 14.0, 4.45, "닮은 곳에서 값이 커집니다", color=CORAL, fs=10.5)
    chip(ax, 14.0, 3.65, "커널은 학습으로 정해집니다", color=INDIGO, fs=10.5)
    save(fig, "ch5_conv_ours.png")


def _prism(ax, x, y, fw, fh, dp, shades):
    """등축 3D 특징맵 볼륨. 앞면 fw×fh(=공간), 깊이 dp(=채널). y=앞면 아래."""
    from matplotlib.patches import Polygon
    fr, tp, rt = shades
    ox, oy = dp*0.5, dp*0.42
    top = [(x, y+fh), (x+fw, y+fh), (x+fw+ox, y+fh+oy), (x+ox, y+fh+oy)]
    right = [(x+fw, y), (x+fw+ox, y+oy), (x+fw+ox, y+fh+oy), (x+fw, y+fh)]
    front = [(x, y), (x+fw, y), (x+fw, y+fh), (x, y+fh)]
    for pts, c in [(top, tp), (right, rt), (front, fr)]:
        ax.add_patch(Polygon(pts, closed=True, fc=c, ec="#4d5766", lw=1.3, zorder=3))
    return x+fw+ox   # 오른쪽 끝 x


def _bar(ax, x, yc, w, h, color):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, yc-h/2), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                 fc=color, ec="#4d5766", lw=1.3, zorder=3))


def fig_cnn_arch_ours():
    from matplotlib.patches import FancyBboxPatch
    from _ours import BRAND, AMBER, INDIGO, INK, SUB
    GRAY3 = ("#cfd4db", "#e3e7eb", "#b7bdc6")
    TEAL3 = ("#79cfb8", "#a7e5d4", "#4fbda3")
    AMB3 = ("#f4cd82", "#f9e1b0", "#e6b24b")
    IND = "#9db2f6"
    fig, ax = new_ax(14.2, 5.2, (0, 20.5), (0, 7.6))
    heading(ax, 0.4, 7.0, "전형적인 CNN 아키텍처")
    yc = 3.6

    ax.add_patch(FancyBboxPatch((2.2, 1.15), 9.6, 4.85, boxstyle="round,pad=0.02,rounding_size=0.2",
                 fc="#effaf6", ec="#c6eadf", lw=1.3, zorder=0))
    ax.text(7.0, 6.15, "① 특징 추출  (합성곱 · 풀링)", fontsize=11.5, color=BRAND, fontweight="bold", ha="center")
    ax.add_patch(FancyBboxPatch((12.4, 1.15), 6.4, 4.85, boxstyle="round,pad=0.02,rounding_size=0.2",
                 fc="#eef1fe", ec="#cbd5f8", lw=1.3, zorder=0))
    ax.text(15.6, 6.15, "② 분류  (완전연결)", fontsize=11.5, color=INDIGO, fontweight="bold", ha="center")

    def label(cx, top_y, name, dim, col):
        ax.text(cx, top_y, name, ha="center", fontsize=10, fontweight="bold", color=col, zorder=6)
        ax.text(cx, 1.65, dim, ha="center", fontsize=9.5, fontweight="bold", color="#33404e", zorder=6)

    # (x, fw, fh, dp, shades, name, dim, labelcolor); fw/fh=공간, dp=채널깊이
    prisms = [
        (0.5, 1.7, 1.7, 0.16, GRAY3, "입력", "28×28×1", "#7a8492"),
        (3.0, 1.7, 1.7, 1.0, TEAL3, "Conv+ReLU", "28×28×32", BRAND),
        (5.6, 1.15, 1.15, 1.0, AMB3, "Pool", "14×14×32", AMBER),
        (7.7, 1.15, 1.15, 1.55, TEAL3, "Conv+ReLU", "14×14×64", BRAND),
        (10.0, 0.72, 0.72, 1.55, AMB3, "Pool", "7×7×64", AMBER),
    ]
    right_ends = []
    for (x, fw, fh, dp, sh, name, dim, lc) in prisms:
        rx = _prism(ax, x, yc-fh/2, fw, fh, dp, sh)
        right_ends.append(rx)
        cx = x + fw/2 + dp*0.25
        label(cx, yc+fh/2 + dp*0.42 + 0.4, name, dim, lc)
    for i in range(1, len(prisms)):
        connect(ax, (right_ends[i-1], yc), (prisms[i][0]-0.05, yc))
    connect(ax, (right_ends[-1], yc), (12.5, yc))

    # 분류부: Flatten → FC → FC (세로 막대, 높이=유닛 수)
    _bar(ax, 12.55, yc, 0.34, 2.7, IND); label(12.72, yc+1.55, "Flatten", "3136", INDIGO)
    _bar(ax, 14.35, yc, 0.34, 1.5, IND); label(14.52, yc+1.55, "FC", "128", INDIGO)
    _bar(ax, 16.1, yc, 0.34, 0.6, IND); label(16.27, yc+1.55, "FC", "10", INDIGO)
    connect(ax, (12.89, yc), (14.35, yc))
    connect(ax, (14.69, yc), (16.1, yc))
    connect(ax, (16.44, yc), (17.7, yc))
    ax.text(18.6, yc, "클래스\n확률", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(9.9, 0.65, "공간 크기는 줄고(28→14→7) 채널 깊이는 커집니다(1→32→64).",
            fontsize=10.5, color=SUB, ha="center")
    save(fig, "ch5_cnn_arch_ours.png")


def fig_pool_ours():
    import numpy as np
    from matplotlib.patches import Rectangle
    from _ours import BRAND, CORAL, INDIGO, AMBER
    fig, ax = new_ax(11.5, 4.8, (0, 17), (0, 8))
    heading(ax, 0.4, 7.3, "풀링 — 크기를 줄이고 강한 특징만 남깁니다")
    ax.text(0.75, 6.55, "2×2 영역마다 가장 큰 값 하나만 남겨, 공간 크기를 절반으로 줄입니다.",
            fontsize=11.5, color=SUB, va="center")

    vals = np.array([[1,3,2,4],[2,9,1,5],[6,2,8,3],[1,4,2,7]])
    tints = [BRAND, AMBER, INDIGO, CORAL]

    # 입력 카드
    card(ax, 0.7, 1.3, 4.5, 4.2, title="입력 특징 맵", color=BRAND)
    u, ox, oy = 0.78, 1.15, 1.85
    for bi in range(2):
        for bj in range(2):
            col = tints[bi*2+bj]
            for i in range(2):
                for j in range(2):
                    r, c = bi*2+i, bj*2+j
                    ax.add_patch(Rectangle((ox+c*u, oy+(3-r)*u), u, u, fc=_soft(col), ec="white", lw=2.5, zorder=3))
                    ax.text(ox+c*u+u/2, oy+(3-r)*u+u/2, f"{vals[r,c]}", ha="center", va="center",
                            fontsize=13, fontweight="bold", color="#28303a", zorder=4)
            ax.add_patch(Rectangle((ox+bj*2*u, oy+(2-bi*2)*u), 2*u, 2*u, fc="none", ec=col, lw=2.4, zorder=5))

    # 노드
    node(ax, 6.4, 3.4, "▷", color=BRAND, r=0.42, fs=15)
    ax.text(6.4, 2.62, "2×2 최대", fontsize=9.5, color=BRAND, fontweight="bold", ha="center")
    connect(ax, (5.3, 3.4), (5.95, 3.4))

    # 출력 카드
    card(ax, 7.4, 2.0, 3.0, 2.9, title="출력", color=CORAL)
    out = np.array([[vals[0:2,0:2].max(), vals[0:2,2:4].max()],
                    [vals[2:4,0:2].max(), vals[2:4,2:4].max()]])
    u2, ox2, oy2 = 1.0, 7.9, 2.55
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((ox2+j*u2, oy2+(1-i)*u2), u2, u2, fc=_soft(tints[i*2+j]), ec="white", lw=2.5, zorder=3))
            ax.text(ox2+j*u2+u2/2, oy2+(1-i)*u2+u2/2, f"{out[i,j]}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color="#28303a", zorder=4)
    connect(ax, (6.85, 3.4), (7.5, 3.4))

    ax.text(13.2, 5.0, "요점", fontsize=11, color=INK, fontweight="bold", ha="center")
    chip(ax, 13.2, 4.25, "크기 절반 → 계산량 감소", color=BRAND, fs=10.5)
    chip(ax, 13.2, 3.45, "위치가 조금 변해도 값 유지", color=CORAL, fs=10.5)
    chip(ax, 13.2, 2.65, "학습 파라미터가 없습니다", color=INDIGO, fs=10.5)
    save(fig, "ch5_pool_ours.png")


def _soft(hexc):
    """진한 색을 카드 배경용 연한 톤으로."""
    from _ours import _tint_of
    return _tint_of(hexc)


if __name__ == "__main__":
    fig_conv_ours()
    fig_cnn_arch_ours()
    fig_pool_ours()
    print("done ours demo")
