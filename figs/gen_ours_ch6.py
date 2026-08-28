# -*- coding: utf-8 -*-
"""6장 우리-스타일 그림: 객체탐지 모델 분류체계(taxonomy)."""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _ours import new_ax, save, heading, card, BRAND, CORAL, INDIGO, INK, SUB, TINT


def leaf(ax, cx, cy, name, desc, color, w=4.6, h=0.94):
    ax.add_patch(FancyBboxPatch((cx-w/2+0.05, cy-h/2-0.06), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc="#dfe3e8", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=color, lw=1.7, zorder=4))
    # 왼쪽 색 태그 바
    ax.add_patch(FancyBboxPatch((cx-w/2+0.14, cy-h/2+0.13), 0.16, h-0.26,
                 boxstyle="round,pad=0.0,rounding_size=0.05", fc=color, ec="none", zorder=5))
    ax.text(cx-w/2+0.5, cy+0.17, name, ha="left", va="center", fontsize=12.5,
            fontweight="bold", color=INK, zorder=5)
    ax.text(cx-w/2+0.5, cy-0.19, desc, ha="left", va="center", fontsize=9.5,
            color=SUB, zorder=5)


def ar(ax, p0, p1, color, lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=lw,
                 color=color, zorder=2, shrinkA=2, shrinkB=2,
                 connectionstyle=f"arc3,rad={rad}"))


def fig_detection_taxonomy():
    fig, ax = new_ax(10.4, 8.0, (0, 15), (0, 11.6))
    heading(ax, 0.4, 11.0, "객체탐지 모델의 분류체계")
    ax.text(0.75, 10.35, "후보 영역을 먼저 뽑느냐(2단계)와 한 번에 예측하느냐(1단계)로 크게 갈립니다.",
            fontsize=11, color=SUB, va="center")

    # 루트 노드
    rx, ry, rw, rh = 6.15, 9.0, 2.7, 0.9
    ax.add_patch(FancyBboxPatch((rx+0.06, ry-0.09), rw, rh,
                 boxstyle="round,pad=0.02,rounding_size=0.2", fc="#dfe3e8", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh,
                 boxstyle="round,pad=0.02,rounding_size=0.2", fc=INK, ec="none", zorder=4))
    ax.text(rx+rw/2, ry+rh/2, "객체탐지", ha="center", va="center", fontsize=14,
            fontweight="bold", color="white", zorder=5)

    # 두 분기 카드
    lx, w = 0.7, 6.4
    card(ax, lx, 1.0, w, 7.1, "2단계 검출기 · 정확·상대적으로 느림", color=BRAND)
    card(ax, 7.9, 1.0, w, 7.1, "1단계 검출기 · 빠름·실시간", color=CORAL)

    # 루트→분기 연결선
    ar(ax, (rx+0.4, ry), (lx+w/2, 8.1+0.15), BRAND, rad=0.12)
    ar(ax, (rx+rw-0.4, ry), (7.9+w/2, 8.1+0.15), CORAL, rad=-0.12)

    two = [("R-CNN", "선택적 탐색 + CNN 특징"),
           ("Fast R-CNN", "RoI 풀링으로 공유 연산"),
           ("Faster R-CNN", "RPN으로 후보까지 학습"),
           ("Mask R-CNN", "분할 마스크 분기 추가"),
           ("Cascade R-CNN", "IoU 임계 단계적 상향")]
    one = [("YOLO", "격자 단위 단일 예측"),
           ("SSD", "다중 스케일 특징맵"),
           ("RetinaNet", "Focal Loss 불균형 완화"),
           ("EfficientDet", "BiFPN·복합 스케일링"),
           ("FCOS", "앵커 없는 점 기반 예측"),
           ("CenterNet", "중심점 히트맵 회귀"),
           ("CornerNet", "코너 쌍으로 상자 구성")]

    cxL, cxR = lx+w/2, 7.9+w/2
    yL = [7.0 - i*1.22 for i in range(5)]
    for (n, d), y in zip(two, yL):
        leaf(ax, cxL, y, n, d, BRAND)
    yR = [7.1 - i*0.92 for i in range(7)]
    for (n, d), y in zip(one, yR):
        leaf(ax, cxR, y, n, d, CORAL, h=0.80)

    save(fig, "ch6_taxonomy_ours.png")


if __name__ == "__main__":
    fig_detection_taxonomy()
    print("done ours ch6")
