# -*- coding: utf-8 -*-
"""5장 아키텍처: CNN 전체 파이프라인, ResNet 잔차 블록."""
import matplotlib.pyplot as plt
from _style import save
from _arch import (box, opnode, arrow, setup,
                   BOX_BLUE, EDGE_BLUE, BOX_GREEN, EDGE_GREEN,
                   BOX_ORANGE, EDGE_ORANGE, BOX_PURPLE, EDGE_PURPLE,
                   BOX_GRAY, EDGE_GRAY, BOX_RED, EDGE_RED)


def fig_cnn_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 3.4), constrained_layout=True)
    setup(ax, (0, 14.2), (0, 3.6))
    blocks = [
        (0.2, 1.3, 0.9, "입력\n28×28×1", BOX_GRAY, EDGE_GRAY),
        (1.7, 1.3, 1.5, "Conv+ReLU\n28×28×32", BOX_BLUE, EDGE_BLUE),
        (3.6, 1.5, 1.1, "Pool\n14×14×32", BOX_ORANGE, EDGE_ORANGE),
        (5.1, 1.55, 1.5, "Conv+ReLU\n14×14×64", BOX_BLUE, EDGE_BLUE),
        (7.0, 1.7, 0.85, "Pool\n7×7×64", BOX_ORANGE, EDGE_ORANGE),
        (8.3, 1.75, 0.9, "Flatten\n3136", BOX_GRAY, EDGE_GRAY),
        (9.9, 1.75, 0.9, "FC\n128", BOX_GREEN, EDGE_GREEN),
        (11.4, 1.8, 0.75, "FC\n10", BOX_GREEN, EDGE_GREEN),
    ]
    W = 1.35
    prev = None
    for (x, yc, h, t, fc, ec) in blocks:
        box(ax, x, 2.0-h/2, W, h, t, fc=fc, ec=ec, fs=8.5)
        if prev is not None:
            arrow(ax, (prev, 2.0), (x, 2.0), color=EDGE_GRAY, lw=1.6)
        prev = x + W
    arrow(ax, (prev, 2.0), (prev+0.7, 2.0), color="#111", lw=1.8)
    ax.text(prev+1.35, 2.0, "클래스\n확률", fontsize=9, ha="center", va="center", fontweight="bold")
    ax.text(7.1, 0.35, "합성곱·풀링으로 특징 추출 → 완전연결층으로 분류  (공간 크기↓, 채널↑)",
            fontsize=9.5, ha="center", color="#475569")
    ax.set_title("전형적인 CNN 아키텍처 (예: 이미지 분류)", fontsize=13, fontweight="bold")
    save(fig, "ch5_cnn_arch.png")


def fig_resnet_block():
    fig, ax = plt.subplots(figsize=(6.2, 6.4), constrained_layout=True)
    setup(ax, (0, 6.5), (0, 9.2))
    cx = 2.6; W = 2.6
    ax.text(cx, 0.4, "입력  $x$", fontsize=12, ha="center", fontweight="bold")
    ys = [
        (1.1, 0.62, "Conv 3×3", BOX_BLUE, EDGE_BLUE),
        (1.95, 0.5, "BatchNorm", "#fff7cc", "#b7791f"),
        (2.68, 0.5, "ReLU", BOX_GREEN, EDGE_GREEN),
        (3.5, 0.62, "Conv 3×3", BOX_BLUE, EDGE_BLUE),
        (4.35, 0.5, "BatchNorm", "#fff7cc", "#b7791f"),
    ]
    for (y, h, t, fc, ec) in ys:
        box(ax, cx-W/2, y, W, h, t, fc=fc, ec=ec, fs=10)
    arrow(ax, (cx, 0.6), (cx, 1.1), color=EDGE_GRAY, lw=1.6)
    for (y0, y1) in [(1.72, 1.95), (2.45, 2.68), (3.18, 3.5), (4.12, 4.35)]:
        arrow(ax, (cx, y0), (cx, y1), color=EDGE_GRAY, lw=1.6)
    add = opnode(ax, cx, 5.5, r"$+$", r=0.22)
    arrow(ax, (cx, 4.85), (cx, 5.28), color=EDGE_GRAY, lw=1.6)
    box(ax, cx-W/2, 6.1, W, 0.5, "ReLU", fc=BOX_GREEN, ec=EDGE_GREEN, fs=10)
    arrow(ax, (cx, 5.72), (cx, 6.1), color=EDGE_GRAY, lw=1.6)
    arrow(ax, (cx, 6.6), (cx, 7.2), color="#111", lw=1.8)
    ax.text(cx, 7.5, r"출력  $\mathcal{F}(x)+x$", fontsize=12, ha="center", fontweight="bold")
    # skip connection (오른쪽으로 우회)
    arrow(ax, (cx+W/2, 0.55), (5.7, 0.55), color=EDGE_RED, lw=2.0, style="-")
    arrow(ax, (5.7, 0.55), (5.7, 5.5), color=EDGE_RED, lw=2.0, style="-")
    arrow(ax, (5.7, 5.5), (cx+0.22, 5.5), color=EDGE_RED, lw=2.0)
    ax.text(5.95, 3.0, "항등 사상\n(skip)", fontsize=9.5, color=EDGE_RED,
            rotation=90, ha="center", va="center")
    ax.text(cx, 4.95, r"$\mathcal{F}(x)$", fontsize=11, ha="center", color=EDGE_BLUE)
    ax.set_title("ResNet 잔차 블록", fontsize=13, fontweight="bold")
    save(fig, "ch5_resnet_block.png")


if __name__ == "__main__":
    fig_cnn_pipeline()
    fig_resnet_block()
    print("done arch5")
