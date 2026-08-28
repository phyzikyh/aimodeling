# -*- coding: utf-8 -*-
"""6장 아키텍처: U-Net (인코더-디코더 + 스킵 연결)."""
import matplotlib.pyplot as plt
from _style import save
from _arch import (box, arrow, setup,
                   BOX_BLUE, EDGE_BLUE, BOX_GREEN, EDGE_GREEN,
                   BOX_ORANGE, EDGE_ORANGE, BOX_RED, EDGE_RED,
                   BOX_GRAY, EDGE_GRAY)


def fig_unet():
    fig, ax = plt.subplots(figsize=(9.6, 6.2), constrained_layout=True)
    setup(ax, (0, 12), (0, 8.2))
    encx, decx = 2.3, 9.5
    levels = [(6.7, "H×W", "64"), (5.3, "H/2×W/2", "128"), (3.9, "H/4×W/4", "256")]
    W, H = 2.1, 0.85
    enc_pts, dec_pts = [], []
    for (y, sp, ch) in levels:
        box(ax, encx-W/2, y-H/2, W, H, f"인코더\n{sp} · {ch}", fc=BOX_BLUE, ec=EDGE_BLUE, fs=8.5)
        box(ax, decx-W/2, y-H/2, W, H, f"디코더\n{sp} · {ch}", fc=BOX_GREEN, ec=EDGE_GREEN, fs=8.5)
        enc_pts.append(y); dec_pts.append(y)
        # 스킵 연결
        arrow(ax, (encx+W/2, y), (decx-W/2, y), color=EDGE_RED, lw=1.6, style="-|>", ls=(0,(4,3)))
    ax.text(6.0, 7.05, "스킵 연결 (같은 해상도 특징 전달)", fontsize=9.5,
            color=EDGE_RED, ha="center")
    # 병목
    box(ax, 5.5-1.2, 2.15, 2.4, 0.8, "병목\nH/8×W/8 · 512", fc=BOX_ORANGE, ec=EDGE_ORANGE, fs=9)
    # 다운샘플 경로 (인코더 하강)
    for i in range(len(levels)-1):
        arrow(ax, (encx, levels[i][0]-H/2), (encx, levels[i+1][0]+H/2), color=EDGE_BLUE, lw=1.8)
    arrow(ax, (encx, levels[-1][0]-H/2), (5.0, 2.75), color=EDGE_BLUE, lw=1.8, rad=-0.15)
    ax.text(3.1, 3.05, "다운샘플 ↓", fontsize=8.5, color=EDGE_BLUE, rotation=90, va="center")
    # 업샘플 경로 (디코더 상승)
    arrow(ax, (6.7, 2.75), (decx, levels[-1][0]-H/2), color=EDGE_GREEN, lw=1.8, rad=-0.15)
    for i in range(len(levels)-1, 0, -1):
        arrow(ax, (decx, levels[i][0]+H/2), (decx, levels[i-1][0]-H/2), color=EDGE_GREEN, lw=1.8)
    ax.text(10.35, 3.05, "업샘플 ↑", fontsize=8.5, color=EDGE_GREEN, rotation=90, va="center")
    # 입출력
    box(ax, encx-W/2, 7.55, W, 0.5, "입력 이미지", fc=BOX_GRAY, ec=EDGE_GRAY, fs=8.5)
    arrow(ax, (encx, 7.55), (encx, 6.7+H/2), color=EDGE_GRAY, lw=1.5)
    box(ax, decx-W/2, 7.55, W, 0.5, "분할 결과", fc=BOX_GRAY, ec=EDGE_GRAY, fs=8.5)
    arrow(ax, (decx, 6.7+H/2), (decx, 7.55), color="#111", lw=1.6)
    ax.set_title("U-Net 아키텍처 (인코더-디코더 + 스킵 연결)", fontsize=13, fontweight="bold")
    save(fig, "ch6_unet_arch.png")


if __name__ == "__main__":
    fig_unet()
    print("done arch6")
