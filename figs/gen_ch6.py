# -*- coding: utf-8 -*-
"""6장(객체탐지) 개념 그림. matplotlib 사각형으로 박스 시각화."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from _style import save, C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_GRAY

rng = np.random.default_rng(6)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua, (ix0, iy0, ix1, iy1)


# 6.1 IoU 시각화
def fig_iou():
    cases = [((1, 1, 4.2, 4.2), (2.4, 2.3, 5.4, 5.2), "(a) 낮은 IoU"),
             ((1, 1, 4.2, 4.2), (1.4, 1.2, 4.5, 4.3), "(b) 높은 IoU")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
    for ax, (gt, pr, title) in zip(axes, cases):
        val, inter = iou(gt, pr)
        ax.add_patch(Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                     fill=False, ec=C_GREEN, lw=2.4, label="정답(GT)"))
        ax.add_patch(Rectangle((pr[0], pr[1]), pr[2]-pr[0], pr[3]-pr[1],
                     fill=False, ec=C_RED, lw=2.4, ls="--", label="예측"))
        ax.add_patch(Rectangle((inter[0], inter[1]), inter[2]-inter[0], inter[3]-inter[1],
                     facecolor=C_BLUE, alpha=0.3, ec="none", label="교집합"))
        ax.set_title(f"{title}   (IoU = {val:.2f})")
        ax.set_xlim(0, 6.5); ax.set_ylim(0, 6.5); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, "ch6_iou.png")


# 6.2 NMS 전/후
def fig_nms():
    obj = (2.6, 2.6, 1.5)   # cx, cy, r
    boxes = [(1.4, 1.5, 3.9, 3.8, 0.92),
             (1.2, 1.2, 3.7, 3.6, 0.75),
             (1.6, 1.7, 4.1, 4.0, 0.68),
             (1.0, 1.4, 3.6, 3.9, 0.60)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
    titles = ["(a) NMS 이전: 중복 박스 다수", "(b) NMS 이후: 최고 확신도만"]
    for k, ax in enumerate(axes):
        ax.add_patch(Circle((obj[0], obj[1]), obj[2], color=C_GRAY, alpha=0.35))
        show = boxes if k == 0 else [max(boxes, key=lambda b: b[4])]
        for (x0, y0, x1, y1, c) in show:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                         ec=C_RED, lw=2.0))
            ax.text(x0, y1+0.08, f"{c:.2f}", color=C_RED, fontsize=8)
        ax.set_title(titles[k], fontsize=11)
        ax.set_xlim(0, 5.5); ax.set_ylim(0, 5.5); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch6_nms.png")


# 6.3 앵커 박스
def fig_anchors():
    cx, cy = 3.0, 3.0
    specs = [(1.6, 1.6, C_BLUE), (2.4, 1.2, C_RED), (1.2, 2.4, C_GREEN),
             (3.2, 3.2, C_ORANGE), (4.2, 2.1, C_PURPLE)]
    fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
    for (w, h, col) in specs:
        ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h, fill=False, ec=col, lw=2.0))
    ax.plot(cx, cy, "k+", ms=12, mew=2)
    ax.set_title("한 위치의 앵커 박스 (여러 크기·비율)")
    ax.set_xlim(0, 6); ax.set_ylim(0, 6); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ch6_anchors.png")


if __name__ == "__main__":
    fig_iou()
    fig_nms()
    fig_anchors()
    print("done ch6")
