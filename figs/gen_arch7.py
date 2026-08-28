# -*- coding: utf-8 -*-
"""7장 아키텍처: 오토인코더/VAE(모래시계), GAN(생성기/판별기)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from _style import save
from _arch import (box, opnode, arrow, setup,
                   BOX_BLUE, EDGE_BLUE, BOX_GREEN, EDGE_GREEN,
                   BOX_ORANGE, EDGE_ORANGE, BOX_PURPLE, EDGE_PURPLE,
                   BOX_RED, EDGE_RED, BOX_GRAY, EDGE_GRAY)


def trap(ax, x, w, y, h_left, h_right, fc, ec, text="", fs=10):
    cy = y
    pts = [(x, cy-h_left/2), (x, cy+h_left/2),
           (x+w, cy+h_right/2), (x+w, cy-h_right/2)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.8, zorder=2))
    if text:
        ax.text(x+w/2, cy, text, ha="center", va="center", fontsize=fs,
                fontweight="bold", zorder=3)


def fig_ae_vae():
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.6), constrained_layout=True)
    # (a) 오토인코더
    ax = axes[0]; setup(ax, (0, 12), (0, 4))
    ax.text(0.4, 2.0, "입력\n$x$", fontsize=10, ha="center", va="center", fontweight="bold")
    arrow(ax, (0.9, 2.0), (1.5, 2.0), color=EDGE_GRAY, lw=1.6)
    trap(ax, 1.6, 2.6, 2.0, 2.8, 1.0, BOX_BLUE, EDGE_BLUE, "인코더", 10)
    box(ax, 4.5, 1.6, 1.0, 0.8, "$z$", fc=BOX_ORANGE, ec=EDGE_ORANGE, fs=12)
    ax.text(5.0, 1.2, "잠재", fontsize=8.5, ha="center", color=EDGE_ORANGE)
    arrow(ax, (4.2, 2.0), (4.5, 2.0), color=EDGE_GRAY, lw=1.4)
    trap(ax, 5.9, 2.6, 2.0, 1.0, 2.8, BOX_GREEN, EDGE_GREEN, "디코더", 10)
    arrow(ax, (5.5, 2.0), (5.9, 2.0), color=EDGE_GRAY, lw=1.4)
    arrow(ax, (8.6, 2.0), (9.3, 2.0), color="#111", lw=1.6)
    ax.text(9.9, 2.0, "복원\n$\\hat{x}$", fontsize=10, ha="center", va="center", fontweight="bold")
    ax.text(10.9, 2.0, "", fontsize=1)
    ax.set_title("(a) 오토인코더", fontsize=12, fontweight="bold", loc="left")

    # (b) VAE
    ax = axes[1]; setup(ax, (0, 12), (0, 4))
    ax.text(0.4, 2.0, "입력\n$x$", fontsize=10, ha="center", va="center", fontweight="bold")
    arrow(ax, (0.9, 2.0), (1.5, 2.0), color=EDGE_GRAY, lw=1.6)
    trap(ax, 1.6, 2.4, 2.0, 2.8, 1.2, BOX_BLUE, EDGE_BLUE, "인코더", 10)
    box(ax, 4.3, 2.5, 1.0, 0.55, r"$\mu$", fc=BOX_PURPLE, ec=EDGE_PURPLE, fs=11)
    box(ax, 4.3, 1.55, 1.0, 0.55, r"$\sigma$", fc=BOX_PURPLE, ec=EDGE_PURPLE, fs=11)
    arrow(ax, (4.05, 2.3), (4.3, 2.75), color=EDGE_GRAY, lw=1.3)
    arrow(ax, (4.05, 1.75), (4.3, 1.8), color=EDGE_GRAY, lw=1.3)
    s = opnode(ax, 6.0, 2.0, r"$z$", r=0.32, fc=BOX_ORANGE, ec=EDGE_ORANGE, fs=11)
    arrow(ax, (5.3, 2.7), (5.75, 2.2), color=EDGE_PURPLE, lw=1.3)
    arrow(ax, (5.3, 1.8), (5.75, 1.9), color=EDGE_PURPLE, lw=1.3)
    ax.text(6.0, 1.25, r"$z=\mu+\sigma\odot\varepsilon$", fontsize=9.5, ha="center", color=EDGE_ORANGE)
    ax.text(6.0, 3.2, r"$\varepsilon\sim\mathcal{N}(0,I)$", fontsize=9, ha="center", color="#475569")
    arrow(ax, (6.0, 2.9), (6.0, 2.32), color="#475569", lw=1.2)
    trap(ax, 6.7, 2.4, 2.0, 1.2, 2.8, BOX_GREEN, EDGE_GREEN, "디코더", 10)
    arrow(ax, (6.32, 2.0), (6.7, 2.0), color=EDGE_GRAY, lw=1.4)
    arrow(ax, (9.1, 2.0), (9.8, 2.0), color="#111", lw=1.6)
    ax.text(10.4, 2.0, "복원\n$\\hat{x}$", fontsize=10, ha="center", va="center", fontweight="bold")
    ax.set_title("(b) 변이형 오토인코더(VAE)", fontsize=12, fontweight="bold", loc="left")
    save(fig, "ch7_ae_vae_arch.png")


def fig_gan():
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    setup(ax, (0, 12.5), (0, 5))
    # 생성기
    ax.text(0.5, 3.4, r"잡음 $z$" + "\n$\\sim\\mathcal{N}(0,I)$", fontsize=9.5,
            ha="center", va="center", fontweight="bold")
    arrow(ax, (1.2, 3.4), (1.8, 3.4), color=EDGE_GRAY, lw=1.5)
    trap(ax, 1.9, 2.3, 3.4, 1.0, 2.6, BOX_BLUE, EDGE_BLUE, "생성기 $G$", 10)
    arrow(ax, (4.2, 3.4), (4.9, 3.4), color=EDGE_GRAY, lw=1.5)
    box(ax, 4.95, 3.0, 1.6, 0.8, "가짜\n$G(z)$", fc=BOX_RED, ec=EDGE_RED, fs=9)
    box(ax, 4.95, 1.1, 1.6, 0.8, "진짜 $x$", fc=BOX_GREEN, ec=EDGE_GREEN, fs=9.5)
    # 판별기
    trap(ax, 7.2, 2.3, 2.4, 2.8, 1.0, BOX_PURPLE, EDGE_PURPLE, "판별기 $D$", 10)
    arrow(ax, (6.55, 3.4), (7.2, 2.75), color=EDGE_RED, lw=1.5)
    arrow(ax, (6.55, 1.5), (7.2, 2.2), color=EDGE_GREEN, lw=1.5)
    arrow(ax, (9.5, 2.4), (10.2, 2.4), color="#111", lw=1.6)
    ax.text(11.2, 2.4, "진짜/가짜\n판정 (0~1)", fontsize=9.5, ha="center", va="center", fontweight="bold")
    ax.text(6.2, 0.35, "생성기는 판별기를 속이도록, 판별기는 둘을 구별하도록 경쟁 학습(미니맥스)",
            fontsize=9.5, ha="center", color="#475569")
    ax.set_title("GAN 아키텍처 (생성기 vs 판별기)", fontsize=13, fontweight="bold")
    save(fig, "ch7_gan_arch.png")


if __name__ == "__main__":
    fig_ae_vae()
    fig_gan()
    print("done arch7")
