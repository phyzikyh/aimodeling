# -*- coding: utf-8 -*-
"""아키텍처 다이어그램용 공통 헬퍼(박스·화살표·연산 노드)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

# 색
BOX_BLUE = "#dbeafe"; EDGE_BLUE = "#2c7be5"
BOX_GREEN = "#d3f5e8"; EDGE_GREEN = "#1AB18B"
BOX_ORANGE = "#ffedd5"; EDGE_ORANGE = "#e8912d"
BOX_RED = "#fde2dd"; EDGE_RED = "#e5533c"
BOX_GRAY = "#eef1f5"; EDGE_GRAY = "#64748b"
BOX_PURPLE = "#ede4fb"; EDGE_PURPLE = "#8957e5"


def box(ax, x, y, w, h, text, fc=BOX_BLUE, ec=EDGE_BLUE, fs=11, bold=True, round=0.09):
    # 부드러운 그림자
    ax.add_patch(FancyBboxPatch((x+0.05, y-0.07), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={round}",
                 fc="#dde1e7", ec="none", zorder=1.5))
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={round}",
                       fc=fc, ec=ec, lw=1.6, zorder=2)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)
    return (x+w/2, y+h/2)


def opnode(ax, x, y, sym, r=0.16, fc="#fff7cc", ec="#b7791f", fs=13):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.6, zorder=4))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, zorder=5)
    return (x, y)


def arrow(ax, p0, p1, color=EDGE_GRAY, lw=1.8, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                        lw=lw, color=color, zorder=1, ls=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def setup(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")
