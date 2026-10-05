# -*- coding: utf-8 -*-
"""논문 스타일 그림 공통 모듈 (6주차 이후).

원칙: 흰 배경, 얇은 선, 절제된 색, 텐서 shape 직접 표기, 패널 라벨 (a)(b)(c).
캡션은 본문(.qmd)에서 달므로 그림 안에는 제목을 넣지 않는다.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle, FancyArrowPatch

from matplotlib import font_manager as _fm
_avail = {f.name for f in _fm.fontManager.ttflist}
_KO = next((f for f in ("Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic") if f in _avail), "DejaVu Sans")

mpl.rcParams.update({
    "font.family": [_KO, "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "font.size": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
})

OUT = os.path.dirname(os.path.abspath(__file__))

# 슬라이드용: 환경변수 PAPER_SLIDE=1.4 처럼 주면 모든 글자를 키우고 파일명에 _s 를 붙인다.
SLIDE = float(os.environ.get("PAPER_SLIDE", "1"))
if SLIDE != 1:
    import matplotlib.text as _mt
    _orig_sf = _mt.Text.set_fontsize

    def _scaled_sf(self, fontsize):
        if isinstance(fontsize, (int, float)):
            fontsize = fontsize * SLIDE
        _orig_sf(self, fontsize)
    _mt.Text.set_fontsize = _scaled_sf

# 인쇄·색각 이상에 안전한 절제된 팔레트
INK = "#1f2328"
SUB = "#57606a"
LINE = "#8c959f"
BLUE = "#2f6db2"
ORANGE = "#d9822b"
GREEN = "#3f9a6b"
RED = "#c4423f"
PURPLE = "#7a5ba6"
GRAY = "#9aa3ad"
# 연한 채움색
F_BLUE = "#dbe8f6"
F_ORANGE = "#fbe6cf"
F_GREEN = "#dcf0e4"
F_RED = "#f6dad9"
F_PURPLE = "#e8e0f2"
F_GRAY = "#eceff2"


def save(fig, name):
    """PNG(300dpi)와 PDF(벡터)를 함께 저장한다."""
    name = name + ("_s" if SLIDE != 1 else "")
    base = os.path.join(OUT, name)
    fig.savefig(base + ".png")
    if SLIDE == 1:
        fig.savefig(base + ".pdf")
    plt.close(fig)
    print("saved", name, os.path.getsize(base + ".png") // 1024, "KB")


def canvas(w, h, xlim=None, ylim=None):
    """좌표를 직접 쓰는 도식용 캔버스 (축 숨김, 가로세로 1:1)."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*(xlim or (0, w)))
    ax.set_ylim(*(ylim or (0, h)))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def panel_label(ax, x, y, text, fs=10):
    ax.text(x, y, text, fontsize=fs, fontweight="bold", color=INK, ha="left", va="center")


def box(ax, x, y, w, h, text="", fc=F_GRAY, ec=LINE, lw=0.8, fs=8, color=INK,
        r=0.06, bold=False, ls="-", z=2, va="center"):
    """둥근 모서리 상자. (x, y)는 좌하단."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va=va, fontsize=fs,
                color=color, fontweight="bold" if bold else "normal", zorder=z + 1,
                linespacing=1.25)
    return p


def arrow(ax, p0, p1, color=SUB, lw=1.0, ls="-", style="-|>", ms=7, rad=0.0, z=4):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                        ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def label(ax, x, y, text, fs=8, color=INK, ha="center", va="center", bold=False, **kw):
    return ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va,
                   fontweight="bold" if bold else "normal", linespacing=1.25, **kw)


def slab(ax, x, yc, w, h, fc, ec=INK, lw=0.7, depth=0.0, z=3):
    """특징 맵 블록: 중심 y=yc, 폭 w(채널 수에 비례), 높이 h(공간 크기에 비례)."""
    ax.add_patch(Rectangle((x, yc - h / 2), w, h, fc=fc, ec=ec, lw=lw, zorder=z))
