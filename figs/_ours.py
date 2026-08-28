# -*- coding: utf-8 -*-
"""'우리 스타일' 시각 언어 툴킷 — 참고 슬라이드와 구별되는 에디토리얼 카드 시스템.
핵심 장치: 라운드 카드 + 상단 색 탭, 가는 연결선 + 번호 노드, 키워드 태그 칩/형광 마커.
문장은 명사구·'~합니다' 체, 글자에 색칠하지 않고 별도 강조 장치를 사용.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
import matplotlib as mpl
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 140
mpl.rcParams["savefig.bbox"] = "tight"
import os
OUT = os.path.dirname(__file__)

# 우리 팔레트 (브랜드 청록-그린 축)
BRAND = "#1AB18B"; CORAL = "#FF6B5C"; INDIGO = "#4C6EF5"; AMBER = "#F1A208"; ROSE = "#EC4B8A"
INK = "#1f2937"; SUB = "#5b6675"; LINE = "#c7ccd3"
TINT = {"brand": "#e7f7f1", "coral": "#ffe9e5", "indigo": "#e9edfd",
        "amber": "#fdf1d8", "rose": "#fce4ef", "gray": "#eef1f4"}


def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p); plt.close(fig)
    print("saved", name, os.path.getsize(p), "bytes")


def new_ax(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def heading(ax, x, y, text, fs=17, accent=BRAND):
    """왼쪽 브랜드 액센트 바 + 굵은 명사구 제목(색칠 없음)."""
    ax.add_patch(Rectangle((x, y-0.28), 0.13, 0.62, fc=accent, ec="none", zorder=5))
    ax.text(x+0.34, y, text, fontsize=fs, fontweight="bold", color=INK, va="center", zorder=5)


def card(ax, x, y, w, h, title=None, color=BRAND, tint="gray"):
    """부드러운 그림자 + 흰 라운드 카드 + 상단 색 탭(라벨)."""
    ax.add_patch(FancyBboxPatch((x+0.06, y-0.09), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", fc="#dfe3e8", ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", fc="white", ec="#e4e8ec", lw=1.2, zorder=2))
    if title:
        tw = 0.42*len(title) + 0.6
        ax.add_patch(FancyBboxPatch((x+0.25, y+h-0.28), tw, 0.5,
                     boxstyle="round,pad=0.02,rounding_size=0.22", fc=color, ec="none", zorder=4))
        ax.text(x+0.25+tw/2, y+h-0.03, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white", zorder=5)


def numgrid(ax, x, y, vals, u, tint="#eef1f4", ec="#ffffff", fs=12, hi=None, hicolor=BRAND):
    """숫자 격자(연한 단색 톤). hi=[(r,c),...] 강조 셀은 브랜드 톤 테두리."""
    vals = np.array(vals)
    R, C = vals.shape
    for i in range(R):
        for j in range(C):
            ax.add_patch(Rectangle((x+j*u, y+(R-1-i)*u), u, u, fc=tint, ec=ec, lw=2, zorder=3))
            ax.text(x+j*u+u/2, y+(R-1-i)*u+u/2, f"{vals[i,j]:g}", ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=INK, zorder=4)
    if hi:
        for (i, j) in hi:
            ax.add_patch(Rectangle((x+j*u, y+(R-1-i)*u), u, u, fc="none", ec=hicolor, lw=2.6, zorder=5))
    return (x, y, C*u, R*u)


def node(ax, cx, cy, sym, color=BRAND, r=0.34, fs=15):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=color, lw=2.4, zorder=6))
    ax.text(cx, cy, sym, ha="center", va="center", fontsize=fs, fontweight="bold", color=color, zorder=7)


def connect(ax, p0, p1, color=LINE, lw=2.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                 lw=lw, color=color, zorder=3, shrinkA=2, shrinkB=2))


def chip(ax, cx, cy, text, color=BRAND, fs=11):
    """키워드 태그 칩(연한 톤 배경 + 색 테두리 + 짙은 글자). 문장 속 색칠 대체."""
    w = 0.235*len(text) + 0.85
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-0.26), w, 0.52,
                 boxstyle="round,pad=0.02,rounding_size=0.26",
                 fc=_tint_of(color), ec=color, lw=1.8, zorder=5))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=color, zorder=6)


def marker(ax, x, y, text, color=BRAND, fs=12):
    """형광 마커 강조(단어 뒤 반투명 톤 배경). 인라인 색칠 대체."""
    t = ax.text(x, y, text, fontsize=fs, fontweight="bold", color=INK, va="center", zorder=6)
    return t


def _tint_of(color):
    m = {BRAND: TINT["brand"], CORAL: TINT["coral"], INDIGO: TINT["indigo"],
         AMBER: TINT["amber"], ROSE: TINT["rose"]}
    return m.get(color, "#eef1f4")


# --- 3D 볼륨(특징맵/텐서)·막대·구역 밴드 (표준) ---
from matplotlib.patches import Polygon as _Poly

SHADE = {  # (front, top, right)
    "gray":  ("#cfd4db", "#e3e7eb", "#b7bdc6"),
    "brand": ("#79cfb8", "#a7e5d4", "#4fbda3"),
    "coral": ("#f7a79a", "#fbc3ba", "#ef8474"),
    "amber": ("#f4cd82", "#f9e1b0", "#e6b24b"),
    "indigo":("#9db2f6", "#bfccfa", "#6f8bf0"),
    "rose":  ("#f2a6c9", "#f7c4dc", "#ec7fb0"),
}


def volume(ax, x, y, fw, fh, dp, kind="brand"):
    """등축 3D 볼륨. 앞면 fw×fh, 깊이 dp(iso). y=앞면 아래. 오른쪽 끝 x 반환."""
    fr, tp, rt = SHADE.get(kind, SHADE["brand"])
    ox, oy = dp*0.5, dp*0.42
    top = [(x, y+fh), (x+fw, y+fh), (x+fw+ox, y+fh+oy), (x+ox, y+fh+oy)]
    right = [(x+fw, y), (x+fw+ox, y+oy), (x+fw+ox, y+fh+oy), (x+fw, y+fh)]
    front = [(x, y), (x+fw, y), (x+fw, y+fh), (x, y+fh)]
    for pts, c in [(top, tp), (right, rt), (front, fr)]:
        ax.add_patch(_Poly(pts, closed=True, fc=c, ec="#4d5766", lw=1.3, zorder=3))
    return x+fw+ox


def bar(ax, x, yc, w, h, kind="indigo"):
    ax.add_patch(FancyBboxPatch((x, yc-h/2), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                 fc=SHADE.get(kind, SHADE["indigo"])[0], ec="#4d5766", lw=1.3, zorder=3))


def zone(ax, x, y, w, h, label, color=BRAND, fc="#effaf6", ec="#c6eadf"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                 fc=fc, ec=ec, lw=1.3, zorder=0))
    ax.text(x+w/2, y+h+0.18, label, fontsize=11.5, color=color, fontweight="bold", ha="center", va="bottom")
