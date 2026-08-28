# -*- coding: utf-8 -*-
"""강의스타일참고 폴더의 친근·컬러풀 슬라이드 스타일 그림.
합성곱 특징 추출(참고 _05), 컬러 채널 합성곱 시트 등.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Polygon, FancyBboxPatch
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
import matplotlib as mpl
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 140
mpl.rcParams["savefig.bbox"] = "tight"
import os
OUT = os.path.dirname(__file__)

# --- 우리 책 고유 팔레트 (브랜드 #1AB18B 청록-그린 축, 참고 이미지와 구별) ---
G = "#1AB18B"    # 브랜드(강조/정답/핵심)
R = "#FF6B5C"    # 코랄(중요/경고)
B = "#4C6EF5"    # 인디고(개념/보조)
O = "#F1A208"    # 앰버(주의)
PUR = "#8B5CF6"  # 바이올렛
P = "#EC4B8A"    # 로즈(알약 라벨)


def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p); plt.close(fig)
    print("saved", name, os.path.getsize(p), "bytes")


def gray_arrow(ax, x, y, dx=0.9, w=0.34, color="#c7ccd3"):
    ax.add_patch(FancyArrow(x, y, dx, 0, width=w, head_width=w*1.9, head_length=0.35,
                            length_includes_head=True, color=color, zorder=3))


def rich_text(ax, x, y, pieces, fs=16, align0=0.0):
    """색 강조 인라인 텍스트를 자동 폭으로 배치. pieces=[(text,color,bold),...]"""
    boxes = [TextArea(t, textprops=dict(color=c, fontsize=fs,
                                        fontweight="bold" if b else "normal"))
             for (t, c, b) in pieces]
    pack = HPacker(children=boxes, align="baseline", pad=0, sep=0)
    ab = AnnotationBbox(pack, (x, y), xycoords="data", box_alignment=(align0, 0.5),
                        frameon=False, zorder=6)
    ax.add_artist(ab)


def big_asterisk(ax, cx, cy, r=0.32, lw=5, color="#111"):
    for ang in (0, 45, 90, 135):
        a = np.deg2rad(ang)
        ax.plot([cx-r*np.cos(a), cx+r*np.cos(a)], [cy-r*np.sin(a), cy+r*np.sin(a)],
                color=color, lw=lw, solid_capstyle="round", zorder=5)


def pill(ax, cx, cy, text, fc=P, fs=14):
    """분홍 알약형 라벨."""
    ax.add_patch(FancyBboxPatch((cx-0.85, cy-0.32), 1.7, 0.64,
                 boxstyle="round,pad=0.02,rounding_size=0.32",
                 fc=fc, ec="none", zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="white", zorder=5)


def sheet(ax, x, y, w, h, fc, skew=0.42, ec="#1f2937", lw=1.6, z=2):
    """기울어진 평행사변형 시트(특징맵 한 장)."""
    pts = [(x, y), (x+w, y), (x+w+skew, y+h), (x+skew, y+h)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z))


def sheet_stack(ax, x, y, w, h, colors, dx=0.18, dy=0.0, skew=0.42):
    """여러 채널 시트를 겹쳐 그림(뒤→앞)."""
    n = len(colors)
    for i, c in enumerate(colors):
        sheet(ax, x+i*dx, y+i*dy, w, h, c, skew=skew, z=2+i)


def callout(ax, x, y, w, h, lines, ec="#f0a01f", fc="#fff6e6", fs=12):
    """주황 테두리 콜아웃 박스 + 색 강조 줄들. lines=[[(txt,color,bold),...], ...]"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.12",
                 fc=fc, ec=ec, lw=2.4, zorder=4))
    for k, line in enumerate(lines):
        rich_text(ax, x+0.25, y+h-0.45-k*0.55, line, fs=fs)


def fig_conv_feature():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.2); ax.set_aspect("equal"); ax.axis("off")
    # 제목 (색 강조, 자동 배치)
    rich_text(ax, 0.3, 7.7, [("그래서 convolution이 어떻게 ", "#111", True),
                             ("특징을 추출", R, True), ("할까?", "#111", True)], fs=16)
    rich_text(ax, 0.3, 6.9, [("•  쭉 ", "#111", True), ("스캔", O, True),
                             ("하면서 필터와 비슷한 ", "#111", True), ("패턴(특징)", R, True),
                             ("을 찾는다   ", "#111", True),
                             ("(곱하고 더하고는 사실 내적!)", G, True)], fs=12)

    # --- 입력 이미지 (4x6), 세로 경계: 왼쪽 1, 오른쪽 2 ---
    img = np.array([[1,1,1,2,2,2]]*4, float)
    u = 0.62; ox, oy = 0.7, 1.3
    for i in range(4):
        for j in range(6):
            v = img[i, j]
            fc = "#8f8f8f" if v == 1 else "#d9d9d9"
            ax.add_patch(Rectangle((ox+j*u, oy+(3-i)*u), u, u, fc=fc, ec="white", lw=2))
            ax.text(ox+j*u+u/2, oy+(3-i)*u+u/2, f"{int(v)}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#222")
    # 현재 스캔 창(3열: j=1,2,3) 노란 강조 + 곱셈값
    for i in range(4):
        for jj, mult in zip((1, 2, 3), (-1, 0, 1)):
            ax.add_patch(Rectangle((ox+jj*u, oy+(3-i)*u), u, u, fc="none", ec="#e0a500", lw=2.2, zorder=4))
    ax.text(ox+3*u, oy-0.55, "〈 세로 특징이 있는 이미지 〉", fontsize=11, fontweight="bold",
            ha="center", color="#333")

    # --- * 기호 ---
    big_asterisk(ax, 5.55, 3.6, r=0.34, lw=5.5)

    # --- 필터 (3x3, -1 0 1, 노랑→주황 그라데이션) ---
    fx, fy = 6.4, 2.4; fu = 0.72
    fcol = ["#eef2fe", "#bcccf8", "#7f97f2"]   # 인디고 그라데이션(우리 팔레트)
    for i in range(3):
        for j in range(3):
            val = [-1, 0, 1][j]
            ax.add_patch(Rectangle((fx+j*fu, fy+(2-i)*fu), fu, fu, fc=fcol[j], ec="white", lw=2))
            ax.text(fx+j*fu+fu/2, fy+(2-i)*fu+fu/2, f"{val}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#222")
    ax.text(fx+1.5*fu, fy+3*fu+0.25, "필터 1", fontsize=12, fontweight="bold", ha="center")

    # --- 화살표 ---
    gray_arrow(ax, 9.1, 3.6, dx=1.1)

    # --- 출력 특징맵 (0 3 3 0 / 0) ---
    ox2, oy2 = 10.7, 3.0; u2 = 0.7
    out_row = [0, 3, 3, 0]
    for j, v in enumerate(out_row):
        ax.add_patch(Rectangle((ox2+j*u2, oy2), u2, u2, fc="#eef0f3", ec="white", lw=2))
        ax.text(ox2+j*u2+u2/2, oy2+u2/2, f"{v}", ha="center", va="center",
                fontsize=13, fontweight="bold", color="#222")
    ax.add_patch(Rectangle((ox2, oy2-u2), u2, u2, fc="#eef0f3", ec="white", lw=2))
    ax.text(ox2+u2/2, oy2-u2/2, "0", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(ox2+2*u2, oy2+u2+0.35, "특징 맵", fontsize=11.5, fontweight="bold", ha="center", color="#333")
    save(fig, "ch5_conv_slide.png")


def fig_pool_slide():
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_xlim(0, 15); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")
    rich_text(ax, 0.3, 7.4, [("pooling: ", "#111", True), ("가장 큰 값", R, True),
                             ("만 남기고 크기를 줄이자", "#111", True)], fs=16)
    rich_text(ax, 0.3, 6.6, [("•  ", "#111", True), ("2×2 영역", B, True),
                             ("에서 ", "#111", True), ("최댓값", R, True),
                             ("만 남김 → 조금 움직여도 결과가 잘 안 변함(", "#111", True),
                             ("이동 불변성", G, True), (")", "#111", True)], fs=12)
    # 입력 4x4
    vals = np.array([[1,3,2,4],[2,9,1,5],[6,2,8,3],[1,4,2,7]])
    u = 0.72; ox, oy = 0.8, 1.2
    wincol = ["#d0f0e6", "#dfe6ff", "#fff0cf", "#ffe0da"]   # 청록·인디고·앰버·코랄 톤
    for bi in range(2):
        for bj in range(2):
            c = wincol[bi*2+bj]
            for i in range(2):
                for j in range(2):
                    r, cc = bi*2+i, bj*2+j
                    ax.add_patch(Rectangle((ox+cc*u, oy+(3-r)*u), u, u, fc=c, ec="white", lw=2.5))
                    ax.text(ox+cc*u+u/2, oy+(3-r)*u+u/2, f"{vals[r,cc]}", ha="center",
                            va="center", fontsize=13, fontweight="bold")
    gray_arrow(ax, 4.6, 3.0, dx=1.2)
    # 출력 2x2 (각 2x2 블록의 max)
    out = np.array([[vals[0:2,0:2].max(), vals[0:2,2:4].max()],
                    [vals[2:4,0:2].max(), vals[2:4,2:4].max()]])
    ox2, oy2 = 6.6, 1.9; u2 = 0.95
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((ox2+j*u2, oy2+(1-i)*u2), u2, u2, fc=wincol[i*2+j], ec="white", lw=2.5))
            ax.text(ox2+j*u2+u2/2, oy2+(1-i)*u2+u2/2, f"{out[i,j]}", ha="center",
                    va="center", fontsize=16, fontweight="bold")
    callout(ax, 9.3, 3.2, 5.0, 1.5,
            [[("각 영역에서 ", "#111", True), ("가장 큰 값", R, True), ("만!", "#111", True)],
             [("→ 크기 절반, 계산량 ↓", O, True)]], fs=12)
    save(fig, "ch5_pool_slide.png")


def synth_photo(ax, x, y, w, h):
    """간단 합성 '사진'(잔디+흰 강아지 느낌)."""
    ax.add_patch(Rectangle((x, y), w, h*0.62, fc="#7cae54", ec="none", zorder=2))       # 잔디
    ax.add_patch(Rectangle((x, y+h*0.62), w, h*0.38, fc="#bfe0f2", ec="none", zorder=2))  # 하늘
    ax.add_patch(plt.matplotlib.patches.Ellipse((x+w*0.45, y+h*0.5), w*0.5, h*0.5,
                 fc="#f4f4ef", ec="#d8d8d0", lw=1, zorder=3))                             # 강아지
    ax.add_patch(plt.matplotlib.patches.Ellipse((x+w*0.45, y+h*0.72), w*0.28, h*0.28,
                 fc="#f8f8f4", ec="#d8d8d0", lw=1, zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#333", lw=1.5, zorder=4))


def fig_cnn_pipeline_slide():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 17); ax.set_ylim(0, 7); ax.set_aspect("equal"); ax.axis("off")
    rich_text(ax, 3.2, 6.4, [("convolution, pooling ", "#111", True),
                             ("반복", R, True), ("하다보면?", "#111", True)], fs=17)
    synth_photo(ax, 0.4, 2.0, 2.3, 2.6)
    def lab(ax, x, t): ax.text(x, 3.0, t, fontsize=10.5, color=B, fontweight="bold", ha="center")
    gray_arrow(ax, 2.9, 3.3, dx=0.8); lab(ax, 3.3, "conv")
    sheet_stack(ax, 4.0, 1.7, 1.6, 3.0, ["#8FB8F2", "#86D9C4"], dx=0.16)
    gray_arrow(ax, 6.2, 3.3, dx=0.8); lab(ax, 6.6, "pooling")
    sheet_stack(ax, 7.2, 2.1, 1.2, 2.2, ["#8FB8F2", "#86D9C4"], dx=0.14)
    gray_arrow(ax, 8.9, 3.3, dx=0.8); lab(ax, 9.3, "conv")
    sheet_stack(ax, 9.8, 2.0, 1.1, 2.3, ["#4C6EF5", "#86D9C4", "#F1A208", "#FF6B5C"], dx=0.16)
    gray_arrow(ax, 11.9, 3.3, dx=0.8); lab(ax, 12.3, "pooling")
    sheet_stack(ax, 12.8, 2.5, 0.7, 1.4, ["#4C6EF5", "#86D9C4", "#F1A208", "#FF6B5C"], dx=0.13)
    gray_arrow(ax, 14.0, 3.3, dx=0.8); lab(ax, 14.4, "conv")
    ax.text(15.4, 3.3, "…", fontsize=20, fontweight="bold", va="center")
    save(fig, "ch5_pipeline_slide.png")


def fig_classify_slide():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.set_aspect("equal"); ax.axis("off")
    rich_text(ax, 0.3, 8.4, [("마지막엔 ", "#111", True), ("MLP", B, True),
                             ("가 특징을 보고 분류!", "#111", True)], fs=16)
    # 특징 맵 그리드 (잔디 배경 + 부위)
    grid = [["잔디","잔디","털","잔디","잔디"],
            ["잔디","귀","눈","귀","잔디"],
            ["잔디","털","코","털","잔디"],
            ["잔디","털","입","털","잔디"],
            ["잔디","잔디","잔디","잔디","잔디"]]
    cmap = {"잔디":"#7FCBA4","귀":"#F3D06B","눈":"#B79BEA","코":"#F19FC2","입":"#8FBEF0","털":"#EEF1F3"}
    u = 0.8; ox, oy = 0.6, 2.0
    for i in range(5):
        for j in range(5):
            key = grid[i][j]
            ax.add_patch(Rectangle((ox+j*u, oy+(4-i)*u), u, u, fc=cmap[key], ec="white", lw=2.5))
            ax.text(ox+j*u+u/2, oy+(4-i)*u+u/2, key, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="#333")
    gray_arrow(ax, 5.2, 4.0, dx=1.2)
    rich_text(ax, 6.7, 5.4, [("흰 털, 둥근 눈·코,", "#111", True)], fs=11.5)
    rich_text(ax, 6.7, 4.95, [("큰 귀를 가졌으니…", "#111", True)], fs=11.5)
    pill(ax, 12.7, 6.2, "강아지", fc=G); ax.text(14.0, 6.2, "◎", fontsize=17, color=G, va="center", ha="center")
    pill(ax, 12.7, 4.8, "고양이", fc=P); big_x(ax, 14.0, 4.8)
    pill(ax, 12.7, 3.4, "소", fc=P); big_x(ax, 14.0, 3.4)
    save(fig, "ch5_classify_slide.png")


def big_x(ax, cx, cy, r=0.28, lw=5, color="#c7ccd3"):
    ax.plot([cx-r, cx+r], [cy-r, cy+r], color=color, lw=lw, solid_capstyle="round")
    ax.plot([cx-r, cx+r], [cy+r, cy-r], color=color, lw=lw, solid_capstyle="round")


if __name__ == "__main__":
    fig_conv_feature()
    fig_pool_slide()
    fig_cnn_pipeline_slide()
    fig_classify_slide()
    print("done slide-style")
