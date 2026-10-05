# -*- coding: utf-8 -*-
"""6주차 설명 보강용 그림. 실행: python gen_paper6b.py [이름 ...]  (슬라이드용: PAPER_SLIDE=1.45)"""
import sys
from _paper import *
from matplotlib.patches import Ellipse, Polygon
from gen_paper6 import _cuboid


def draw_scene(ax, size=10.0, cat=True, cx=None, cy=None, w=None, h=None):
    """단순한 장면: 배경 + 고양이 모양(타원 몸통, 원 머리, 귀)."""
    ax.add_patch(Rectangle((0, 0), size, size, fc="#eef3f8", ec=LINE, lw=0.7, zorder=0))
    ax.add_patch(Rectangle((0, 0), size, size * 0.3, fc="#e4ebe2", ec="none", zorder=0))
    if cat:
        cx = cx if cx is not None else size * 0.49
        cy = cy if cy is not None else size * 0.55
        w = w if w is not None else size * 0.40
        h = h if h is not None else size * 0.30
        ax.add_patch(Ellipse((cx, cy - h * 0.1), w, h * 0.8, fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=2))
        ax.add_patch(Circle((cx + w * 0.34, cy + h * 0.28), h * 0.34, fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=3))
        for dxe in (-0.14, 0.14):
            x0 = cx + w * 0.34 + dxe * h * 1.6
            ax.add_patch(Polygon([(x0 - h * 0.1, cy + h * 0.5), (x0 + h * 0.1, cy + h * 0.5), (x0, cy + h * 0.78)],
                                 fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=2))


# ───────────────────────── YOLO의 입력과 출력 ─────────────────────────
def fig_yolo_io():
    fig, ax = canvas(12.4, 4.9, (0, 12.4), (-1.0, 3.9))
    # 입력
    ax2 = ax
    s = 2.7
    x0, y0 = 0.2, 0.55
    # 장면은 데이터 좌표계에 직접 그린다
    ax.add_patch(Rectangle((x0, y0), s, s, fc="#eef3f8", ec=LINE, lw=0.7))
    ax.add_patch(Rectangle((x0, y0), s, s * 0.3, fc="#e4ebe2", ec="none"))
    cxs, cys = x0 + s * 0.49, y0 + s * 0.55
    ax.add_patch(Ellipse((cxs, cys - 0.08), 1.1, 0.65, fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=2))
    ax.add_patch(Circle((cxs + 0.4, cys + 0.2), 0.27, fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=3))
    for dxe in (-0.12, 0.12):
        ax.add_patch(Polygon([(cxs + 0.4 + dxe - 0.08, cys + 0.42), (cxs + 0.4 + dxe + 0.08, cys + 0.42),
                              (cxs + 0.4 + dxe, cys + 0.62)], fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=2))
    label(ax, x0 + s / 2, y0 + s + 0.32, "① 입력", fs=8.6, bold=True)
    label(ax, x0 + s / 2, y0 - 0.32, "이미지 1장", fs=8, color=INK)
    label(ax, x0 + s / 2, y0 - 0.62, "448×448×3 (RGB)", fs=7.4, color=SUB)
    arrow(ax, (3.1, y0 + s / 2), (3.75, y0 + s / 2), color=SUB, lw=1.2, ms=8)
    # 네트워크
    box(ax, 3.85, y0 + 0.2, 2.0, s - 0.4, "", fc=F_BLUE, ec=BLUE, lw=1.0)
    label(ax, 4.85, y0 + s / 2 + 0.45, "② 신경망", fs=8.6, bold=True)
    label(ax, 4.85, y0 + s / 2 - 0.08, "합성곱 24층", fs=7.6)
    label(ax, 4.85, y0 + s / 2 - 0.4, "+ 완전연결 2층", fs=7.6)
    label(ax, 4.85, y0 - 0.32, "이미지를 한 번만 통과", fs=7.4, color=SUB)
    arrow(ax, (5.95, y0 + s / 2), (6.6, y0 + s / 2), color=SUB, lw=1.2, ms=8)
    # 출력 텐서
    gx, gy, g = 6.75, 0.75, 2.1
    dx, dy = _cuboid(ax, gx, gy, g, g, 2.6, F_ORANGE, "#fdf0e0", "#f1d3ae", ec=INK, lw=0.8)
    for k in range(1, 7):
        ax.plot([gx, gx + g], [gy + k * g / 7] * 2, color=ORANGE, lw=0.4, zorder=4)
        ax.plot([gx + k * g / 7] * 2, [gy, gy + g], color=ORANGE, lw=0.4, zorder=4)
    label(ax, gx + g / 2, gy + g + dy + 0.38, "③ 출력", fs=8.6, bold=True)
    label(ax, gx + g / 2, gy - 0.3, "7×7×30 = 숫자 1,470개", fs=8, color=INK)
    label(ax, gx + g / 2, gy - 0.6, "7×7 칸 × 칸마다 30개 값", fs=7.4, color=SUB)
    arrow(ax, (9.95, y0 + s / 2), (10.35, y0 + s / 2), color=SUB, lw=1.2, ms=8)
    # 해석 결과
    rx, ry, rs = 10.4, 0.55, 1.85
    ax.add_patch(Rectangle((rx, ry), rs, rs, fc="#eef3f8", ec=LINE, lw=0.7))
    ax.add_patch(Rectangle((rx, ry), rs, rs * 0.3, fc="#e4ebe2", ec="none"))
    cx2, cy2 = rx + rs * 0.49, ry + rs * 0.55
    ax.add_patch(Ellipse((cx2, cy2 - 0.05), 0.76, 0.44, fc="#aab2bb", ec="#7d8791", lw=0.7, zorder=2))
    ax.add_patch(Circle((cx2 + 0.28, cy2 + 0.14), 0.18, fc="#aab2bb", ec="#7d8791", lw=0.7, zorder=3))
    ax.add_patch(Rectangle((rx + 0.38, ry + 0.62), 1.15, 0.85, fc="none", ec=RED, lw=1.8, zorder=5))
    ax.text(rx + 0.38, ry + 1.5, "고양이 0.76", fontsize=7.2, color="white", va="bottom", zorder=6,
            bbox=dict(fc=RED, ec="none", pad=1.5))
    label(ax, rx + rs / 2, ry + rs + 0.32, "④ 해석", fs=8.6, bold=True)
    label(ax, rx + rs / 2, ry - 0.32, "상자 + 클래스 + 점수", fs=8, color=INK)
    # 학습/추론 구분
    box(ax, 0.2, -0.95, 12.0, 0.5, "", fc=F_GRAY, ec=LINE, r=0.05)
    label(ax, 6.2, -0.7, "학습: 이미지와 함께 정답(물체마다 상자 + 클래스)을 주고, 출력이 정답에 가까워지도록 가중치를 갱신     추론: 이미지만 입력",
          fs=7.4)
    save(fig, "ch6_p_yolo_io")


# ───────────────────────── 격자·담당 칸 (설명 주석 포함) ─────────────────────────
def fig_yolo_cell():
    fig, ax = canvas(10.4, 5.0, (0, 10.4), (-0.3, 4.7))
    S, g = 7, 4.4
    gx, gy = 0.2, 0.1
    c = g / S
    # 장면
    ax.add_patch(Rectangle((gx, gy), g, g, fc="#eef3f8", ec=LINE, lw=0.7))
    ax.add_patch(Rectangle((gx, gy), g, g * 0.3, fc="#e4ebe2", ec="none"))
    ccx, ccy = gx + 3.58 * c, gy + 3.45 * c
    ax.add_patch(Ellipse((ccx, ccy - 0.12), 1.75, 1.05, fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=2))
    ax.add_patch(Circle((ccx + 0.62, ccy + 0.33), 0.42, fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=3))
    for dxe in (-0.17, 0.17):
        ax.add_patch(Polygon([(ccx + 0.62 + dxe - 0.12, ccy + 0.66), (ccx + 0.62 + dxe + 0.12, ccy + 0.66),
                              (ccx + 0.62 + dxe, ccy + 0.98)], fc="#aab2bb", ec="#7d8791", lw=0.8, zorder=2))
    bx0, by0, bx1, by1 = ccx - 1.05, ccy - 0.8, ccx + 1.05, ccy + 0.95      # 정답 상자
    mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
    ci, cj = int((mx - gx) // c), int((my - gy) // c)
    ax.add_patch(Rectangle((gx + ci * c, gy + cj * c), c, c, fc=F_ORANGE, ec="none", zorder=1))
    for k in range(S + 1):
        ax.plot([gx, gx + g], [gy + k * c] * 2, color=LINE, lw=0.5, zorder=3)
        ax.plot([gx + k * c] * 2, [gy, gy + g], color=LINE, lw=0.5, zorder=3)
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fc="none", ec=GREEN, lw=1.9, zorder=5))
    ax.add_patch(Rectangle((gx + ci * c, gy + cj * c), c, c, fc="none", ec=ORANGE, lw=2.0, zorder=6))
    ax.plot([mx], [my], "o", color=RED, ms=5, zorder=7)
    label(ax, gx + g / 2, gy + g + 0.22, "입력 이미지를 7×7 격자로 나눈 모습", fs=8.2, color=SUB)

    def note(y, title, body, col, xy, xt=5.3):
        ax.text(xt, y, title, fontsize=8.8, fontweight="bold", color=col, va="center")
        ax.text(xt, y - 0.36, body, fontsize=7.8, color=INK, va="top", linespacing=1.35)
        arrow(ax, (xt - 0.08, y), xy, color=col, lw=1.0, ms=6, rad=0.0)

    note(4.0, "정답 상자 (초록)", "사람이 그려 둔 물체의 위치(라벨).\n학습할 때만 쓰이고, 추론할 때는 없음", GREEN, (bx1 - 0.1, by1 - 0.05))
    note(2.75, "정답 상자의 중심 (빨간 점)", "이 점이 어느 칸 안에 있는지가 담당 칸을 정함", RED, (mx + 0.05, my))
    note(1.7, "담당 칸 (주황)", "중심이 속한 칸. 이 칸의 출력이 고양이의 상자와\n클래스를 맞히도록 학습", ORANGE,
         (gx + (ci + 1) * c - 0.03, gy + cj * c + 0.15))
    ax.text(5.3, 0.55, "나머지 48칸", fontsize=8.8, fontweight="bold", color=SUB, va="center")
    ax.text(5.3, 0.19, "\"물체 없음\"(확신도 0)을 출력하도록 학습", fontsize=7.8, color=INK, va="top")
    save(fig, "ch6_p_yolo_cell")


# ───────────────────────── 한 칸의 출력: 실제 값 예시 ─────────────────────────
def fig_yolo_vector_ex():
    fig, ax = canvas(12.4, 4.35, (0, 12.4), (-1.35, 3.0))
    vals1 = [0.55, 0.55, 0.46, 0.46, 0.90]
    vals2 = [0.40, 0.60, 0.50, 0.44, 0.70]
    names = ["x", "y", "w", "h", "확신도"]
    cw, y0, hh = 0.66, 0.9, 0.62
    x = 0.3
    xs = []
    for grp, vals, fc, col in ((1, vals1, F_BLUE, BLUE), (2, vals2, F_PURPLE, PURPLE)):
        gx0 = x
        for i, (n, v) in enumerate(zip(names, vals)):
            ax.add_patch(Rectangle((x, y0), cw, hh, fc=fc, ec=LINE, lw=0.6, zorder=2))
            label(ax, x + cw / 2, y0 + hh / 2, f"{v:.2f}", fs=8.4)
            label(ax, x + cw / 2, y0 + hh + 0.22, n, fs=7.6, color=col, bold=True)
            x += cw
        xs.append((gx0, x, col, grp))
        x += 0.0
    # 클래스 20칸
    cw2 = 0.25
    gx0 = x
    classes = {7: ("고양이", 0.85), 11: ("강아지", 0.05)}
    for i in range(20):
        ax.add_patch(Rectangle((x, y0), cw2, hh, fc=F_GREEN, ec=LINE, lw=0.5, zorder=2))
        if i in classes:
            label(ax, x + cw2 / 2, y0 + hh / 2, "", fs=6)
        x += cw2
    gx1 = x
    for i, (nm, v) in classes.items():
        cx = gx0 + i * cw2 + cw2 / 2
        ax.add_patch(Rectangle((gx0 + i * cw2, y0), cw2, hh, fc="#bfe3cd", ec=GREEN, lw=1.0, zorder=3))
        ax.plot([cx, cx], [y0 + hh, y0 + hh + 0.38], color=GREEN, lw=0.8)
        label(ax, cx, y0 + hh + 0.62, f"{nm}\n{v:.2f}", fs=7.4, color=GREEN)
    label(ax, (gx0 + gx1) / 2 + 0.7, y0 + hh / 2, "…", fs=9, color=SUB)
    # 구간 표시와 설명
    def brace(a, b, y, col, t1, t2):
        ax.plot([a + 0.04, b - 0.04], [y, y], color=col, lw=1.4)
        label(ax, (a + b) / 2, y - 0.3, t1, fs=8.4, bold=True)
        label(ax, (a + b) / 2, y - 0.72, t2, fs=7.4, color=INK)
    brace(xs[0][0], xs[0][1], 0.62, BLUE, "상자 1 (5개)", "칸 안 가로 55%·세로 55% 지점이 중심,\n너비 46%·높이 46%, 확신도 0.90")
    brace(xs[1][0], xs[1][1], 0.62, PURPLE, "상자 2 (5개)", "같은 칸에서 낸 두 번째 후보\n(확신도 0.70)")
    brace(gx0, gx1, 0.62, GREEN, "클래스 확률 (20개)", "이 칸의 물체가 각 종류일 확률\n고양이 0.85 → 이 칸은 고양이")
    label(ax, 6.2, 2.75, "한 칸이 내는 값 30개의 예 (칸 1개 = 길이 30의 벡터)", fs=9, bold=True)
    save(fig, "ch6_p_yolo_vector_ex")


# ───────────────────────── 박스 복원: 픽셀 값으로 따라가기 ─────────────────────────
def fig_yolo_decode_ex():
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.7), gridspec_kw={"width_ratios": [1.2, 1]})
    P = 448; cell = 64
    cx, cy = (3 + 0.55) * cell, (3 + 0.55) * cell
    w = h = 0.46 * P
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    ax = axs[0]
    ax.set_xlim(-45, P + 12); ax.set_ylim(P + 12, -40); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) 입력 이미지 448×448 픽셀", fontsize=8.8, fontweight="bold", pad=4)
    ax.add_patch(Rectangle((0, 0), P, P, fc="#eef3f8", ec=LINE, lw=0.7))
    ax.add_patch(Rectangle((192, 192), cell, cell, fc=F_ORANGE, ec=ORANGE, lw=1.6, zorder=3))
    for k in range(8):
        ax.plot([0, P], [k * cell] * 2, color=LINE, lw=0.4, zorder=2); ax.plot([k * cell] * 2, [0, P], color=LINE, lw=0.4, zorder=2)
    for k in range(0, 8):
        ax.text(k * cell, -10, str(k * cell), fontsize=5.8, ha="center", color=SUB)
        if k:
            ax.text(-8, k * cell, str(k * cell), fontsize=5.8, ha="right", va="center", color=SUB)
    ax.add_patch(Rectangle((x1, y1), w, h, fc="none", ec=RED, lw=1.9, zorder=5))
    ax.plot([cx], [cy], "o", color=RED, ms=4.5, zorder=6)
    ax.text(x1, y1 - 8, f"({x1:.0f}, {y1:.0f})", fontsize=7.2, color=RED, ha="left", va="bottom", fontweight="bold")
    ax.text(x2, y2 + 8, f"({x2:.0f}, {y2:.0f})", fontsize=7.2, color=RED, ha="right", va="top", fontweight="bold")
    ax.text(192 + 66, 256 + 22, "담당 칸\n(열 3, 행 3)", fontsize=6.6, color=ORANGE, va="top")
    ax = axs[1]
    ax.set_xlim(-60, 100); ax.set_ylim(86, -40); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) 담당 칸 확대 (64×64 픽셀)", fontsize=8.8, fontweight="bold", pad=4)
    ax.add_patch(Rectangle((0, 0), 64, 64, fc=F_ORANGE, ec=ORANGE, lw=1.8))
    ox, oy = 0.55 * 64, 0.55 * 64
    ax.plot([ox], [oy], "o", color=RED, ms=6, zorder=5)
    ax.plot([0, ox], [oy, oy], color=BLUE, lw=1.2, ls="--"); ax.plot([ox, ox], [0, oy], color=BLUE, lw=1.2, ls="--")
    arrow(ax, (0, -8), (ox, -8), color=BLUE, lw=1.1, style="<|-|>", ms=6)
    ax.text(ox / 2, -14, "x = 0.55 → +35 픽셀", fontsize=7.4, color=BLUE, ha="center", va="bottom")
    arrow(ax, (-8, 0), (-8, oy), color=BLUE, lw=1.1, style="<|-|>", ms=6)
    ax.text(-13, oy / 2, "y = 0.55\n→ +35 픽셀", fontsize=7.2, color=BLUE, ha="right", va="center")
    ax.text(0, 70, "칸의 왼쪽 위 = (192, 192) 픽셀", fontsize=7.4, color=SUB, va="top")
    ax.text(ox + 3, oy + 5, f"중심 ({cx:.0f}, {cy:.0f})", fontsize=7.4, color=RED, va="top", fontweight="bold")
    fig.subplots_adjust(wspace=0.05)
    save(fig, "ch6_p_yolo_decode_ex")
    print("계산 확인:", round(cx, 1), round(w, 1), round(x1, 1), round(x2, 1))


# ───────────────────────── 2단계 vs 1단계 (쉬운 설명) ─────────────────────────
def fig_detectors2():
    fig, ax = canvas(9.7, 4.4, (0, 9.7), (0, 4.4))
    bh = 1.0

    def row(y, title, items, note):
        label(ax, 0.05, y + 0.95, title, fs=9.2, ha="left", bold=True)
        x = 0.1
        for k, (t, fc, ec, w) in enumerate(items):
            box(ax, x, y - bh / 2, w, bh, t, fc=fc, ec=ec, fs=7.6)
            if k < len(items) - 1:
                arrow(ax, (x + w + 0.03, y), (x + w + 0.3, y), color=SUB, lw=1.0, ms=6)
            x += w + 0.33
        label(ax, 0.1, y - 0.82, note, fs=7.8, ha="left", color=SUB)

    row(3.25, "2단계 검출기 (예: Faster R-CNN)",
        [("입력\n이미지", F_GRAY, GRAY, 1.15), ("CNN\n특징 추출", F_BLUE, BLUE, 1.3),
         ("① 후보 영역 제안\n물체가 있을 법한 곳을\n수백~수천 개 먼저 고름", F_ORANGE, ORANGE, 2.5),
         ("② 후보마다\n분류 + 상자 보정", F_BLUE, BLUE, 1.9), ("결과", F_GREEN, GREEN, 1.0)],
        "후보를 먼저 좁힌 뒤 하나씩 정밀하게 확인  →  정확도 높음, 속도 느림")
    row(1.05, "1단계 검출기 (예: YOLO)",
        [("입력\n이미지", F_GRAY, GRAY, 1.15), ("CNN\n특징 추출", F_BLUE, BLUE, 1.3),
         ("모든 위치에서 한 번에\n(상자, 클래스, 점수)\n예측", F_ORANGE, ORANGE, 2.5),
         ("NMS\n중복 상자 제거", F_PURPLE, PURPLE, 1.9), ("결과", F_GREEN, GREEN, 1.0)],
        "후보 제안 없이 이미지를 한 번에 처리  →  속도 빠름(실시간), 최근에는 정확도도 근접")
    save(fig, "ch6_p_detectors2")


# ───────────────────────── YOLOv8 백본 ─────────────────────────
def fig_v8_backbone():
    fig, ax = canvas(12.4, 5.5, (0, 12.4), (-2.0, 3.6))
    stages = [("입력", "3", "640×640", 640, 3, None), ("Conv", "16", "320×320", 320, 16, None),
              ("Conv + C2f", "32", "160×160", 160, 32, None), ("Conv + C2f", "64", "80×80", 80, 64, "P3"),
              ("Conv + C2f", "128", "40×40", 40, 128, "P4"), ("Conv + C2f\n+ SPPF", "256", "20×20", 20, 256, "P5")]
    hh = lambda s: 0.5 + 2.0 * (s / 640) ** 0.5
    ww = lambda c: 0.12 + c * 0.0042
    x = 0.4
    yc = 1.55
    prev = None
    for i, (op, ch, size, s, c, tap) in enumerate(stages):
        h, w = hh(s), ww(c)
        fc = F_GRAY if i == 0 else F_BLUE
        ec = INK
        ax.add_patch(Rectangle((x, yc - h / 2), w, h, fc=fc, ec=ec, lw=0.8, zorder=3))
        if tap:
            ax.add_patch(Rectangle((x - 0.05, yc - h / 2 - 0.05), w + 0.1, h + 0.1, fc="none", ec=ORANGE, lw=1.8, zorder=4))
            label(ax, x + w / 2, yc + h / 2 + 0.3, f"{tap}", fs=9.2, bold=True, color=ORANGE)
            label(ax, x + w / 2, yc + h / 2 + 0.62, f"stride {640 // s}", fs=7, color=SUB)
        label(ax, x + w / 2, yc - h / 2 - 0.3, f"{ch}채널", fs=7.6)
        label(ax, x + w / 2, yc - h / 2 - 0.58, size, fs=7.4, color=SUB)
        if prev is not None:
            arrow(ax, (prev + 0.04, yc), (x - 0.06, yc), color=RED, lw=1.1, ms=6)
            label(ax, (prev + x) / 2, yc + 0.28, "÷2", fs=7, color=RED)
        label(ax, x + w / 2, 3.35, op, fs=7.8, bold=True, color=INK)
        prev = x + w
        x += w + 1.2
    # 설명 상자
    box(ax, 0.3, -1.95, 11.9, 1.3, "", fc=F_GRAY, ec=LINE, r=0.05)
    label(ax, 0.5, -0.95, "Conv : 3×3 합성곱(stride 2로 크기 ½) + 배치 정규화 + SiLU", fs=7.6, ha="left")
    label(ax, 0.5, -1.3, "C2f : 채널을 둘로 나눠 한쪽만 작은 블록 여러 개에 통과시키고 모든 중간 결과를 이어 붙임", fs=7.6, ha="left")
    label(ax, 0.5, -1.65, "SPPF : 5×5 최대 풀링을 3번 연속 적용한 결과를 이어 붙여 넓은 범위의 문맥을 한 번에 요약", fs=7.6, ha="left")
    save(fig, "ch6_p_v8_backbone")


# ───────────────────────── YOLOv8 넥 (PAN-FPN) ─────────────────────────
def fig_v8_neck():
    fig, ax = canvas(12.4, 6.4, (0, 12.4), (-1.6, 5.0))
    rows = {"P3": 3.9, "P4": 2.35, "P5": 0.8}
    bw, bh = 1.7, 0.78

    def node(x, y, t, sub, fc=F_BLUE, ec=BLUE, lw=0.9):
        box(ax, x, y - bh / 2, bw, bh, "", fc=fc, ec=ec, lw=lw, r=0.06)
        label(ax, x + bw / 2, y + 0.13, t, fs=8, bold=True)
        label(ax, x + bw / 2, y - 0.16, sub, fs=7, color=SUB)

    cols = [0.2, 3.3, 6.4, 9.5]
    for c, t in zip(cols, ["백본 출력", "하향식 (위로 올리기)", "상향식 (아래로 내리기)", "넥 출력 → 헤드"]):
        label(ax, c + bw / 2, 4.75, t, fs=8.4, bold=True)
    node(cols[0], rows["P3"], "P3", "64채널 · 80×80")
    node(cols[0], rows["P4"], "P4", "128채널 · 40×40")
    node(cols[0], rows["P5"], "P5", "256채널 · 20×20")
    node(cols[1], rows["P4"], "T4", "128채널 · 40×40", F_GREEN, GREEN)
    node(cols[1], rows["P3"], "N3", "64채널 · 80×80", F_GREEN, GREEN)
    node(cols[2], rows["P4"], "N4", "128채널 · 40×40", F_ORANGE, ORANGE)
    node(cols[2], rows["P5"], "N5", "256채널 · 20×20", F_ORANGE, ORANGE)
    node(cols[3], rows["P3"], "N3", "→ 헤드 (작은 물체)", F_GREEN, GREEN, 1.5)
    node(cols[3], rows["P4"], "N4", "→ 헤드 (중간 물체)", F_ORANGE, ORANGE, 1.5)
    node(cols[3], rows["P5"], "N5", "→ 헤드 (큰 물체)", F_ORANGE, ORANGE, 1.5)
    # 하향식
    arrow(ax, (cols[0] + bw + 0.05, rows["P5"] + 0.15), (cols[1] + 0.2, rows["P4"] - bh / 2 - 0.02), color=GREEN, lw=1.2, ms=7)
    label(ax, 3.3, 1.2, "↑ 2배 키움", fs=7, color=GREEN)
    arrow(ax, (cols[0] + bw + 0.05, rows["P4"] + 0.15), (cols[1] - 0.02, rows["P4"] + 0.15), color=GRAY, lw=1.0, ls=(0, (3, 2)), ms=6)
    arrow(ax, (cols[1] + bw / 2, rows["P4"] + bh / 2 + 0.02), (cols[1] + bw / 2, rows["P3"] - bh / 2 - 0.02), color=GREEN, lw=1.2, ms=7)
    label(ax, cols[1] + bw / 2 + 0.72, 3.15, "↑ 2배 키움", fs=7, color=GREEN)
    arrow(ax, (cols[0] + bw + 0.05, rows["P3"] + 0.05), (cols[1] - 0.02, rows["P3"] + 0.05), color=GRAY, lw=1.0, ls=(0, (3, 2)), ms=6)
    # 상향식
    arrow(ax, (cols[1] + bw + 0.05, rows["P3"] - 0.1), (cols[2] + 0.25, rows["P4"] + bh / 2 + 0.02), color=RED, lw=1.2, ms=7)
    label(ax, 7.35, 3.5, "↓ Conv(stride 2)", fs=7, color=RED)
    arrow(ax, (cols[1] + bw + 0.05, rows["P4"] - 0.05), (cols[2] - 0.02, rows["P4"] - 0.05), color=GRAY, lw=1.0, ls=(0, (3, 2)), ms=6)
    arrow(ax, (cols[2] + bw / 2, rows["P4"] - bh / 2 - 0.02), (cols[2] + bw / 2, rows["P5"] + bh / 2 + 0.02), color=RED, lw=1.2, ms=7)
    label(ax, cols[2] + bw / 2 + 0.95, 1.6, "↓ Conv(stride 2)", fs=7, color=RED)
    arrow(ax, (cols[0] + bw + 0.05, rows["P5"] - 0.12), (cols[2] - 0.02, rows["P5"] - 0.12), color=GRAY, lw=1.0, ls=(0, (3, 2)), ms=6)
    # 출력으로
    for r in ("P3", "P4", "P5"):
        pass
    arrow(ax, (cols[1] + bw + 0.05, rows["P3"]), (cols[3] - 0.03, rows["P3"]), color=SUB, lw=1.0, ms=6)
    arrow(ax, (cols[2] + bw + 0.05, rows["P4"]), (cols[3] - 0.03, rows["P4"]), color=SUB, lw=1.0, ms=6)
    arrow(ax, (cols[2] + bw + 0.05, rows["P5"]), (cols[3] - 0.03, rows["P5"]), color=SUB, lw=1.0, ms=6)
    # 설명
    box(ax, 0.2, -1.55, 12.0, 1.5, "", fc=F_GRAY, ec=LINE, r=0.05)
    label(ax, 0.4, -0.3, "점선 = 같은 크기의 특징을 채널 방향으로 이어 붙임(Concat), 이어 붙인 뒤 C2f로 다시 섞음", fs=7.4, ha="left")
    label(ax, 0.4, -0.65, "T4 = C2f(↑P5 + P4)      N3 = C2f(↑T4 + P3)", fs=7.4, ha="left", color=SUB)
    label(ax, 0.4, -1.0, "N4 = C2f(↓N3 + T4)      N5 = C2f(↓N4 + P5)", fs=7.4, ha="left", color=SUB)
    label(ax, 0.4, -1.35, "채널: 256+128→128 · 128+64→64 · 64+128→128 · 128+256→256", fs=7.4, ha="left", color=SUB)
    save(fig, "ch6_p_v8_neck")


# ───────────────────────── YOLOv8 헤드 ─────────────────────────
def fig_v8_head():
    fig, ax = canvas(12.4, 5.6, (0, 12.4), (-1.2, 4.7))
    rows = [(3.75, "N3", "64채널 · 80×80", "80×80", 6400), (2.1, "N4", "128채널 · 40×40", "40×40", 1600),
            (0.45, "N5", "256채널 · 20×20", "20×20", 400)]
    for y, nm, sub, hw, n in rows:
        fc, ec = (F_GREEN, GREEN) if nm == "N3" else (F_ORANGE, ORANGE)
        box(ax, 0.2, y - 0.45, 1.9, 0.9, "", fc=fc, ec=ec, lw=0.9)
        label(ax, 1.15, y + 0.14, nm, fs=8.4, bold=True); label(ax, 1.15, y - 0.2, sub, fs=7, color=SUB)
        arrow(ax, (2.15, y + 0.15), (2.65, y + 0.42), color=SUB, lw=1.0, ms=6)
        arrow(ax, (2.15, y - 0.15), (2.65, y - 0.42), color=SUB, lw=1.0, ms=6)
        box(ax, 2.7, y + 0.12, 3.6, 0.62, "박스 분기\nConv 3×3 → Conv 3×3 → Conv 1×1", fc=F_ORANGE, ec=ORANGE, fs=7.0)
        box(ax, 2.7, y - 0.74, 3.6, 0.62, "클래스 분기\nConv 3×3 → Conv 3×3 → Conv 1×1", fc=F_PURPLE, ec=PURPLE, fs=7.0)
        arrow(ax, (6.35, y + 0.43), (6.8, y + 0.43), color=ORANGE, lw=1.0, ms=6)
        arrow(ax, (6.35, y - 0.43), (6.8, y - 0.43), color=PURPLE, lw=1.0, ms=6)
        label(ax, 6.85, y + 0.43, "64채널", fs=7.8, ha="left", bold=True, color=ORANGE)
        label(ax, 6.85, y - 0.43, "80채널", fs=7.8, ha="left", bold=True, color=PURPLE)
        label(ax, 8.95, y, f"{hw} = 위치\n{n:,}개", fs=7.2, color=INK)
        arrow(ax, (9.75, y), (10.25, y), color=SUB, lw=1.0, ms=6)
    box(ax, 10.3, 0.35, 2.0, 3.9, "", fc="white", ec=LINE, r=0.07)
    for yy, t, b_, c in [(3.85, "세 스케일 결합", True, INK), (3.4, "6,400+1,600+400", False, INK),
                         (3.0, "= 8,400 위치", True, INK), (2.4, "위치마다 후보 1개", False, SUB),
                         (1.7, "박스 64 + 클래스 80", False, INK), (1.3, "= 144개 값", False, INK),
                         (0.7, "디코딩 후 (84, 8400)", False, SUB)]:
        label(ax, 11.3, yy, t, fs=7.0, bold=b_, color=c)
    label(ax, 0.2, -0.45, "박스 64 = 변 4개(l, t, r, b) × 분포 16구간 (DFL)", fs=7.4, ha="left", color=SUB)
    label(ax, 0.2, -0.85, "클래스 80 = COCO 80종 각각의 점수 (sigmoid)", fs=7.4, ha="left", color=SUB)
    save(fig, "ch6_p_v8_head")


# ───────────────────────── U-Net의 입력과 출력 ─────────────────────────
def _unet_scene(n=64, seed=3):
    rng = np.random.RandomState(seed)
    img = np.zeros((n, n, 3)) + 0.9
    mask = np.zeros((n, n), dtype=int)
    yy, xx = np.mgrid[0:n, 0:n]
    shapes = [("c", 18, 20, 9, 1), ("c", 44, 16, 7, 1), ("r", 40, 42, 14, 2), ("r", 14, 46, 10, 2)]
    for kind, cx, cy, r, k in shapes:
        m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2) if kind == "c" else ((abs(xx - cx) <= r) & (abs(yy - cy) <= r))
        mask[m] = k
    col = {0: np.array([0.93, 0.94, 0.95]), 1: np.array([0.80, 0.70, 0.66]), 2: np.array([0.62, 0.70, 0.78])}
    for k, c in col.items():
        img[mask == k] = c
    img += rng.normal(0, 0.03, img.shape)
    return np.clip(img, 0, 1), mask


def fig_unet_io():
    img, mask = _unet_scene()
    cmap_cols = np.array([[0.93, 0.94, 0.95], [217 / 255, 130 / 255, 43 / 255], [47 / 255, 109 / 255, 178 / 255]])
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.7), gridspec_kw={"width_ratios": [1, 0.55, 1]})
    axs[0].imshow(img); axs[0].set_title("① 입력 이미지\n(N, 3, H, W) — RGB", fontsize=8.6, fontweight="bold", pad=5)
    axs[2].imshow(cmap_cols[mask]); axs[2].set_title("③ 출력: 픽셀마다 클래스 번호\n(N, K, H, W) 점수 → 가장 큰 클래스", fontsize=8.6, fontweight="bold", pad=5)
    for a in (axs[0], axs[2]):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color(LINE)
    ax = axs[1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.05, 0.36, 0.9, 0.3, "② U-Net", fc=F_BLUE, ec=BLUE, fs=9, bold=True)
    arrow(ax, (0.0, 0.51), (0.05, 0.51), color=SUB, lw=1.2, ms=8)
    arrow(ax, (0.95, 0.51), (1.0, 0.51), color=SUB, lw=1.2, ms=8)
    ax.text(0.5, 0.2, "입력과 같은 크기의\n출력(H×W 유지)", fontsize=7.6, ha="center", va="top", color=SUB)
    for i, (c, nm) in enumerate(zip(cmap_cols, ["0 배경", "1 원", "2 사각형"])):
        fig.patches.append(Rectangle((0.36 + 0.095 * i * 1.0, 0.04), 0.012, 0.03, transform=fig.transFigure, fc=c, ec=LINE))
        fig.text(0.375 + 0.095 * i * 1.0, 0.055, nm, fontsize=7.2, va="center")
    fig.text(0.5, -0.04, "학습 때는 입력 이미지와 정답 마스크(③과 같은 형식의 라벨)를 한 쌍으로 주고, 출력이 정답 마스크에 가까워지도록 학습합니다 (그림은 형식을 보이는 모식도)",
             ha="center", fontsize=7.4, color=SUB)
    save(fig, "ch6_p_unet_io")


# ───────────────────────── U-Net 인코더 한 단계 ─────────────────────────
def fig_unet_encoder():
    import torch
    import torch.nn.functional as F
    fig, ax = canvas(12.4, 4.9, (0, 12.4), (-0.9, 4.0))
    steps = [("입력 특징", "64채널", "256×256", 1.0, 2.2, F_GRAY), ("3×3 합성곱\n+ BN + ReLU", "128채널", "256×256", 1.5, 2.2, F_BLUE),
             ("3×3 합성곱\n+ BN + ReLU", "128채널", "256×256", 1.5, 2.2, F_BLUE), ("2×2 최대 풀링", "128채널", "128×128", 1.5, 1.4, F_ORANGE)]
    x = 0.3
    xs = []
    for i, (op, ch, size, w, h, fc) in enumerate(steps):
        yc = 1.7
        ax.add_patch(Rectangle((x, yc - h / 2), w * 0.55, h, fc=fc, ec=INK, lw=0.8, zorder=3))
        label(ax, x + w * 0.275, yc - h / 2 - 0.3, ch, fs=7.8)
        label(ax, x + w * 0.275, yc - h / 2 - 0.6, size, fs=7.4, color=SUB)
        label(ax, x + w * 0.275, yc + h / 2 + 0.45, op, fs=7.6, bold=True)
        xs.append(x + w * 0.55)
        if i < len(steps) - 1:
            arrow(ax, (x + w * 0.55 + 0.05, yc), (x + w * 0.55 + 0.75, yc), color=BLUE if i < 2 else RED, lw=1.2, ms=7)
        x += w * 0.55 + 0.85
    label(ax, 4.3, 3.8, "인코더 한 단계: 채널은 늘리고(64→128), 풀링으로 해상도는 절반(256→128)", fs=8.4, bold=True)
    # 최대 풀링 수치 예시
    a = torch.tensor([[1., 3., 2., 0.], [4., 2., 1., 5.], [0., 1., 6., 2.], [3., 2., 4., 1.]]).view(1, 1, 4, 4)
    p = F.max_pool2d(a, 2)[0, 0].numpy(); a = a[0, 0].numpy()
    ox, oy, cs = 8.0, 1.0, 0.5
    cols = [F_BLUE, F_GREEN, F_ORANGE, F_PURPLE]
    for i in range(4):
        for j in range(4):
            blk = (i // 2) * 2 + (j // 2)
            ax.add_patch(Rectangle((ox + j * cs, oy + (3 - i) * cs), cs, cs, fc=cols[blk], ec="white", lw=1.2, zorder=3))
            label(ax, ox + j * cs + cs / 2, oy + (3 - i) * cs + cs / 2, f"{a[i, j]:.0f}", fs=8)
    arrow(ax, (ox + 2.15, oy + 1.0), (ox + 2.75, oy + 1.0), color=RED, lw=1.2, ms=7)
    for i in range(2):
        for j in range(2):
            blk = i * 2 + j
            ax.add_patch(Rectangle((ox + 2.9 + j * cs, oy + 0.5 + (1 - i) * cs), cs, cs, fc=cols[blk], ec="white", lw=1.2, zorder=3))
            label(ax, ox + 2.9 + j * cs + cs / 2, oy + 0.5 + (1 - i) * cs + cs / 2, f"{p[i, j]:.0f}", fs=8.4, bold=True)
    label(ax, ox + 1.0, oy - 0.3, "4×4 입력", fs=7.4, color=SUB)
    label(ax, ox + 3.4, oy + 0.15, "2×2 출력", fs=7.4, color=SUB)
    label(ax, ox + 1.7, oy + 2.35, "최대 풀링 예시: 2×2 영역마다 최댓값만 남김", fs=7.8, bold=True)
    box(ax, 0.3, -0.85, 11.9, 0.55, "", fc=F_GRAY, ec=LINE, r=0.05)
    label(ax, 6.25, -0.575, "3×3 합성곱은 3×3 이웃을 보고, 풀링 뒤의 3×3 합성곱은 원본의 더 넓은 영역(약 2배)을 보게 됨 → 깊어질수록 큰 문맥을 파악", fs=7.6)
    save(fig, "ch6_p_unet_encoder")
    print("maxpool check:", p.tolist())


FIGS = {"yolo_io": fig_yolo_io, "yolo_cell": fig_yolo_cell, "yolo_vector_ex": fig_yolo_vector_ex,
        "yolo_decode_ex": fig_yolo_decode_ex, "detectors2": fig_detectors2, "v8_backbone": fig_v8_backbone,
        "v8_neck": fig_v8_neck, "v8_head": fig_v8_head, "unet_io": fig_unet_io, "unet_encoder": fig_unet_encoder}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(FIGS)):
        FIGS[n]()
