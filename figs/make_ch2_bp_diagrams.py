# -*- coding: utf-8 -*-
"""2장 역전파 절: 2-2-2 예제 신경망 다이어그램(순전파/출력층 역전파/은닉층 역전파).
PPT 손그림을 깔끔하게 재현한다."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.font_manager as fm

for cand in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

RED, GREEN, GRAY = "#d1495b", "#2a9d5c", "#9aa0a8"
NODE_FC, NODE_EC = "#ffffff", "#333333"
SHADE = "#e8edf2"

# 노드 좌표
IN  = {"x1": (0, 1.1), "x2": (0, -1.1)}
HID = {"h1": (3, 1.1), "h2": (3, -1.1)}
OUT = {"o1": (6, 1.1), "o2": (6, -1.1)}
R = 0.42

# 가중치 색: 직진(11,22)=빨강, 교차(12,21)=초록
W1 = [("x1","h1",RED,  r"$w^{(1)}_{11}$", 0.22),
      ("x2","h1",GREEN,r"$w^{(1)}_{12}$", -0.30),
      ("x1","h2",GREEN,r"$w^{(1)}_{21}$", 0.30),
      ("x2","h2",RED,  r"$w^{(1)}_{22}$", -0.22)]
W2 = [("h1","o1",RED,  r"$w^{(2)}_{11}$", 0.22),
      ("h2","o1",GREEN,r"$w^{(2)}_{12}$", -0.30),
      ("h1","o2",GREEN,r"$w^{(2)}_{21}$", 0.30),
      ("h2","o2",RED,  r"$w^{(2)}_{22}$", -0.22)]


def edge(ax, p, q, color, lw=1.6, alpha=1.0, back=False):
    (x0, y0), (x1, y1) = p, q
    import numpy as np
    dx, dy = x1 - x0, y1 - y0
    L = (dx**2 + dy**2) ** 0.5
    ux, uy = dx / L, dy / L
    x0o, y0o = x0 + ux * R, y0 + uy * R
    x1o, y1o = x1 - ux * R, y1 - uy * R
    if back:
        x0o, y0o, x1o, y1o = x1o, y1o, x0o, y0o
        arr = FancyArrowPatch((x0o, y0o), (x1o, y1o), arrowstyle="-|>",
                              mutation_scale=12, color=color, lw=lw, alpha=alpha)
        ax.add_patch(arr)
    else:
        ax.plot([x0o, x1o], [y0o, y1o], color=color, lw=lw, alpha=alpha, zorder=1)


def node(ax, xy, label, shade=False, sub=None):
    c = Circle(xy, R, fc=(SHADE if shade else NODE_FC), ec=NODE_EC, lw=1.6, zorder=3)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=13, zorder=4)
    if sub:
        ax.text(xy[0] + 0.62, xy[1] + 0.55, sub, ha="left", va="center",
                fontsize=12, color="#333", zorder=5)


def wlabel(ax, p, q, color, txt, off):
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    ax.text(mx, my + off, txt, color=color, fontsize=9.5, ha="center",
            va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.85))


def bias(ax, xy, txt):
    ax.annotate("", xy=(xy[0], xy[1] + R), xytext=(xy[0], xy[1] + R + 0.55),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
    ax.text(xy[0] + 0.12, xy[1] + R + 0.62, txt, fontsize=10, color="#555",
            ha="left", va="center")


def base(ax, edges1, edges2, e1a=1.0, e2a=1.0, back1=False, back2=False,
         shade_hid=False, shade_out=False, deltas_out=None, deltas_hid=None,
         labelw=True):
    for (a, b, col, txt, off) in edges1:
        edge(ax, {**IN, **HID}[a], {**IN, **HID}[b], col, alpha=e1a, back=back1)
        if labelw and e1a > 0.6:
            wlabel(ax, {**IN, **HID}[a], {**IN, **HID}[b], col, txt, off)
    for (a, b, col, txt, off) in edges2:
        edge(ax, {**HID, **OUT}[a], {**HID, **OUT}[b], col, alpha=e2a, back=back2)
        if labelw and e2a > 0.6:
            wlabel(ax, {**HID, **OUT}[a], {**HID, **OUT}[b], col, txt, off)
    bias(ax, HID["h1"], r"$b^{(1)}_1$"); bias(ax, HID["h2"], r"$b^{(1)}_2$")
    bias(ax, OUT["o1"], r"$b^{(2)}_1$"); bias(ax, OUT["o2"], r"$b^{(2)}_2$")
    node(ax, IN["x1"], r"$x_1$"); node(ax, IN["x2"], r"$x_2$")
    node(ax, HID["h1"], r"$h_1$", shade=shade_hid,
         sub=(deltas_hid[0] if deltas_hid else None))
    node(ax, HID["h2"], r"$h_2$", shade=shade_hid,
         sub=(deltas_hid[1] if deltas_hid else None))
    node(ax, OUT["o1"], r"$o_1$", shade=shade_out,
         sub=(deltas_out[0] if deltas_out else None))
    node(ax, OUT["o2"], r"$o_2$", shade=shade_out,
         sub=(deltas_out[1] if deltas_out else None))
    ax.text(0, -2.15, "입력층", ha="center", fontsize=11, color="#555")
    ax.text(3, -2.15, "은닉층", ha="center", fontsize=11, color="#555")
    ax.text(6, -2.15, "출력층", ha="center", fontsize=11, color="#555")
    ax.text(1.5, 1.95, r"$W^{(1)}$", ha="center", fontsize=12, color="#333")
    ax.text(4.5, 1.95, r"$W^{(2)}$", ha="center", fontsize=12, color="#333")
    ax.set_xlim(-1.1, 7.5); ax.set_ylim(-2.5, 2.5); ax.axis("off")
    ax.set_aspect("equal")


def make(fname, title, **kw):
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=130)
    ax.text(-1.0, 2.35, title, fontsize=13, fontweight="bold", color="#1f2937")
    base(ax, **kw)
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", fname)


# 1) 순방향 전파
make("ch2_bp_forward.png", "순방향 전파 (forward)", edges1=W1, edges2=W2)

# 2) 출력층 역전파: 출력 노드에 δ, W2를 거꾸로
make("ch2_bp_output.png", "역전파 ① 출력층", edges1=W1, edges2=W2,
     e1a=0.18, back2=True, shade_out=True,
     deltas_out=(r"$\delta_1$", r"$\delta_2$"))

# 3) 은닉층 역전파: 은닉 노드에 δ^(1), W1를 거꾸로
make("ch2_bp_hidden.png", "역전파 ② 은닉층", edges1=W1, edges2=W2,
     e2a=0.18, back1=True, shade_hid=True, shade_out=True,
     deltas_hid=(r"$\delta^{(1)}_1$", r"$\delta^{(1)}_2$"),
     deltas_out=(r"$\delta_1$", r"$\delta_2$"))
