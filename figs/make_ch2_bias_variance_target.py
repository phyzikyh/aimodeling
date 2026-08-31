# -*- coding: utf-8 -*-
"""2장 편향-분산: 과녁(다트판) 비유 2x2.
행=편향(낮음/높음), 열=분산(낮음/높음). 탄착점의 중심 치우침=편향, 흩어짐=분산."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.font_manager as fm

for cand in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.0), dpi=130)

# (편향 offset, 분산 spread)
configs = {
    (0, 0): ((0.0, 0.0), 0.10, "낮은 편향 · 낮은 분산", "정확하고 안정적"),
    (0, 1): ((0.0, 0.0), 0.42, "낮은 편향 · 높은 분산", "평균은 맞지만 흔들림"),
    (1, 0): ((0.62, 0.42), 0.10, "높은 편향 · 낮은 분산", "일관되게 빗나감"),
    (1, 1): ((0.62, 0.42), 0.42, "높은 편향 · 높은 분산", "부정확하고 불안정"),
}

for (r, c), ((ox, oy), sp, title, sub) in configs.items():
    ax = axes[r][c]
    # 과녁 원
    for rad, col in [(1.0, "#eef2f7"), (0.66, "#dbe4ee"), (0.34, "#c3d3e6"), (0.12, "#E8402E")]:
        ax.add_patch(Circle((0, 0), rad, fc=col, ec="#b8c0cb", lw=1, zorder=1))
    # 탄착점 (편향=중심 offset, 분산=퍼짐)
    pts = rng.normal([ox, oy], sp, size=(12, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=48, c="#1f3a5f", edgecolors="white",
               linewidths=1.0, zorder=3)
    ax.plot(0, 0, "+", ms=14, mew=2.2, color="#333", zorder=4)  # 정답(중심)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.25)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#1f2937", pad=6)
    ax.text(0, -1.32, sub, ha="center", fontsize=10.5, color="#6b7280")

fig.text(0.5, 0.045,
         "가운데 십자(＋)가 실제 값. 편향 = 탄착군 중심이 정답에서 치우친 정도, 분산 = 탄착점들이 서로 흩어진 정도",
         ha="center", fontsize=10, color="#57606a")
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig("ch2_bias_variance_target.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved: ch2_bias_variance_target.png")
