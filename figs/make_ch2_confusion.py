# -*- coding: utf-8 -*-
"""2장 모델 성능 평가: 혼동행렬(2x2) — 질병 진단 예시."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

for cand in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

GREEN, RED = "#e7f7ef", "#fdecea"
GTX, RTX = "#0d6e56", "#a5281c"

# 예시: 100명 중 실제 양성 20, 음성 80
cells = [
    # (row, col, 배경, 이름, 기호, 값, 글자색)
    (0, 0, GREEN, "참양성", "TP", 18, GTX),
    (0, 1, RED,   "거짓음성", "FN", 2, RTX),
    (1, 0, RED,   "거짓양성", "FP", 10, RTX),
    (1, 1, GREEN, "참음성", "TN", 70, GTX),
]

fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=130)
for (r, c, bg, name, sym, val, tx) in cells:
    x, y = c, 1 - r
    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=bg, edgecolor="#cfd6de", lw=1.5))
    ax.text(x + 0.5, y + 0.66, f"{name} ({sym})", ha="center", va="center",
            fontsize=13, fontweight="bold", color=tx)
    ax.text(x + 0.5, y + 0.30, f"{val}명", ha="center", va="center",
            fontsize=15, color="#1f2937")

# 축 라벨
ax.text(1.0, 2.34, "예측 (Prediction)", ha="center", fontsize=13, fontweight="bold", color="#1f2937")
ax.text(0.5, 2.08, "양성 (P)", ha="center", fontsize=11.5, color="#374151")
ax.text(1.5, 2.08, "음성 (N)", ha="center", fontsize=11.5, color="#374151")
ax.text(-0.42, 1.0, "실제 (Actual)", ha="center", va="center", rotation=90,
        fontsize=13, fontweight="bold", color="#1f2937")
ax.text(-0.12, 1.5, "양성 (P)", ha="center", va="center", rotation=90, fontsize=11.5, color="#374151")
ax.text(-0.12, 0.5, "음성 (N)", ha="center", va="center", rotation=90, fontsize=11.5, color="#374151")

ax.text(1.0, -0.35, "예시: 100명 중 실제 양성 20명 · 음성 80명",
        ha="center", fontsize=10.5, color="#6b7280")

ax.set_xlim(-0.6, 2.1); ax.set_ylim(-0.6, 2.5)
ax.set_aspect("equal"); ax.axis("off")
fig.savefig("ch2_confusion.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved: ch2_confusion.png")
