# -*- coding: utf-8 -*-
"""2장 배치 정규화: BN vs LN 정규화 축 비교.
(B, D) 격자에서 BN은 '특성별로 배치에 걸쳐'(열), LN은 '샘플별로 특성에 걸쳐'(행) 정규화."""
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

B, D = 4, 5   # 샘플(배치) 4, 특성 5
TEAL, AMBER = "#1AB18B", "#F08C1D"

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), dpi=130)

def draw_grid(ax, mode, title, hl_color):
    for r in range(B):
        for c in range(D):
            # 정규화 그룹 강조: BN=열 c==1, LN=행 r==1
            hot = (mode == "BN" and c == 1) or (mode == "LN" and r == 1)
            ax.add_patch(Rectangle((c, B - 1 - r), 1, 1,
                         facecolor=(hl_color if hot else "#eef2f7"),
                         edgecolor="#b8c0cb", lw=1.2, alpha=(0.9 if hot else 1)))
    # 축 라벨
    ax.text(D / 2, B + 0.45, "특성 (D)", ha="center", fontsize=11.5, color="#374151")
    ax.text(-0.7, B / 2, "샘플 (배치 B)", va="center", rotation=90, fontsize=11.5, color="#374151")
    for c in range(D):
        ax.text(c + 0.5, B + 0.05, f"f{c+1}", ha="center", fontsize=9, color="#6b7280")
    for r in range(B):
        ax.text(-0.12, B - 1 - r + 0.5, f"s{r+1}", ha="right", va="center", fontsize=9, color="#6b7280")
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#1f2937", pad=8)
    ax.set_xlim(-0.9, D + 0.2); ax.set_ylim(-0.7, B + 0.9)
    ax.set_aspect("equal"); ax.axis("off")

draw_grid(axes[0], "BN", "배치 정규화 (BN)", TEAL)
axes[0].text(D / 2, -0.45, "특성마다 배치 전체에 걸쳐 정규화 (열)", ha="center", fontsize=10, color=TEAL)
draw_grid(axes[1], "LN", "층 정규화 (LN)", AMBER)
axes[1].text(D / 2, -0.45, "샘플마다 특성 전체에 걸쳐 정규화 (행)", ha="center", fontsize=10, color=AMBER)

plt.tight_layout()
fig.savefig("ch2_batchnorm.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved: ch2_batchnorm.png")
