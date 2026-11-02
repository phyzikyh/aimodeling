# -*- coding: utf-8 -*-
from _ours import *
fig, ax = new_ax(9, 3.8, (0, 18), (0, 7.6))
s = 5.0  # 정사각형 변
def box(x, y, w, h, fc, ec, label, sub):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=2, zorder=3))
    ax.text(x+w/2, y+h+0.45, label, ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
    return
y0 = 1.8
box(0.4, y0, s, s, TINT["gray"], SUB, "ΔW", "")
ax.text(0.4+s/2, y0+s/2+0.4, "d × k", ha="center", fontsize=15, color=INK, fontweight="bold")
ax.text(0.4+s/2, y0+s/2-0.5, "2048 × 2048", ha="center", fontsize=12, color=SUB)
ax.text(0.4+s/2, y0-0.6, "원소 4,194,304개", ha="center", fontsize=11.5, color=INK)
ax.text(6.35, y0+s/2, "=", fontsize=26, ha="center", va="center", color=INK)
bw = 0.55
box(7.3, y0, bw, s, "#d6f1e8", BRAND, "B", "")
ax.text(7.3+bw/2, y0-0.6, "2048 × r", ha="center", fontsize=11.5, color=INK)
ax.text(8.9, y0+s/2, "×", fontsize=22, ha="center", va="center", color=INK)
ah = 0.55
box(9.7, y0+s/2-ah/2, 7.0, ah, "#dfe6fd", INDIGO, "A", "")
ax.text(9.7+3.5, y0+s/2-ah/2-0.6, "r × 2048", ha="center", fontsize=11.5, color=INK)
ax.text(9.7+3.5, y0+s/2+ah/2+1.0, "r = 16  ← 가운데(중간) 차원", ha="center", fontsize=12, color=INDIGO, fontweight="bold")
ax.text(13.2, 0.45, "학습 대상 2048×16 + 16×2048 = 65,536개 (전체의 1.56%)", ha="center", fontsize=12, color=INK)
ax.text(13.2, 0.0, "※ 실제 크기 비율이 아닌 행렬의 형태", ha="center", fontsize=9.5, color=SUB)
save(fig, "ch10_lora_rank.png")
