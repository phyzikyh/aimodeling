# -*- coding: utf-8 -*-
"""그림 1.1 GPU 계산 스택의 계층 구조 생성 스크립트.
제목과 첫 박스가 겹치지 않도록 상단 여백을 확보한다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm

# 한글 폰트
for cand in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# 색상 (torch=cuDNN 초록 계열, CUDA/드라이버 파랑 계열, 하드웨어 주황)
LAYERS = [
    # (제목, 부제, 면색, 테두리색, 좌측바색)
    ("파이토치 (torch)", "사용자 코드 · 텐서·자동미분·신경망", "#e7f7ef", "#34c98a", "#1faf76"),
    ("cuDNN",            "딥러닝 연산 최적화 라이브러리",       "#e7f7ef", "#34c98a", "#1faf76"),
    ("CUDA Toolkit",     "GPU 병렬 연산 API·컴파일러",         "#e9edfb", "#8aa0ee", "#5f78e0"),
    ("NVIDIA 드라이버",  "운영체제 ↔ GPU 통신",               "#e9edfb", "#8aa0ee", "#5f78e0"),
    ("NVIDIA GPU 하드웨어", "실제 연산을 수행하는 물리 장치",   "#fdf1dc", "#f2b23e", "#e8971a"),
]

TITLE_TXT = "#3f4a54"
SUB_TXT   = "#8a97a3"

fig, ax = plt.subplots(figsize=(10, 7.2), dpi=110)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ── 제목 (상단, 박스 영역과 분리) ──
ax.add_patch(Rectangle((4.5, 92.0), 1.4, 5.0, facecolor="#1faf76",
                       edgecolor="none", zorder=5))
ax.text(7.5, 94.5, "GPU 계산 스택의 계층 구조", fontsize=21, fontweight="bold",
        color="#2f3a44", va="center", ha="left", zorder=5)

# ── 박스 레이아웃 ──
box_x, box_w = 18.0, 64.0
top, bottom = 84.0, 6.0          # 제목(92~) 아래에서 시작 → 겹침 없음
n = len(LAYERS)
gap = 2.6
box_h = (top - bottom - gap * (n - 1)) / n

for i, (title, sub, face, edge, bar) in enumerate(LAYERS):
    y = top - box_h - i * (box_h + gap)
    ax.add_patch(FancyBboxPatch(
        (box_x, y), box_w, box_h,
        boxstyle="round,pad=0,rounding_size=1.6",
        linewidth=1.6, facecolor=face, edgecolor=edge, zorder=2))
    # 좌측 강조 바
    ax.add_patch(Rectangle((box_x + 1.6, y + box_h * 0.16), 1.1, box_h * 0.68,
                           facecolor=bar, edgecolor="none", zorder=3))
    ax.text(box_x + 5.2, y + box_h * 0.62, title, fontsize=15.5,
            fontweight="bold", color=TITLE_TXT, va="center", ha="left", zorder=4)
    ax.text(box_x + 5.2, y + box_h * 0.28, sub, fontsize=11.5,
            color=SUB_TXT, va="center", ha="left", zorder=4)

# ── 좌측 세로 화살표 (사용자 코드 ↔ 하드웨어) ──
ax.annotate("", xy=(11.0, bottom + 1.0), xytext=(11.0, top),
            arrowprops=dict(arrowstyle="-|>", color="#b7c0c9", lw=2.2))
ax.text(11.0, top + 2.0, "사용자 코드", fontsize=11, color=SUB_TXT,
        va="bottom", ha="center")
ax.text(11.0, bottom - 1.4, "하드웨어", fontsize=11, color=SUB_TXT,
        va="top", ha="center")

# ── 우측 주석 (첫 박스 오른쪽, 겹치지 않게 여백 확보) ──
ax.text(box_x + box_w + 2.0, top - box_h * 0.5, "각 층은 바로\n아래 층을 호출",
        fontsize=12, fontweight="bold", color="#1faf76",
        va="center", ha="left", linespacing=1.3)

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
out = "ch1_gpu_stack_ours.png"
fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.12,
            facecolor="white")
print("saved:", out)
