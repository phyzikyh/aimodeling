# -*- coding: utf-8 -*-
"""2장 초기화: 깊은 ReLU 망에서 초기화별 활성값 RMS의 층별 변화.
large/small/xavier/he 4가지. NumPy 시뮬레이션(순전파 직후, 학습 없음)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for cand in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"   # 로그축 음수 지수 마이너스 정상 표기

rng = np.random.default_rng(0)
n, depth, batch = 256, 30, 1024

def std_of(method):
    if method == "large":  return 1.0
    if method == "small":  return 0.02
    if method == "xavier": return np.sqrt(2.0 / (n + n))   # Var=2/(nin+nout)
    if method == "he":     return np.sqrt(2.0 / n)          # Var=2/nin

styles = {
    "large":  ("#E8402E", "분산 과다 (std=1.0)"),
    "small":  ("#8b5cf6", "분산 과소 (std=0.02)"),
    "xavier": ("#2c7be5", "Xavier (gain=1)"),
    "he":     ("#1AB18B", "He (ReLU)"),
}

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=130)
x0 = rng.standard_normal((batch, n))
for method, (col, lab) in styles.items():
    s = std_of(method)
    x = x0.copy()
    rms = [np.sqrt(np.mean(x**2))]
    for _ in range(depth):
        W = rng.standard_normal((n, n)) * s
        x = np.maximum(x @ W, 0.0)          # 선형층 + ReLU (편향 0)
        rms.append(np.sqrt(np.mean(x**2)))
    ax.plot(range(depth + 1), rms, marker="o", ms=3, color=col, lw=1.8, label=lab)

ax.set_yscale("log")
from matplotlib.ticker import LogLocator, FuncFormatter
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda y, _: "" if y <= 0 else f"1e{int(round(np.log10(y)))}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ""))
ax.set_title("가중치 초기화에 따른 신호 전파 (ReLU 심층망)", fontsize=12.5, fontweight="bold")
ax.set_xlabel("층 깊이"); ax.set_ylabel("활성값 RMS (로그 척도)")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=9.5, framealpha=0.92)
plt.tight_layout()
fig.savefig("ch2_init.png", dpi=150, bbox_inches="tight", facecolor="white")
last = {m: np.sqrt(np.mean((lambda: None)() or 0) if False else 0) for m in styles}
print("saved: ch2_init.png")
