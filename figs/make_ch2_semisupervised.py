# -*- coding: utf-8 -*-
"""2장 준지도학습 절: 소량 라벨 + 다량 비라벨.
(a) 라벨만 사용(지도) vs (b) 비라벨 구조 활용(준지도) 결정경계 비교."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading
from sklearn.neighbors import KNeighborsClassifier

for cand in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# 데이터: 두 반달. 소수만 라벨, 나머지는 비라벨(-1)
X, y = make_moons(n_samples=260, noise=0.08, random_state=3)
rng = np.random.default_rng(1)
labeled = []
for c in (0, 1):                       # 클래스마다 3개씩만 라벨
    idx = np.where(y == c)[0]
    labeled += list(rng.choice(idx, 3, replace=False))
labeled = np.array(labeled)
y_semi = np.full_like(y, -1)           # -1 = 비라벨
y_semi[labeled] = y[labeled]

# (a) 지도학습: 라벨된 소수만으로 로지스틱 회귀
clf_sup = LogisticRegression().fit(X[labeled], y[labeled])
# (b) 준지도: 라벨전파로 비라벨에 유사라벨 부여 → 전체로 매끄러운 경계 학습
ls = LabelSpreading(kernel="knn", n_neighbors=10).fit(X, y_semi)
clf_semi = KNeighborsClassifier(15).fit(X, ls.transduction_)

# 그리드
gx, gy = np.meshgrid(np.linspace(X[:,0].min()-0.4, X[:,0].max()+0.4, 300),
                     np.linspace(X[:,1].min()-0.4, X[:,1].max()+0.4, 300))
grid = np.c_[gx.ravel(), gy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=130)
BG = ["#eaf3fb", "#fdeeea"]; PT = ["#2c7be5", "#E8402E"]

def draw(ax, Z, title, acc):
    ax.contourf(gx, gy, Z.reshape(gx.shape), levels=[-.5,.5,1.5], colors=BG, alpha=.9)
    ax.contour(gx, gy, Z.reshape(gx.shape), levels=[.5], colors="#333", linewidths=1.4)
    # 비라벨(회색 작은 점)
    un = np.setdiff1d(np.arange(len(X)), labeled)
    ax.scatter(X[un,0], X[un,1], s=10, c="#c2c8d0", edgecolors="none", label="비라벨", zorder=2)
    # 라벨(큰 테두리 점)
    for c in (0,1):
        m = labeled[y[labeled]==c]
        ax.scatter(X[m,0], X[m,1], s=140, c=PT[c], edgecolors="black", linewidths=1.6,
                   marker="*", zorder=4, label=f"라벨 클래스 {c}")
    ax.set_title(f"{title}\n(전체 정확도 {acc*100:.0f}%)", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

acc_sup = (clf_sup.predict(X) == y).mean()
acc_ls  = (clf_semi.predict(X) == y).mean()
nlab, nun = len(labeled), len(X) - len(labeled)
draw(axes[0], clf_sup.predict(grid), f"(a) 라벨 {nlab}개만 사용 (지도학습)", acc_sup)
draw(axes[1], clf_semi.predict(grid), f"(b) 비라벨 {nun}개 구조 활용 (준지도학습)", acc_ls)
axes[1].legend(loc="upper right", fontsize=8, framealpha=.9)

plt.tight_layout()
fig.savefig("ch2_semisupervised.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved. sup acc={acc_sup:.3f}  semi acc={acc_ls:.3f}  labeled={len(labeled)}")
