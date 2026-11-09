# -*- coding: utf-8 -*-
"""12주차 슬라이드용 보충 그림: U-Net이 학습과 생성에서 하는 일, '이전 단계'의 의미."""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from _ours import new_ax, heading, BRAND, CORAL, INDIGO, AMBER, ROSE, INK, SUB, LINE, TINT
from gen_ours_ch9 import rb, tx, arr, blend, save
mpl.rcParams["mathtext.fontset"] = "dejavusans"
MONO = "#3a3f45"  # 색 없이 그리는 도식의 선 색


def fig_unet_role():
    fig, ax = new_ax(14.0, 8.6, (0, 28.0), (-2.0, 15.6))
    heading(ax, 0.4, 14.8, "U-Net은 학습할 때도, 생성할 때도 '노이즈'를 예측합니다")
    xs = [0.6, 6.2, 11.8, 17.4, 23.0]
    w = 4.6
    # U-Net 열 강조 띠
    ax.add_patch(Rectangle((11.5, 1.0), 5.2, 12.4, fc="#f1f3f5", ec="none", zorder=1))
    tx(ax, 14.1, 13.0, "같은 일: 노이즈 예측", fs=11, bold=True, color=INK)
    # 학습 줄
    tx(ax, 0.6, 11.7, "학습", fs=13, bold=True, color=INK, ha="left")
    tr = [("원본 $x_0$", "깨끗한 이미지", "#ffffff", MONO),
          ("노이즈 섞기", r"무작위 $t$, 실제 $\epsilon$", "#ffffff", MONO),
          ("U-Net", r"$x_t$, $t$ → 예측 $\hat{\epsilon}$", "#ffffff", MONO),
          ("정답과 비교", r"$\|\epsilon-\hat{\epsilon}\|^2$ (MSE)", "#ffffff", MONO),
          ("역전파", "U-Net 가중치 갱신", "#ffffff", MONO)]
    for k, (a, b, fc, ec) in enumerate(tr):
        rb(ax, xs[k], 7.6, w, 3.4, fc, ec, lw=2.0, shadow=True, z=4)
        tx(ax, xs[k] + w / 2, 9.8, a, fs=11.5, bold=True, z=6)
        tx(ax, xs[k] + w / 2, 8.5, b, fs=9.5, color=INK, z=6)
        if k < 4:
            arr(ax, (xs[k] + w + 0.05, 9.3), (xs[k + 1] - 0.05, 9.3), SUB, lw=2.2)
    # 생성 줄
    tx(ax, 0.6, 5.7, "생성", fs=13, bold=True, color=INK, ha="left")
    ge = [(r"순수 노이즈 $x_T$", "여기서 시작", "#ffffff", MONO),
          (r"현재 $x_t$ + 단계 $t$", "U-Net에 입력", "#ffffff", MONO),
          ("U-Net", r"→ 예측 $\hat{\epsilon}$", "#ffffff", MONO),
          ("역방향 식", r"계수 + 작은 무작위 $\sigma_t z$", "#ffffff", MONO),
          (r"이전 단계 $x_{t-1}$", "노이즈가 조금 줄어듦", "#ffffff", MONO)]
    for k, (a, b, fc, ec) in enumerate(ge):
        rb(ax, xs[k], 1.6, w, 3.4, fc, ec, lw=2.0, shadow=True, z=4)
        tx(ax, xs[k] + w / 2, 3.8, a, fs=11.5, bold=True, z=6)
        tx(ax, xs[k] + w / 2, 2.5, b, fs=9.5, color=INK, z=6)
        if k < 4:
            arr(ax, (xs[k] + w + 0.05, 3.3), (xs[k + 1] - 0.05, 3.3), SUB, lw=2.2)
    # 반복 화살표
    arr(ax, (xs[4] + w / 2, 1.5), (xs[1] + w / 2, 1.5), MONO, lw=2.0, rad=-0.12)
    tx(ax, 14.1, -1.2, r"$x_{t-1}$을 다시 입력으로 · $t=T,\ T-1,\ \cdots,\ 1$ 반복", fs=10, color=INK, bold=True)
    save(fig, "ch11_unet_role.png")


def fig_chain5():
    fig, ax = new_ax(14.0, 6.6, (0, 28.0), (0, 13.2))
    heading(ax, 0.4, 12.4, "'이전 단계'는 순방향에서 붙인 단계 번호를 기준으로 합니다 (예: 5단계)")
    xs = [0.8 + 4.6 * k for k in range(6)]
    w = 3.4
    labs = [("$x_0$", "원본"), ("$x_1$", "노이즈 조금"), ("$x_2$", "더 많이"), ("$x_3$", "더 많이"),
            ("$x_4$", "거의 노이즈"), ("$x_5$", "순수 노이즈")]
    cols = [MONO] * 6
    for x, (a, b), c in zip(xs, labs, cols):
        rb(ax, x, 4.8, w, 3.0, "#ffffff", c, lw=2.0, shadow=True, z=4)
        tx(ax, x + w / 2, 6.9, a, fs=14, bold=True, z=6)
        tx(ax, x + w / 2, 5.6, b, fs=9.5, color=INK, z=6)
    for k in range(5):
        arr(ax, (xs[k] + w + 0.05, 7.3), (xs[k + 1] - 0.05, 7.3), MONO, lw=2.2, rad=-0.35)
        arr(ax, (xs[k + 1] - 0.05, 5.3), (xs[k] + w + 0.05, 5.3), MONO, lw=2.2, rad=-0.35)
    tx(ax, 14.0, 9.3, "순방향: 노이즈를 더함 (규칙으로 정함, 학습하지 않음) →", fs=11, bold=True, color=INK)
    tx(ax, 14.0, 3.5, "← 역방향(생성): 노이즈를 덜어 냄 (U-Net이 예측한 노이즈를 이용)", fs=11, bold=True, color=INK)
    ax.add_patch(Rectangle((xs[4] - 0.3, 4.4), 4.6 + w + 0.6, 3.8, fc="none", ec=INK, lw=2.2, ls="--", zorder=5))
    tx(ax, 14.0, 1.5, r"현재가 $x_5$라면 노이즈를 한 단계 덜어 낸 $x_4$가 '이전 단계 이미지'입니다 ($t\rightarrow t-1$)", fs=11, color=INK)
    save(fig, "ch11_chain5.png")


if __name__ == "__main__":
    fig_unet_role(); fig_chain5()
