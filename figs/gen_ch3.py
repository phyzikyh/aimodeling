# -*- coding: utf-8 -*-
"""3장(RNN·LSTM·GRU) 개념 그림."""
import numpy as np
import matplotlib.pyplot as plt
from _style import save, C_BLUE, C_RED, C_GREEN, C_PURPLE, C_ORANGE, C_GRAY

rng = np.random.default_rng(3)


# 3.1 시계열 구성요소 분해
def fig_ts_decomp():
    t = np.arange(0, 96)
    trend = 0.06 * t + 2
    seasonal = 2.2 * np.sin(2*np.pi*t/12)
    noise = 0.6 * rng.standard_normal(len(t))
    obs = trend + seasonal + noise
    fig, axes = plt.subplots(4, 1, figsize=(7.6, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(t, obs, color=C_BLUE, lw=1.6); axes[0].set_ylabel("관측")
    axes[0].set_title("시계열 = 추세 + 계절성 + 잡음")
    axes[1].plot(t, trend, color=C_RED, lw=2); axes[1].set_ylabel("추세")
    axes[2].plot(t, seasonal, color=C_GREEN, lw=1.8); axes[2].set_ylabel("계절성")
    axes[3].plot(t, noise, color=C_GRAY, lw=1.2); axes[3].set_ylabel("잡음")
    axes[3].set_xlabel("시간 $t$")
    save(fig, "ch3_ts_decomp.png")


# 3.2 기울기 소실/폭발 (거리별 기울기 크기)
def fig_vanishing():
    k = np.arange(0, 31)
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    ax.plot(k, 0.8**k, color=C_BLUE, lw=2.2, marker="o", ms=3, label="스펙트럼 반경 ρ=0.8 (소실)")
    ax.plot(k, 1.0**k, color=C_GREEN, lw=2.2, marker="^", ms=3, label="ρ=1.0 (안정)")
    ax.plot(k, 1.1**k, color=C_RED, lw=2.2, marker="s", ms=3, label="ρ=1.1 (폭발)")
    ax.set_yscale("log")
    ax.set_xlabel("시점 간 거리 $k$ (역전파 단계 수)")
    ax.set_ylabel("기울기 크기 (로그)")
    ax.set_title("순환 신경망의 기울기 소실·폭발")
    ax.legend(fontsize=9, loc="center left")
    save(fig, "ch3_vanishing.png")


# 3.3 사인파 시계열과 슬라이딩 윈도 예측
def fig_ts_forecast():
    t = np.linspace(0, 24, 260)
    series = np.sin(t) + 0.5*np.sin(3*t) + 0.08*rng.standard_normal(len(t))
    win_s, win_e = 150, 190       # 입력 윈도 인덱스
    fig, ax = plt.subplots(figsize=(8.4, 3.6), constrained_layout=True)
    ax.plot(t[:win_e+1], series[:win_e+1], color=C_GRAY, lw=1.4, label="과거 관측")
    ax.plot(t[win_s:win_e], series[win_s:win_e], color=C_BLUE, lw=2.6, label="입력 윈도")
    ax.plot(t[win_e:win_e+30], series[win_e:win_e+30], color=C_RED, lw=2.2, ls="--", label="예측 대상(미래)")
    ax.axvspan(t[win_s], t[win_e], color=C_BLUE, alpha=0.07)
    ax.scatter([t[win_e]], [series[win_e]], color=C_RED, s=40, zorder=5)
    ax.set_xlabel("시간 $t$"); ax.set_ylabel("값")
    ax.set_title("슬라이딩 윈도 기반 시계열 예측")
    ax.legend(fontsize=9, loc="lower left", ncol=3)
    save(fig, "ch3_ts_forecast.png")


if __name__ == "__main__":
    fig_ts_decomp()
    fig_vanishing()
    fig_ts_forecast()
    print("done ch3")
