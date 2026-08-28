# -*- coding: utf-8 -*-
"""3장 우리-스타일 그림: 양방향 LSTM(BiLSTM)."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _ours import new_ax, save, heading, chip, BRAND, CORAL, INDIGO, INK, SUB, TINT

SUB_T = {1: "₁", 2: "₂", 3: "₃"}


def nb(ax, cx, cy, text, edge, tint, w=1.15, h=0.72, fs=12):
    ax.add_patch(FancyBboxPatch((cx+0.05-w/2, cy-0.08-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.1", fc="#dee2e7", ec="none", zorder=2))
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.1", fc=tint, ec=edge, lw=1.7, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold",
            color="#28303a", zorder=4)


def ar(ax, p0, p1, color, lw=2.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=lw,
                 color=color, zorder=2.5, shrinkA=3, shrinkB=3,
                 connectionstyle=f"arc3,rad={rad}"))


def fig_bilstm_ours():
    fig, ax = new_ax(9.8, 6.4, (0, 12.4), (0, 9))
    heading(ax, 0.4, 8.4, "양방향 LSTM (BiLSTM)")
    ax.text(0.75, 7.65, "순방향과 역방향 두 LSTM이 같은 입력을 읽고, 각 시점의 두 은닉 상태를 이어 붙입니다.",
            fontsize=11, color=SUB, va="center")

    cols = [3.1, 6.9, 10.7]
    y_x, y_f, y_b, y_o = 1.1, 3.0, 5.0, 6.9

    # 세로 층 라벨
    ax.text(0.75, y_f, "순방향\nLSTM →", fontsize=10, color=BRAND, fontweight="bold", ha="center", va="center")
    ax.text(0.75, y_b, "역방향\nLSTM ←", fontsize=10, color=INDIGO, fontweight="bold", ha="center", va="center")
    ax.text(0.75, y_o, "출력\n(concat)", fontsize=10, color=CORAL, fontweight="bold", ha="center", va="center")

    for k, cx in enumerate(cols, 1):
        s = SUB_T[k]
        nb(ax, cx, y_x, f"x{s}", "#9098a4", TINT["gray"])
        nb(ax, cx, y_f, f"→h{s}", BRAND, TINT["brand"])
        nb(ax, cx, y_b, f"←h{s}", INDIGO, TINT["indigo"])
        nb(ax, cx, y_o, f"h{s}=[→h{s};←h{s}]", CORAL, TINT["coral"], w=2.5, fs=10.5)
        # x_t -> 순방향(수직), x_t -> 역방향(오른쪽으로 우회하여 순방향 통과 안 함)
        ar(ax, (cx, y_x+0.36), (cx, y_f-0.36), BRAND)
        ar(ax, (cx+0.62, y_x+0.2), (cx+0.62, y_b-0.36), INDIGO)
        ar(ax, (cx+0.62, y_b-0.36), (cx+0.2, y_b-0.36), INDIGO)
        # 순방향/역방향 -> 출력
        ar(ax, (cx-0.2, y_f+0.36), (cx-0.35, y_o-0.36), BRAND, rad=0.05)
        ar(ax, (cx+0.2, y_b+0.36), (cx+0.35, y_o-0.36), INDIGO, rad=-0.05)

    # 순방향 체인 (왼→오)
    for i in range(len(cols)-1):
        ar(ax, (cols[i]+0.58, y_f), (cols[i+1]-0.58, y_f), BRAND, lw=2.4)
    # 역방향 체인 (오→왼)
    for i in range(len(cols)-1, 0, -1):
        ar(ax, (cols[i]-0.58, y_b), (cols[i-1]+0.58, y_b), INDIGO, lw=2.4)

    save(fig, "ch3_bilstm_ours.png")


def fig_rnn_unroll_ours():
    fig, ax = new_ax(9.8, 5.6, (0, 12.4), (0, 8))
    heading(ax, 0.4, 7.5, "순환 신경망의 펼침 (unroll)")
    ax.text(0.75, 6.8, "같은 셀이 시점마다 반복됩니다. 은닉 상태 h가 과거 정보를 다음 시점으로 나릅니다.",
            fontsize=11, color=SUB, va="center")
    cols = [3.3, 7.0, 10.7]
    y_x, y_h, y_y = 1.2, 3.5, 5.9
    # 초기 은닉 상태 h0
    nb(ax, 1.2, y_h, "h0", "#9098a4", TINT["gray"], w=0.95, fs=12)
    ar(ax, (1.7, y_h), (cols[0]-0.6, y_h), BRAND, lw=2.4)
    for k, cx in enumerate(cols, 1):
        s = SUB_T[k]
        nb(ax, cx, y_x, f"x{s}", "#9098a4", TINT["gray"])
        nb(ax, cx, y_h, f"h{s}", BRAND, TINT["brand"])
        nb(ax, cx, y_y, f"y{s}", CORAL, TINT["coral"])
        ar(ax, (cx, y_x+0.36), (cx, y_h-0.36), BRAND)      # x→h
        ar(ax, (cx, y_h+0.36), (cx, y_y-0.36), CORAL)      # h→y
        if k < 3:
            ar(ax, (cx+0.6, y_h), (cols[k]-0.6, y_h), BRAND, lw=2.4)  # h_{t}→h_{t+1}
    save(fig, "ch3_rnn_unroll_ours.png")


if __name__ == "__main__":
    fig_bilstm_ours()
    fig_rnn_unroll_ours()
    print("done ours ch3")
