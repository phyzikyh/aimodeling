# -*- coding: utf-8 -*-
"""3장 우리-스타일 그림: 양방향 LSTM(BiLSTM)."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _ours import new_ax, save, heading, chip, card, BRAND, CORAL, INDIGO, INK, SUB, TINT

VIOLET = "#8b5cf6"  # 다대다 칩 — 은닉 초록과 구분되는 보라

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


def fig_rnn_io_types():
    """순환 신경망 입출력 4유형 — 범례를 카드 위로 충분히 띄워 겹침 제거."""
    fig, ax = new_ax(11.6, 5.4, (0, 23), (0, 10.6))
    heading(ax, 0.4, 9.7, "순환 신경망의 입출력 유형")

    # 범례: 카드(상단 8.2)보다 확실히 위(9.4~9.9)에 배치해 제목 칩과 겹치지 않게 함
    def swatch(x, edge, tint, label):
        ax.add_patch(FancyBboxPatch((x-0.33, 9.4), 0.66, 0.5,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc=tint, ec=edge, lw=1.8, zorder=6))
        ax.text(x+0.52, 9.65, label, fontsize=11, color=SUB, va="center", ha="left", zorder=6)
    swatch(13.2, "#9098a4", TINT["gray"], "입력")
    swatch(16.6, BRAND, TINT["brand"], "은닉")
    swatch(20.0, CORAL, TINT["coral"], "출력")

    cols = [3.0, 8.4, 13.8, 19.2]
    titles = ["일대일", "일대다", "다대일", "다대다"]
    tcolors = [INDIGO, CORAL, BRAND, VIOLET]
    captions = ["이미지 분류", "이미지 캡셔닝", "감성 분류·다음값 예측", "번역·시퀀스 라벨링"]
    cw, ch = 4.4, 7.4
    y0, y_x, y_h, y_y = 0.8, 2.3, 4.5, 6.7
    off4 = [-1.5, -0.5, 0.5, 1.5]
    specs = {0: {"in": [0.0], "hid": [0.0], "out": [0.0]},
             1: {"in": [-1.5], "hid": off4, "out": off4},
             2: {"in": off4, "hid": off4, "out": [1.5]},
             3: {"in": off4, "hid": off4, "out": off4}}

    def sq(cx, cy, edge, tint, w=0.86, h=0.62):
        ax.add_patch(FancyBboxPatch((cx+0.05-w/2, cy-0.07-h/2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12", fc="#dee2e7", ec="none", zorder=2))
        ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12", fc=tint, ec=edge, lw=1.8, zorder=3))

    for idx, cx in enumerate(cols):
        card(ax, cx-cw/2, y0, cw, ch, title=titles[idx], color=tcolors[idx])
        sp = specs[idx]
        hid = sorted(sp["hid"])
        for o in hid:
            sq(cx+o, y_h, BRAND, TINT["brand"])
        for a, b in zip(hid[:-1], hid[1:]):
            ar(ax, (cx+a+0.44, y_h), (cx+b-0.44, y_h), BRAND, lw=2.0)
        for o in sp["in"]:
            sq(cx+o, y_x, "#9098a4", TINT["gray"])
            ar(ax, (cx+o, y_x+0.33), (cx+o, y_h-0.33), "#9098a4", lw=1.8)
        for o in sp["out"]:
            sq(cx+o, y_y, CORAL, TINT["coral"])
            ar(ax, (cx+o, y_h+0.33), (cx+o, y_y-0.33), CORAL, lw=1.8)
        ax.text(cx, y0+0.72, captions[idx], fontsize=10.5, color=SUB, ha="center", va="center")
    save(fig, "ch3_rnn_io_types_ours.png")


if __name__ == "__main__":
    fig_bilstm_ours()
    fig_rnn_unroll_ours()
    fig_rnn_io_types()
    print("done ours ch3")
