# -*- coding: utf-8 -*-
"""4장 우리-스타일 그림: 시계열 트랜스포머 계보 · 쿼리·키·값 어텐션."""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from _ours import new_ax, save, heading, BRAND, CORAL, INDIGO, AMBER, INK, SUB, TINT, LINE

TINT_OF = {INDIGO: TINT["indigo"], AMBER: TINT["amber"], BRAND: TINT["brand"], CORAL: TINT["coral"]}


def branchcard(ax, cx, y, title, members, color, w=3.15, h=2.9):
    ax.add_patch(FancyBboxPatch((cx-w/2+0.06, y-0.09), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", fc="#dfe3e8", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((cx-w/2, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.14", fc="white", ec="#e4e8ec", lw=1.2, zorder=4))
    tw = 0.285*len(title) + 0.55
    ax.add_patch(FancyBboxPatch((cx-tw/2, y+h-0.28), tw, 0.5,
                 boxstyle="round,pad=0.02,rounding_size=0.24", fc=color, ec="none", zorder=5))
    ax.text(cx, y+h-0.02, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="white", zorder=6)
    for i, (nm, yr) in enumerate(members):
        yy = y+h-0.95-i*0.62
        ax.add_patch(FancyBboxPatch((cx-w/2+0.25, yy-0.24), w-0.5, 0.48,
                     boxstyle="round,pad=0.02,rounding_size=0.12", fc=TINT["gray"], ec=color, lw=1.3, zorder=5))
        ax.text(cx-w/2+0.45, yy, nm, ha="left", va="center", fontsize=11,
                fontweight="bold", color=INK, zorder=6)
        ax.text(cx+w/2-0.42, yy, yr, ha="right", va="center", fontsize=9.5,
                color=SUB, zorder=6)


def ar(ax, p0, p1, color, lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=lw,
                 color=color, zorder=2, shrinkA=3, shrinkB=3,
                 connectionstyle=f"arc3,rad={rad}"))


def fig_ts_transformer_lineage():
    fig, ax = new_ax(11.4, 6.6, (0, 16), (0, 9.4))
    heading(ax, 0.4, 8.9, "시계열 트랜스포머의 계보")
    ax.text(0.8, 8.25, "표준 트랜스포머의 이차 복잡도와 시계열 특성을 겨냥해 네 방향으로 변형이 발전했습니다.",
            fontsize=11, color=SUB, va="center")

    # 루트
    rx, ry, rw, rh = 5.3, 7.0, 5.4, 0.98
    ax.add_patch(FancyBboxPatch((rx+0.06, ry-0.09), rw, rh,
                 boxstyle="round,pad=0.02,rounding_size=0.2", fc="#dfe3e8", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh,
                 boxstyle="round,pad=0.02,rounding_size=0.2", fc=INK, ec="none", zorder=4))
    ax.text(rx+rw/2, ry+rh/2+0.12, "Attention is All You Need", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="white", zorder=5)
    ax.text(rx+rw/2, ry+rh/2-0.24, "Vaswani et al., 2017", ha="center", va="center",
            fontsize=9.5, color="#c9ccd2", zorder=5)

    xs = [2.15, 6.05, 9.95, 13.85]
    colors = [BRAND, INDIGO, AMBER, CORAL]
    branches = [
        ("효율적 어텐션", [("LogSparse", "2019"), ("Informer", "2021"), ("Pyraformer", "2022")]),
        ("시계열 분해 결합", [("Autoformer", "2021"), ("FEDformer", "2022")]),
        ("표현·패치화", [("PatchTST", "2023")]),
        ("해석·불확실성", [("TFT", "2021")]),
    ]
    for (t, ms), cx, c in zip(branches, xs, colors):
        branchcard(ax, cx, 1.2, t, ms, c)
        ar(ax, (rx+rw/2, ry), (cx, 4.15), c, rad=(0.0 if abs(cx-8)<3 else 0.06*(1 if cx>8 else -1)))

    save(fig, "ch4_ts_transformer_ours.png")


def _box(ax, cx, cy, text, edge, tint, w=1.5, h=0.78, fs=13):
    ax.add_patch(FancyBboxPatch((cx+0.05-w/2, cy-0.07-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc="#dee2e7", ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc=tint, ec=edge, lw=1.9, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold", color="#28303a", zorder=5)


def fig_qkv_ours():
    """쿼리·키·값의 작동: 하나의 쿼리가 모든 키와 비교→가중치→값의 가중합."""
    fig, ax = new_ax(11.5, 6.8, (0, 21), (0, 12.4))
    heading(ax, 0.4, 11.7, "쿼리·키·값 어텐션의 작동")
    ax.text(0.78, 11.0, "위치 2의 쿼리가 모든 키와 유사도를 재고, 그 가중치로 값을 모아 새 표현을 만듭니다.",
            fontsize=10.5, color=SUB, va="center")

    rows = [7.7, 5.95, 4.2, 2.45]
    alpha = [0.10, 0.60, 0.06, 0.24]
    zc = (18.4, 5.05)

    # 열 머리
    ax.text(5.7, 9.95, "키 $k_i$ · 값 $v_i$", fontsize=10.5, color=INK, ha="center", fontweight="bold")
    ax.text(12.1, 9.95, "어텐션 가중치 $\\alpha_i$", fontsize=10.5, color=INK, ha="center", fontweight="bold")
    ax.text(18.4, 9.95, "출력", fontsize=10.5, color=CORAL, ha="center", fontweight="bold")

    # 쿼리 (상단)
    _box(ax, 2.4, 9.3, r"$q_2$", INDIGO, TINT["indigo"], w=1.5, h=0.72, fs=13)
    ax.text(2.4, 8.65, "위치 2의 쿼리", fontsize=9, color=INDIGO, ha="center", va="center")

    for i, (y, a) in enumerate(zip(rows, alpha), 1):
        _box(ax, 2.4, y, rf"$x_{i}$", "#9098a4", TINT["gray"], w=1.4, h=0.72)
        _box(ax, 5.7, y+0.42, rf"$k_{i}$", AMBER, TINT["amber"], w=1.25, h=0.56, fs=11)
        _box(ax, 5.7, y-0.42, rf"$v_{i}$", BRAND, TINT["brand"], w=1.25, h=0.56, fs=11)
        ar(ax, (3.1, y+0.12), (5.05, y+0.42), AMBER, lw=1.7)
        ar(ax, (3.1, y-0.12), (5.05, y-0.42), BRAND, lw=1.7)
        # 쿼리→키 비교(점선)
        ax.add_patch(FancyArrowPatch((2.95, 8.98), (5.05, y+0.42), arrowstyle="-", lw=1.0,
                     color=LINE, zorder=1, linestyle=(0, (3, 3)), connectionstyle="arc3,rad=-0.1"))
        # 가중치 막대
        bx0, bmax = 9.5, 5.0
        ax.add_patch(Rectangle((bx0, y-0.2), bmax, 0.4, fc="#eef1f4", ec="none", zorder=2))
        ax.add_patch(Rectangle((bx0, y-0.2), bmax*a, 0.4, fc=INDIGO, ec="none", zorder=3))
        ax.text(bx0+bmax+0.35, y, f"{a:.2f}", fontsize=10.5, color=INDIGO, va="center", fontweight="bold")
        # 값→출력 (굵기 ∝ α)
        ar(ax, (6.35, y-0.42), (zc[0]-0.98, zc[1]), BRAND, lw=1.0+a*7.5,
           rad=(0.10 if y > zc[1] else -0.10))

    _box(ax, zc[0], zc[1], r"$z_2$", CORAL, TINT["coral"], w=1.9, h=1.0, fs=14)
    ax.text(zc[0], zc[1]-0.95, r"$=\sum_i \alpha_i v_i$", fontsize=11, color=SUB, ha="center", va="center")

    # 범례
    for j, (t, c) in enumerate([("$q$ 질문", INDIGO), ("$k$ 색인", AMBER),
                                ("$v$ 내용", BRAND), ("$z$ 결과", CORAL)]):
        x = 6.2 + j*3.2
        ax.add_patch(FancyBboxPatch((x-0.28, 0.7), 0.5, 0.42, boxstyle="round,pad=0.02,rounding_size=0.1",
                     fc=TINT_OF[c], ec=c, lw=1.6, zorder=4))
        ax.text(x+0.42, 0.91, t, fontsize=9.5, color=SUB, va="center", ha="left")
    save(fig, "ch4_qkv_ours.png")


MATSH = {"x": (INDIGO, TINT["indigo"]), "q": (AMBER, TINT["amber"]),
         "k": (CORAL, TINT["coral"]), "v": (BRAND, TINT["brand"]),
         "score": ("#b0761f", "#fdf1d8"), "a": ("#b0761f", "#fdf1d8"), "z": (CORAL, TINT["coral"])}


def mat(ax, cx, cy, w, h, kind, gr=(4, 6), name=None, shape=None, rowlabels=None, fs_name=12):
    edge, tint = MATSH[kind]
    ax.add_patch(FancyBboxPatch((cx-w/2+0.05, cy-h/2-0.07), w, h,
                 boxstyle="round,pad=0.01,rounding_size=0.06", fc="#dee2e7", ec="none", zorder=2))
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.01,rounding_size=0.06", fc=tint, ec=edge, lw=2.0, zorder=3))
    gr_r, gr_c = gr
    for i in range(1, gr_r):
        yy = cy-h/2 + h*i/gr_r
        ax.plot([cx-w/2+0.07, cx+w/2-0.07], [yy, yy], color=edge, lw=0.7, alpha=0.35, zorder=4)
    for j in range(1, gr_c):
        xx = cx-w/2 + w*j/gr_c
        ax.plot([xx, xx], [cy-h/2+0.07, cy+h/2-0.07], color=edge, lw=0.7, alpha=0.35, zorder=4)
    if name:
        ax.text(cx, cy+h/2+0.3, name, ha="center", va="bottom", fontsize=fs_name,
                fontweight="bold", color=edge, zorder=5)
    if shape:
        ax.text(cx, cy-h/2-0.28, shape, ha="center", va="top", fontsize=8.6, color=SUB, zorder=5)
    if rowlabels:
        for i, lb in enumerate(rowlabels):
            yy = cy+h/2 - h*(i+0.5)/len(rowlabels)
            ax.text(cx-w/2-0.16, yy, lb, ha="right", va="center", fontsize=8.5,
                    color=SUB, style="italic", zorder=5)


def op(ax, x, y, s, fs=19):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold", zorder=5)


def fig_qkv_matrix_ours():
    """문장 행렬 X에 W^Q/W^K/W^V를 곱해 Q/K/V 행렬을 만드는 과정."""
    fig, ax = new_ax(10.5, 8.6, (0, 15), (0, 13))
    heading(ax, 0.4, 12.5, "Q·K·V 행렬 생성")
    ax.text(0.78, 11.8, "문장 행렬 X에 세 가중치 행렬을 곱해 쿼리·키·값 행렬을 한 번에 만듭니다.",
            fontsize=10.5, color=SUB, va="center")

    words = ["I", "am", "a", "student"]
    mat(ax, 2.7, 6.0, 2.4, 3.2, "x", gr=(4, 8), name="X (입력 문장)",
        shape="(4 × 512)\nseq_len × d_model", rowlabels=words)

    lanes = [(9.5, "q", r"$W^Q$", "Q"), (6.0, "k", r"$W^K$", "K"), (2.5, "v", r"$W^V$", "V")]
    for y, kind, wname, rname in lanes:
        op(ax, 5.5, y, "×")
        mat(ax, 7.3, y, 1.5, 2.0, kind, gr=(8, 4), name=wname, shape="(512 × 64)")
        op(ax, 9.4, y, "=")
        mat(ax, 11.6, y, 1.35, 1.9, kind, gr=(4, 4), name=rname,
            shape="(4 × 64)", rowlabels=words)
        ax.add_patch(FancyArrowPatch((3.95, 6.0), (6.55, y), arrowstyle="-", lw=1.0,
                     color=LINE, zorder=1, linestyle=(0, (3, 3)), connectionstyle="arc3,rad=0.0"))
    ax.text(7.5, 0.55, "W: (d_model × d_k) = (512 × 64),   Q·K·V: (seq_len × d_k) = (4 × 64)",
            fontsize=9, color=SUB, ha="center", va="center")
    save(fig, "ch4_qkv_matrix_ours.png")


def fig_attention_matrix_ours():
    """Q·K^T → 스케일·softmax → ·V = Z 의 행렬 연산."""
    fig, ax = new_ax(10.8, 6.2, (0, 15.5), (0, 9.0))
    heading(ax, 0.4, 8.4, "셀프 어텐션의 행렬 연산")
    ax.text(0.78, 7.7, "쿼리·키의 내적으로 스코어를 얻고, 스케일·소프트맥스 후 값과 곱해 결과를 만듭니다.",
            fontsize=10.5, color=SUB, va="center")

    words = ["I", "am", "a", "student"]
    y1, y2 = 5.4, 1.9
    # 1행: Q × K^T = Score
    mat(ax, 2.0, y1, 1.25, 1.9, "q", gr=(4, 4), name="Q", shape="(4 × 64)", rowlabels=words)
    op(ax, 3.35, y1, "×")
    mat(ax, 5.1, y1, 2.1, 1.25, "k", gr=(4, 8), name=r"$K^\top$", shape="(64 × 4)")
    op(ax, 6.9, y1, "=")
    mat(ax, 8.5, y1, 1.6, 1.6, "score", gr=(4, 4), name="Score", shape="(4 × 4)")
    # 2행: softmax(Score/√d_k) × V = Z
    mat(ax, 2.6, y2, 1.6, 1.6, "a", gr=(4, 4), name=r"softmax(Score / $\sqrt{d_k}$)",
        shape="(4 × 4)", fs_name=10)
    op(ax, 4.5, y2, "×")
    mat(ax, 6.1, y2, 1.25, 1.9, "v", gr=(4, 4), name="V", shape="(4 × 64)", rowlabels=words)
    op(ax, 7.5, y2, "=")
    mat(ax, 9.4, y2, 1.25, 1.9, "z", gr=(4, 4), name="Z (어텐션 값)", shape="(4 × 64)", rowlabels=words)
    # Score → softmax(A) 로 내려가는 화살표
    ax.add_patch(FancyArrowPatch((8.5, y1-0.85), (2.6, y2+0.9), arrowstyle="-|>", mutation_scale=14,
                 lw=2.0, color=INDIGO, zorder=2, shrinkA=4, shrinkB=4, connectionstyle="arc3,rad=0.28"))
    ax.text(5.0, 3.75, r"÷ $\sqrt{d_k}$ 후 softmax", fontsize=10, color=INDIGO,
            fontweight="bold", ha="center", va="center")
    save(fig, "ch4_attention_matrix_ours.png")


if __name__ == "__main__":
    fig_ts_transformer_lineage()
    fig_qkv_ours()
    fig_qkv_matrix_ours()
    fig_attention_matrix_ours()
    print("done ours ch4")
