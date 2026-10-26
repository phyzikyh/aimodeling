# -*- coding: utf-8 -*-
"""9·10주차(텍스트 임베딩·트랜스포머·BERT) 교재용 그림. 모두 직접 작성한 그림이다."""
import os, math
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from _ours import new_ax, heading, BRAND, CORAL, INDIGO, AMBER, ROSE, INK, SUB, LINE, TINT

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name, os.path.getsize(p))


def rb(ax, x, y, w, h, fc, ec, lw=1.8, r=0.14, z=3, shadow=False):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.05, y - 0.07), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}",
                     fc="#dde1e7", ec="none", zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, zorder=z))


def tx(ax, x, y, s, fs=11, color=INK, bold=False, ha="center", va="center", z=6, style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, fontweight="bold" if bold else "normal",
            zorder=z, style=style)


def arr(ax, p0, p1, color=SUB, lw=2.0, rad=0.0, ms=14, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                 zorder=z, shrinkA=1, shrinkB=1, connectionstyle=f"arc3,rad={rad}"))


def blend(base, a):
    base = np.array(mpl.colors.to_rgb(base))
    return tuple(1 - a * (1 - base))


def vec(ax, x, y, n, cw, ch, base, vals, ec=None):
    """n칸짜리 가로 벡터(칸 색 농도 = 값)"""
    for j in range(n):
        ax.add_patch(Rectangle((x + j * cw, y), cw, ch, fc=blend(base, 0.18 + 0.72 * vals[j]),
                     ec=ec or "white", lw=1.6, zorder=3))


# ================= 그림 9-1: 텍스트가 모델 입력이 되기까지 =================
def fig_pipeline():
    fig, ax = new_ax(13.2, 5.6, (0, 26.4), (0, 11.2))
    heading(ax, 0.4, 10.4, "텍스트가 모델 입력이 되기까지")
    tx(ax, 0.9, 9.45, "입력 문장:  Transformers are changing the world!", fs=11.5, color=SUB, ha="left")
    toks = ["Transform", "##ers", "are", "changing", "the", "world!"]
    ids = [3745, 4026, 38935, 290, 262, 5297]
    rng = np.random.default_rng(3)
    E = rng.random((6, 4)); P = np.array([[0.5 + 0.5 * math.sin(p * f) for f in (1.0, 1.0, 0.1, 0.1)] for p in range(6)])
    I = np.clip((E + P) / 2 + 0.15 * rng.random((6, 4)), 0, 1)
    heads = [("① 토큰", 2.8, BRAND), ("② 토큰 ID", 6.7, AMBER), ("③ 토큰 임베딩", 11.5, INDIGO),
             ("④ 위치 인코딩", 17.0, ROSE), ("⑤ 입력 벡터", 22.6, CORAL)]
    for lab, cx, col in heads:
        rb(ax, cx - 1.9, 8.15, 3.8, 0.8, TINT["gray"], col, lw=1.8, r=0.3)
        tx(ax, cx, 8.55, lab, fs=11.5, color=col, bold=True)
    cw, chh = 0.85, 0.78
    for i in range(6):
        y = 6.55 - i * 1.18
        rb(ax, 1.3, y, 3.0, chh, TINT["brand"], BRAND, lw=1.6, r=0.1)
        tx(ax, 2.8, y + chh / 2, toks[i], fs=11.5, bold=True)
        rb(ax, 5.6, y, 2.2, chh, TINT["amber"], AMBER, lw=1.6, r=0.1)
        tx(ax, 6.7, y + chh / 2, str(ids[i]), fs=11.5, bold=True)
        arr(ax, (4.35, y + chh / 2), (5.55, y + chh / 2), "#9aa3af", lw=1.6, ms=11)
        arr(ax, (7.85, y + chh / 2), (9.7, y + chh / 2), "#9aa3af", lw=1.6, ms=11)
        vec(ax, 9.9, y, 4, cw, chh, INDIGO, E[i])
        tx(ax, 14.0 + 0.4, y + chh / 2, "+", fs=20, bold=True)
        vec(ax, 15.3, y, 4, cw, chh, ROSE, P[i])
        tx(ax, 19.55, y + chh / 2, "=", fs=20, bold=True)
        vec(ax, 20.9, y, 4, cw, chh, CORAL, I[i])
        tx(ax, 26.0, y + chh / 2, f"x{i}", fs=10.5, color=SUB, ha="left")
        tx(ax, 14.0 + 3.6, y + chh / 2 - 0.0, "", fs=1)
    tx(ax, 17.0, 0.35, "위치 0 ~ 5마다 서로 다른 벡터 P[i]를 더합니다", fs=10.5, color=SUB)
    save(fig, "ch9_pipeline.png")


# ================= 그림 9-2: BPE 병합 =================
def fig_bpe():
    steps = [
        ("시작", None, ["l o w </w>", "l o w e r </w>", "n e w e s t </w>", "w i d e s t </w>"], None),
        ("병합 1", "(e, s) -> es", ["l o w </w>", "l o w e r </w>", "n e w es t </w>", "w i d es t </w>"], "es"),
        ("병합 2", "(es, t) -> est", ["l o w </w>", "l o w e r </w>", "n e w est </w>", "w i d est </w>"], "est"),
        ("병합 3", "(est, </w>) -> est</w>", ["l o w </w>", "l o w e r </w>", "n e w est</w>", "w i d est</w>"], "est</w>"),
        ("병합 4", "(l, o) -> lo", ["lo w </w>", "lo w e r </w>", "n e w est</w>", "w i d est</w>"], "lo"),
        ("병합 5", "(lo, w) -> low", ["low </w>", "low e r </w>", "n e w est</w>", "w i d est</w>"], "low"),
    ]
    words = ["low", "lower", "newest", "widest"]
    sw = lambda s: 0.5 + 0.2 * len(s)           # 기호 상자 폭
    gap = 0.08
    def wlen(seq):
        syms = seq.split()
        return sum(sw(s) for s in syms) + gap * (len(syms) - 1)
    maxw = [max(wlen(st[2][k]) for st in steps) for k in range(4)]
    x_start, colgap = 7.4, 1.4
    lefts = []
    x = x_start
    for k in range(4):
        lefts.append(x); x += maxw[k] + colgap
    W = x
    fig, ax = new_ax(13.4, 6.4, (0, W), (0, 14.0))
    heading(ax, 0.4, 13.2, "BPE - 가장 자주 나오는 쌍을 하나씩 합칩니다")
    for w, lx, mw in zip(words, lefts, maxw):
        tx(ax, lx + mw / 2, 11.9, w, fs=13, bold=True, color=INK)
    for k, (name, rule, seqs, new) in enumerate(steps):
        y = 10.2 - k * 1.85
        tx(ax, 0.7, y + 0.55, name, fs=11.5, bold=True, color=BRAND if k else SUB, ha="left")
        if rule:
            tx(ax, 0.7, y - 0.05, rule, fs=9, color=SUB, ha="left")
        for seq, lx in zip(seqs, lefts):
            x = lx
            for s in seq.split():
                hi = new is not None and s == new
                w = sw(s)
                rb(ax, x, y, w, 0.8, TINT["brand"] if hi else TINT["gray"], BRAND if hi else "#9aa3af",
                   lw=2.2 if hi else 1.2, r=0.08)
                tx(ax, x + w / 2, y + 0.4, s, fs=9 if len(s) > 3 else 10, bold=hi)
                x += w + gap
    tx(ax, W / 2, 0.3, "초록 칸 = 새로 만들어진 서브워드.  동률이면 알파벳 순으로 선택", fs=10.5, color=SUB)
    save(fig, "ch9_bpe_steps.png")


# ================= 그림 9-3: 임베딩 층 = 행 조회 =================
def fig_lookup():
    W = np.array([[-1.13, -1.15, -0.25, -0.43], [0.85, 0.69, -0.32, -2.12], [0.47, -0.16, 1.44, 0.27],
                  [0.17, 0.87, -0.14, -0.11], [0.93, 1.26, 2.00, 0.05], [0.62, -0.41, -0.84, -2.32]])
    fig, ax = new_ax(11.8, 5.6, (0, 23.6), (0, 11.2))
    heading(ax, 0.4, 10.4, "임베딩 층은 행렬에서 행을 읽는 표입니다")
    rb(ax, 0.8, 4.6, 4.2, 1.5, TINT["amber"], AMBER, lw=2)
    tx(ax, 2.9, 5.7, "토큰 ID", fs=11, color=SUB)
    tx(ax, 2.9, 5.05, "2", fs=22, bold=True, color=AMBER)
    u = 1.15
    x0, y0 = 8.2, 1.7
    tx(ax, x0 + 2 * u, 9.1, "임베딩 행렬  W  (어휘 6 × 차원 4)", fs=12, bold=True, color=INDIGO)
    for i in range(6):
        tx(ax, x0 - 0.7, y0 + (5 - i) * u + u / 2, str(i), fs=11, color=SUB)
        for j in range(4):
            sel = i == 2
            ax.add_patch(Rectangle((x0 + j * u, y0 + (5 - i) * u), u, u, fc=TINT["indigo"] if not sel else "#ffe9e5",
                         ec="white", lw=2, zorder=3))
            tx(ax, x0 + j * u + u / 2, y0 + (5 - i) * u + u / 2, f"{W[i, j]:.2f}", fs=11, bold=sel)
    ax.add_patch(Rectangle((x0, y0 + 3 * u), 4 * u, u, fc="none", ec=CORAL, lw=3, zorder=5))
    arr(ax, (5.1, 5.35), (x0 - 1.2, y0 + 3.5 * u), AMBER, lw=2.4)
    arr(ax, (x0 + 4 * u + 0.15, y0 + 3.5 * u), (17.4, y0 + 3.5 * u), CORAL, lw=2.4)
    rb(ax, 17.6, y0 + 3.5 * u - 0.7, 5.2, 1.4, TINT["coral"], CORAL, lw=2)
    tx(ax, 20.2, y0 + 3.5 * u + 0.28, "W[2]  =  토큰 2의 벡터", fs=10.5, color=SUB)
    tx(ax, 20.2, y0 + 3.5 * u - 0.28, "0.47  -0.16  1.44  0.27", fs=11.5, bold=True)
    tx(ax, 12.0, 0.45, "원-핫(2)과 W의 곱은 W의 2번 행과 같습니다 — 실제로는 곱하지 않고 행만 읽습니다", fs=10.5, color=SUB)
    save(fig, "ch9_embedding_lookup.png")


# ================= 그림 9-4: 정적 vs 문맥 임베딩 =================
def fig_static_ctx():
    fig, ax = new_ax(12.8, 5.6, (0, 25.6), (0, 11.2))
    heading(ax, 0.4, 10.4, "정적 임베딩과 문맥 임베딩")
    for cx, title, col in [(6.6, "정적 임베딩  (Word2Vec, GloVe)", INDIGO), (19.2, "문맥 임베딩  (BERT, GPT)", BRAND)]:
        rb(ax, cx - 5.9, 1.0, 11.8, 8.2, "white", "#d5dae1", lw=1.6, r=0.3, z=1, shadow=True)
        tx(ax, cx, 8.45, title, fs=12.5, bold=True, color=col)
    sents = ["배가 고프다", "배가 출항한다"]
    for cx in (6.6, 19.2):
        for k, s in enumerate(sents):
            rb(ax, cx - 5.0, 5.6 - k * 2.5, 3.6, 1.0, TINT["gray"], "#9aa3af", lw=1.5, r=0.12)
            tx(ax, cx - 3.2, 6.1 - k * 2.5, s, fs=11.5, bold=True)
    # 정적: 하나의 벡터
    vals = np.array([0.9, 0.3, 0.7, 0.5])
    vec(ax, 8.3, 4.5, 4, 0.8, 0.9, INDIGO, vals)
    tx(ax, 9.9, 5.8, "v(배)", fs=11.5, bold=True, color=INDIGO)
    arr(ax, (5.35, 6.1), (8.2, 5.1), INDIGO, lw=2.2, rad=-0.05)
    arr(ax, (5.35, 3.6), (8.2, 4.9), INDIGO, lw=2.2, rad=0.05)
    tx(ax, 6.6, 1.9, "문장이 달라도 '배'의 벡터는 하나뿐", fs=10.5, color=SUB)
    # 문맥: 서로 다른 벡터
    v1 = np.array([0.9, 0.2, 0.8, 0.3]); v2 = np.array([0.2, 0.9, 0.3, 0.8])
    vec(ax, 21.0, 5.6, 4, 0.8, 0.9, BRAND, v1)
    vec(ax, 21.0, 3.1, 4, 0.8, 0.9, CORAL, v2)
    tx(ax, 22.6, 6.8, "v(배 | 고프다)", fs=10.5, bold=True, color=BRAND)
    tx(ax, 22.6, 2.55, "v(배 | 출항한다)", fs=10.5, bold=True, color=CORAL)
    arr(ax, (17.95, 6.1), (20.9, 6.05), BRAND, lw=2.2)
    arr(ax, (17.95, 3.6), (20.9, 3.55), CORAL, lw=2.2)
    tx(ax, 19.2, 1.9, "같은 단어도 문맥에 따라 다른 벡터", fs=10.5, color=SUB)
    save(fig, "ch9_static_vs_contextual.png")


# ================= 그림 9-5: 코사인 유사도 =================
def fig_cosine():
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.set_xlim(-2.8, 4.8); ax.set_ylim(-2.8, 4.8); ax.set_aspect("equal")
    ax.grid(True, color="#e8ebef", lw=1); ax.axhline(0, color="#c7ccd3", lw=1.4); ax.axvline(0, color="#c7ccd3", lw=1.4)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=SUB, labelsize=9)
    vs = [("b = (2, 4)", (2, 4), INDIGO, (2.2, 4.1), 3.2, 1),
          ("c = (2, -1)", (2, -1), AMBER, (2.15, -1.05), 3.2, 1),
          ("d = (-1, -2)", (-1, -2), CORAL, (-2.6, -2.25), 3.2, 1),
          ("a = (1, 2)", (1, 2), BRAND, (-1.75, 1.95), 4.6, 5)]
    for lab, v, col, pos, lw, z in vs:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                    mutation_scale=18), zorder=z)
        ax.text(pos[0], pos[1], lab, color=col, fontsize=11.5, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
    ax.text(-2.65, 4.45, "cos(a, b) = 1    방향이 같다", fontsize=11, color=INK, bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
    ax.text(-2.65, 3.95, "cos(a, c) = 0    직각, 무관", fontsize=11, color=INK, bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
    ax.text(-2.65, 3.45, "cos(a, d) = -1   방향이 반대", fontsize=11, color=INK, bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
    ax.set_title("코사인 유사도는 길이가 아니라 방향을 비교합니다", fontsize=13, fontweight="bold", color=INK, loc="left", pad=12)
    save(fig, "ch9_cosine.png")


# ================= 그림 9-6: 사인·코사인 위치 인코딩 =================
def fig_pe_waves():
    d = 64
    pos = np.linspace(0, 60, 600)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, color="#e8ebef", lw=1)
    cols = [CORAL, AMBER, BRAND, INDIGO]
    for dim, col in zip((0, 8, 16, 24), cols):
        i = dim // 2
        f = 1.0 / (10000 ** (2 * i / d))
        ax.plot(pos, np.sin(pos * f), color=col, lw=2.6, label=f"차원 {dim}  (주파수 ~ {f:.3f})")
        ip = np.arange(0, 61, 5)
        ax.scatter(ip, np.sin(ip * f), color=col, s=26, zorder=4)
    ax.set_xlabel("위치 pos", fontsize=11, color=SUB); ax.set_ylabel("sin( pos / 10000^(2i/d) )", fontsize=11, color=SUB)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=10, frameon=False, ncol=2)
    ax.set_title("차원마다 주파수가 다른 파동 — 낮은 차원은 빠르게, 높은 차원은 천천히 변합니다", fontsize=12.5,
                 fontweight="bold", color=INK, loc="left", pad=12)
    ax.tick_params(colors=SUB)
    save(fig, "ch9_pe_waves.png")


# ================= 그림 9-8: 인코더 레이어 =================
def _stack_box(ax, cx, y, w, label, col, h=0.9):
    rb(ax, cx - w / 2, y, w, h, TINT[col] if col in TINT else col, {"brand": BRAND, "amber": AMBER, "indigo": INDIGO,
       "coral": CORAL, "rose": ROSE}.get(col, col), lw=1.9, r=0.14, shadow=True)
    tx(ax, cx, y + h / 2, label, fs=11, bold=True)


def fig_encoder_layer():
    fig, ax = new_ax(7.6, 9.4, (0, 12.4), (0, 15.4))
    heading(ax, 0.3, 14.5, "인코더 레이어")
    cx = 5.0
    tx(ax, cx, 0.5, "입력  x", fs=12, bold=True)
    _stack_box(ax, cx, 2.0, 6.0, "멀티 헤드 셀프 어텐션", "brand")
    _stack_box(ax, cx, 4.1, 3.6, "Add & Norm", "amber", h=0.8)
    _stack_box(ax, cx, 6.5, 6.0, "피드포워드 네트워크", "indigo")
    _stack_box(ax, cx, 8.6, 3.6, "Add & Norm", "amber", h=0.8)
    tx(ax, cx, 11.3, "출력  y", fs=12, bold=True)
    for a, b in [(0.9, 1.95), (2.95, 4.05), (4.95, 6.45), (7.45, 8.55), (9.45, 10.85)]:
        arr(ax, (cx, a), (cx, b), SUB, lw=2)
    # 잔차 연결
    for ys, ye in [(1.3, 4.5), (5.8, 9.0)]:
        ax.plot([cx, 9.2], [ys, ys], color=CORAL, lw=2.2, zorder=2)
        ax.plot([9.2, 9.2], [ys, ye], color=CORAL, lw=2.2, zorder=2)
        arr(ax, (9.2, ye), (cx + 1.85, ye), CORAL, lw=2.2)
    tx(ax, 10.5, 3.0, "잔차\n연결", fs=10, color=CORAL, bold=True)
    tx(ax, 10.5, 7.5, "잔차\n연결", fs=10, color=CORAL, bold=True)
    tx(ax, cx, 12.7, "h = LN( x + MHA(x) )     y = LN( h + FFN(h) )", fs=11, color=SUB)
    save(fig, "ch9_encoder_layer.png")


# ================= 그림 9-9: 디코더 레이어 =================
def fig_decoder_layer():
    fig, ax = new_ax(10.4, 11.4, (0, 16.0), (0, 17.6))
    heading(ax, 0.3, 16.7, "디코더 레이어")
    cx = 10.0
    tx(ax, cx, 0.5, "디코더 입력  y", fs=12, bold=True)
    _stack_box(ax, cx, 1.9, 6.4, "마스크드 셀프 어텐션", "brand")
    _stack_box(ax, cx, 3.9, 3.6, "Add & Norm", "amber", h=0.8)
    _stack_box(ax, cx, 5.9, 6.4, "인코더–디코더 어텐션", "coral")
    _stack_box(ax, cx, 7.9, 3.6, "Add & Norm", "amber", h=0.8)
    _stack_box(ax, cx, 9.9, 6.4, "피드포워드 네트워크", "indigo")
    _stack_box(ax, cx, 11.9, 3.6, "Add & Norm", "amber", h=0.8)
    tx(ax, cx, 14.4, "디코더 출력", fs=12, bold=True)
    for a, b in [(0.9, 1.85), (2.85, 3.85), (4.75, 5.85), (6.85, 7.85), (8.75, 9.85), (10.85, 11.85), (12.75, 14.0)]:
        arr(ax, (cx, a), (cx, b), SUB, lw=2)
    # 잔차
    for ys, ye in [(1.3, 4.3), (5.4, 8.3), (9.4, 12.3)]:
        ax.plot([cx, 14.3], [ys, ys], color=CORAL, lw=2, zorder=2)
        ax.plot([14.3, 14.3], [ys, ye], color=CORAL, lw=2, zorder=2)
        arr(ax, (14.3, ye), (cx + 1.85, ye), CORAL, lw=2)
    # 인코더 출력 → 교차 어텐션 (K, V)
    rb(ax, 0.8, 5.35, 3.4, 1.6, TINT["indigo"], INDIGO, lw=2, shadow=True)
    tx(ax, 2.5, 6.15, "인코더 출력", fs=11.5, bold=True, color=INDIGO)
    arr(ax, (4.25, 6.15), (cx - 3.25, 6.35), INDIGO, lw=2.6)
    tx(ax, 5.35, 7.0, "K, V", fs=11.5, bold=True, color=INDIGO)
    tx(ax, cx - 0.5, 5.4, "Q", fs=11.5, bold=True, color=CORAL)
    tx(ax, 4.4, 2.35, "미래 위치를\n가립니다", fs=10, color=SUB)
    save(fig, "ch9_decoder_layer.png")


# ================= 그림 9-10: 어텐션 마스크 비교 =================
def fig_masks():
    words = ["I", "really", "love", "my", "cute", "dog"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    full = np.ones((6, 6)); causal = np.tril(np.ones((6, 6)))
    for ax, M, title, col in [(axes[0], full, "양방향 (BERT) — 모든 단어를 참조", INDIGO),
                              (axes[1], causal, "인과 마스크 (GPT·디코더) — 앞쪽만 참조", BRAND)]:
        cm = mpl.colors.ListedColormap(["#eceff3", col])
        ax.imshow(M, cmap=cm, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(7):
            ax.axhline(i - 0.5, color="white", lw=2.4); ax.axvline(i - 0.5, color="white", lw=2.4)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, "1" if M[i, j] else "0", ha="center", va="center", fontsize=12,
                        color="white" if M[i, j] else "#9aa3af", fontweight="bold")
        ax.set_xticks(range(6)); ax.set_xticklabels(words, fontsize=10.5)
        ax.set_yticks(range(6)); ax.set_yticklabels(words, fontsize=10.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=22)
        ax.set_ylabel("현재 단어 (Q)", fontsize=10.5, color=SUB)
        ax.set_xlabel("참조되는 단어 (K)", fontsize=10.5, color=SUB, labelpad=8)
        ax.xaxis.set_label_position("bottom")
    fig.tight_layout()
    save(fig, "ch9_masks.png")


# ================= 그림 9-12: MLM 마스킹 =================
def fig_mlm():
    fig, ax = new_ax(12.6, 5.4, (0, 25.2), (0, 10.8))
    heading(ax, 0.4, 10.0, "마스크드 언어 모델 — 고른 토큰을 가리고 맞힙니다")
    toks = ["my", "dog", "is", "cute"]
    tx(ax, 0.9, 8.4, "원문", fs=11.5, bold=True, color=SUB, ha="left")
    for k, t in enumerate(toks):
        hi = t == "cute"
        rb(ax, 3.2 + k * 2.5, 7.9, 2.2, 1.0, TINT["coral"] if hi else TINT["gray"], CORAL if hi else "#9aa3af", lw=2.2 if hi else 1.5, r=0.12)
        tx(ax, 4.3 + k * 2.5, 8.4, t, fs=12, bold=hi)
    tx(ax, 14.6, 8.4, "← 15% 중 하나로 'cute'가 뽑혔다고 가정", fs=10.5, color=SUB, ha="left")
    rows = [("80%", "[MASK]로 바꿈", ["my", "dog", "is", "[MASK]"], BRAND, 5.6),
            ("10%", "임의의 다른 토큰", ["my", "dog", "is", "apple"], AMBER, 3.6),
            ("10%", "그대로 둠", ["my", "dog", "is", "cute"], INDIGO, 1.6)]
    for pct, rule, seq, col, y in rows:
        rb(ax, 0.6, y - 0.1, 5.6, 1.2, TINT["gray"], col, lw=2, r=0.15)
        tx(ax, 1.6, y + 0.5, pct, fs=14, bold=True, color=col)
        tx(ax, 4.2, y + 0.5, rule, fs=11, color=INK)
        arr(ax, (6.3, y + 0.5), (7.7, y + 0.5), col, lw=2.2)
        for k, t in enumerate(seq):
            sp = t in ("[MASK]", "apple") and k == 3
            rb(ax, 8.0 + k * 2.4, y, 2.1, 1.0, TINT["brand"] if sp and t == "[MASK]" else (TINT["amber"] if sp else TINT["gray"]),
               col if k == 3 else "#9aa3af", lw=2.2 if k == 3 else 1.4, r=0.12)
            tx(ax, 9.05 + k * 2.4, y + 0.5, t, fs=11, bold=(k == 3))
        arr(ax, (17.9, y + 0.5), (19.6, y + 0.5), "#9aa3af", lw=1.8)
        rb(ax, 19.8, y, 4.6, 1.0, TINT["coral"], CORAL, lw=2, r=0.12)
        tx(ax, 22.1, y + 0.5, "정답: cute", fs=11.5, bold=True, color=CORAL)
    save(fig, "ch9_mlm_masking.png")


if __name__ == "__main__":
    fig_pipeline(); fig_bpe(); fig_lookup(); fig_static_ctx(); fig_cosine()
    fig_pe_waves(); fig_encoder_layer(); fig_decoder_layer(); fig_masks(); fig_mlm()
    print("done ch9")
