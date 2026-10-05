# -*- coding: utf-8 -*-
"""6주차 논문 스타일 그림 생성.  실행: python gen_paper6.py [그림이름 ...]"""
import sys
from _paper import *


# ───────────────────────── U-Net 전체 구조 ─────────────────────────
def fig_unet(numbered=False):
    chans = [64, 128, 256, 512, 1024]
    sizes = ["H×W", "H/2×W/2", "H/4×W/4", "H/8×W/8", "H/16×W/16"]
    cw = lambda c: 0.10 + 0.00042 * c          # 채널 수 → 폭
    hh = lambda i: 0.95 * 0.78 ** i            # 해상도 → 높이(압축 척도)
    ys = [4.55, 3.45, 2.40, 1.45, 0.55]
    gap = 1.15
    xe = [0.75 + i * gap for i in range(5)]
    xd = [xe[4] + (4 - i) * gap + 0.35 for i in range(5)]
    xd[4] = None

    fig, ax = canvas(11.6, 5.4 + (0.7 if numbered else 0), (-0.1, 11.6), (-0.6, 5.35 + (0.7 if numbered else 0)))

    slab(ax, 0.0, ys[0], 0.10, hh(0), F_GRAY)
    label(ax, 0.05, ys[0] + hh(0) / 2 + 0.16, "3", fs=7.5)
    label(ax, 0.05, ys[0] - hh(0) / 2 - 0.16, "입력", fs=7.5, color=SUB)
    arrow(ax, (0.14, ys[0]), (xe[0] - 0.05, ys[0]), color=BLUE, lw=1.0)

    for i, c in enumerate(chans):
        w = cw(c)
        slab(ax, xe[i], ys[i], w, hh(i), F_BLUE if i < 4 else F_ORANGE)
        label(ax, xe[i] + w / 2, ys[i] + hh(i) / 2 + 0.16, str(c), fs=7.5)
        label(ax, xe[i] + w / 2, ys[i] - hh(i) / 2 - 0.16, sizes[i], fs=6.5, color=SUB)
        if i < 4:
            x0 = xd[i]
            slab(ax, x0, ys[i], w, hh(i), F_BLUE)           # 스킵(인코더에서 복사)
            slab(ax, x0 + w, ys[i], w, hh(i), F_GREEN)      # 업샘플 결과
            label(ax, x0 + w, ys[i] + hh(i) / 2 + 0.16, f"{c}+{c}", fs=7.5)
            label(ax, x0 + w, ys[i] - hh(i) / 2 - 0.16, sizes[i], fs=6.5, color=SUB)

    # 풀링: 인코더 i → i+1
    for i in range(4):
        p0 = (xe[i] + cw(chans[i]) + 0.04, ys[i] - hh(i) * 0.25)
        p1 = (xe[i + 1] - 0.04, ys[i + 1] + hh(i + 1) * 0.25)
        arrow(ax, p0, p1, color=RED, lw=1.1)
    # 업샘플: 병목 → 디코더 3, 디코더 k+1 → k
    for i in (3, 2, 1, 0):
        if i == 3:
            p0 = (xe[4] + cw(1024) + 0.04, ys[4] + hh(4) * 0.25)
        else:
            p0 = (xd[i + 1] + 2 * cw(chans[i + 1]) + 0.04, ys[i + 1] + hh(i + 1) * 0.25)
        p1 = (xd[i] - 0.04, ys[i] - hh(i) * 0.25)
        arrow(ax, p0, p1, color=GREEN, lw=1.1)
    # 스킵 연결
    for i in range(4):
        y = ys[i] + hh(i) * 0.30
        arrow(ax, (xe[i] + cw(chans[i]) + 0.04, y), (xd[i] - 0.04, y),
              color=GRAY, lw=1.0, ls=(0, (4, 2.5)))
    label(ax, (xe[0] + xd[0]) / 2 + 0.3, ys[0] + hh(0) * 0.30 + 0.17,
          "스킵 연결 (복사 후 채널 방향 결합)", fs=7.5, color=SUB)

    # 출력 1×1 합성곱
    xo = xd[0] + 2 * cw(64) + 0.7
    arrow(ax, (xd[0] + 2 * cw(64) + 0.04, ys[0]), (xo - 0.05, ys[0]), color=PURPLE, lw=1.0)
    slab(ax, xo, ys[0], 0.10, hh(0), F_PURPLE)
    label(ax, xo + 0.05, ys[0] + hh(0) / 2 + 0.16, "K", fs=7.5)
    label(ax, xo + 0.05, ys[0] - hh(0) / 2 - 0.16, "분할 지도", fs=7.5, color=SUB)

    label(ax, 2.3, -0.02, "인코더 (수축 경로)", fs=8, color=SUB, bold=True)
    label(ax, 9.3, -0.02, "디코더 (확장 경로)", fs=8, color=SUB, bold=True)

    items = [(BLUE, "-", "3×3 합성곱 ×2 + ReLU"), (RED, "-", "2×2 최대 풀링 (↓2)"),
             (GREEN, "-", "2×2 전치 합성곱 (↑2)"), (GRAY, (0, (4, 2.5)), "스킵 연결 (복사·결합)"),
             (PURPLE, "-", "1×1 합성곱 → K 클래스")]
    if SLIDE > 1:
        ax.set_ylim(-1.35, 5.35)
        pos = [(0.2, -0.5), (4.1, -0.5), (8.0, -0.5), (0.2, -1.1), (4.1, -1.1)]
    else:
        pos = [(0.2 + 2.25 * i, -0.45) for i in range(5)]
    for (col, ls, txt), (x, ly) in zip(items, pos):
        arrow(ax, (x, ly), (x + 0.42, ly), color=col, lw=1.2, ls=ls)
        label(ax, x + 0.52, ly, txt, fs=7.2, ha="left")
    if numbered:
        def mark(x, y, n):
            ax.add_patch(Circle((x, y), 0.17, fc=INK, ec="none", zorder=9))
            ax.text(x, y, str(n), fontsize=8.5, color="white", ha="center", va="center", fontweight="bold", zorder=10)
        mark(0.2, ys[0] + hh(0) / 2 + 0.58, 1)                        # ① 입력
        mark(xe[1] - 0.72, ys[1] + 0.12, 2)                            # ② 인코더
        mark(xe[4] - 0.55, ys[4] - 0.12, 3)                             # ③ 병목
        mark(xd[1] + 2 * cw(128) + 0.72, ys[1] + 0.12, 4)              # ④ 디코더
        mark((xe[0] + xd[0]) / 2 - 2.35, ys[0] + hh(0) * 0.30 + 0.17, 5)   # ⑤ 스킵 연결
        mark(xo + 0.05, ys[0] + hh(0) / 2 + 0.58, 6)                   # ⑥ 출력
    save(fig, "ch6_p_unet_num" if numbered else "ch6_p_unet")


# ───────────────────────── 과제 비교: 분류·탐지·분할 ─────────────────────────
def _scene(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 10, 10, fc="#eef3f8", ec="none"))
    ax.add_patch(Rectangle((0, 0), 10, 3.2, fc="#e4ebe2", ec="none"))


def _objs():
    """고양이 두 마리(같은 클래스, 일부 겹침)와 자동차 한 대. (cx, cy, w, h)"""
    return {"cat1": (3.0, 4.6, 3.8, 2.8), "cat2": (5.6, 3.9, 3.4, 2.5), "car": (7.7, 7.4, 3.6, 2.0)}


def _draw_objs(ax, mode):
    o = _objs()
    from matplotlib.patches import Ellipse
    kinds = {"cat1": "e", "cat2": "e", "car": "r"}
    if mode == "plain":
        cols = {"cat1": "#b9c0c8", "cat2": "#a4acb5", "car": "#8f98a3"}
    elif mode == "semantic":
        cols = {"cat1": ORANGE, "cat2": ORANGE, "car": BLUE}
    else:
        cols = {"cat1": ORANGE, "cat2": GREEN, "car": BLUE}
    for k in ("car", "cat1", "cat2"):
        cx, cy, w, h = o[k]
        if kinds[k] == "e":
            p = Ellipse((cx, cy), w, h, fc=cols[k], ec="white" if mode != "plain" else "#7d8791", lw=0.8,
                        alpha=0.95 if mode == "plain" else 0.78)
        else:
            p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.4",
                               fc=cols[k], ec="white" if mode != "plain" else "#7d8791", lw=0.8,
                               alpha=0.95 if mode == "plain" else 0.78)
        ax.add_patch(p)


def fig_tasks():
    fig, axs = plt.subplots(1, 5, figsize=(10.4, 2.5))
    titles = ["(a) 입력", "(b) 이미지 분류", "(c) 객체탐지", "(d) 의미 분할", "(e) 인스턴스 분할"]
    for ax, t in zip(axs, titles):
        _scene(ax)
        ax.set_title(t, fontsize=9, fontweight="bold", color=INK, pad=5)
    _draw_objs(axs[0], "plain")
    _draw_objs(axs[1], "plain")
    box(axs[1], 0.8, 0.5, 8.4, 2.2, "출력: 클래스 1개\n\"고양이\"", fc="white", ec=LINE, fs=7.5)
    _draw_objs(axs[2], "plain")
    o = _objs()
    for k, col, name in (("cat1", ORANGE, "고양이"), ("cat2", ORANGE, "고양이"), ("car", BLUE, "자동차")):
        cx, cy, w, h = o[k]
        axs[2].add_patch(Rectangle((cx - w / 2 - 0.15, cy - h / 2 - 0.15), w + 0.3, h + 0.3,
                                   fc="none", ec=col, lw=1.4, zorder=5))
        axs[2].text(cx - w / 2 - 0.15, cy + h / 2 + 0.35, name, fontsize=6.5, color="white", va="bottom",
                    bbox=dict(fc=col, ec="none", pad=1.2), zorder=6)
    _draw_objs(axs[3], "semantic")
    _draw_objs(axs[4], "instance")
    # 범례(세로 배치)
    for ax, items in ((axs[3], [(ORANGE, "고양이"), (BLUE, "자동차")]),
                      (axs[4], [(ORANGE, "고양이 1"), (GREEN, "고양이 2"), (BLUE, "자동차")])):
        for k, (col, name) in enumerate(items[::-1]):
            y = 0.45 + k * 0.95
            ax.add_patch(Rectangle((0.5, y - 0.3), 0.6, 0.6, fc=col, ec="none", alpha=0.85))
            ax.text(1.35, y, name, fontsize=6.4, va="center", color=INK)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "ch6_p_tasks")


# ───────────────────────── 전이학습 전략 ─────────────────────────
def fig_transfer():
    stages = [("conv1\n+pool", "64×56²"), ("layer1", "64×56²"), ("layer2", "128×28²"),
              ("layer3", "256×14²"), ("layer4", "512×7²"), ("avgpool", "512")]
    dx = 1.5 if SLIDE > 1 else 0.0
    fig, ax = canvas(10.5 + dx, 4.9, (0, 10.5 + dx), (-0.7, 4.7))
    bw, bh, gap = 0.98, 0.62, 0.22
    x0 = 1.95 + dx

    def row(y, states, fc_txt, fc_state, title, sub):
        label(ax, 0.05, y + 0.15, title, fs=8.5, ha="left", bold=True)
        label(ax, 0.05, y - 0.17, sub, fs=6.8, ha="left", color=SUB)
        x = x0
        for (nm, shp), st in zip(stages, states):
            fc, ec, ls = {"pre": (F_PURPLE, PURPLE, "-"), "frozen": (F_GRAY, GRAY, "-"),
                          "tune": (F_BLUE, BLUE, "-")}[st]
            box(ax, x, y - bh / 2, bw, bh, nm, fc=fc, ec=ec, fs=7.2, ls=ls)
            if st == "frozen":
                ax.plot([x + 0.06, x + bw - 0.06], [y - bh / 2 + 0.06] * 2, color=GRAY, lw=0.0)
            x += bw + gap
            if nm != "avgpool":
                arrow(ax, (x - gap + 0.01, y), (x - 0.01, y), color=SUB, lw=0.8, ms=5)
        arrow(ax, (x - gap + 0.01, y), (x - 0.01, y), color=SUB, lw=0.8, ms=5)
        fc, ec = {"old": (F_PURPLE, PURPLE), "new": (F_ORANGE, ORANGE)}[fc_state]
        box(ax, x, y - bh / 2, bw + 0.15, bh, fc_txt, fc=fc, ec=ec, fs=7.2)

    # 열 머리(텐서 크기)
    x = x0
    for nm, shp in stages:
        label(ax, x + bw / 2, 4.45, shp, fs=6.8, color=SUB)
        x += bw + gap
    label(ax, x + 0.5, 4.45, "출력", fs=6.8, color=SUB)
    label(ax, 0.05, 4.45, "입력 3×224²", fs=6.8, ha="left", color=SUB)
    # 저수준→고수준 브래킷
    arrow(ax, (x0, 4.12), (x0 + 6 * (bw + gap) - gap, 4.12), color=LINE, lw=0.9, style="<|-|>", ms=6)
    label(ax, x0 + 0.3, 3.92, "저수준 특징 (모서리·색·질감) — 과제 공통", fs=6.8, ha="left", color=SUB)
    label(ax, x0 + 6 * (bw + gap) - gap - 0.05, 3.92, "고수준 특징 (부분·물체) — 과제 특화", fs=6.8, ha="right", color=SUB)

    row(3.15, ["pre"] * 6, "fc\n1000 클래스", "old", "(a) 사전학습 모델", "ImageNet 학습 완료")
    row(1.85, ["frozen"] * 6, "fc (신규)\n10 클래스", "new", "(b) 특징 추출", "합성곱부 동결, fc만 학습")
    row(0.55, ["frozen", "frozen", "tune", "tune", "tune", "tune"], "fc (신규)\n10 클래스", "new",
        "(c) 미세조정", "앞쪽 동결, 뒤쪽 재학습")

    ly = -0.45
    for i, (fc, ec, txt) in enumerate([(F_PURPLE, PURPLE, "사전학습 완료"), (F_GRAY, GRAY, "동결"),
                                       (F_BLUE, BLUE, "재학습"), (F_ORANGE, ORANGE, "새로 학습")]):
        xx = 1.0 + dx + i * (2.25 + 0.55 * (SLIDE > 1))
        box(ax, xx, ly - 0.11, 0.32, 0.22, "", fc=fc, ec=ec, r=0.03)
        label(ax, xx + 0.42, ly, txt, fs=7.4, ha="left")
    save(fig, "ch6_p_transfer")


# ───────────────────────── 2단계 vs 1단계, R-CNN 계보 ─────────────────────────
def fig_detectors():
    fig, ax = canvas(10.0, 5.6, (0, 10.0), (0, 5.6))
    bh = 0.78

    def chain(y, items, fcs, x0=1.6, bw=1.5, gap=0.38):
        x = x0
        for k, (t, fc, ec) in enumerate(zip(items, fcs, [None] * len(items))):
            f, e = fc
            box(ax, x, y - bh / 2, bw, bh, t, fc=f, ec=e, fs=7.4)
            if k < len(items) - 1:
                arrow(ax, (x + bw + 0.03, y), (x + bw + gap - 0.03, y), color=SUB, lw=0.9, ms=6)
            x += bw + gap
        return x

    label(ax, 0.05, 5.25, "(a) 2단계 검출기", fs=9, ha="left", bold=True)
    chain(4.45, ["입력\n이미지", "백본 CNN\n(특징 맵)", "후보 영역 제안\n(RPN 등)", "RoI 풀링\n(고정 크기화)", "분류 +\n박스 회귀"],
          [(F_GRAY, GRAY), (F_BLUE, BLUE), (F_ORANGE, ORANGE), (F_BLUE, BLUE), (F_GREEN, GREEN)], x0=0.25, bw=1.5, gap=0.42)
    label(ax, 0.25, 3.82, "먼저 후보를 좁힌 뒤 정밀하게 분류  →  정확도 높음, 속도 느림", fs=7.2, ha="left", color=SUB)

    label(ax, 0.05, 3.3, "(b) 1단계 검출기", fs=9, ha="left", bold=True)
    chain(2.5, ["입력\n이미지", "백본 + 넥\n(다중 스케일)", "검출 헤드\n(모든 위치에서\n한 번에 예측)", "NMS\n(중복 제거)"],
          [(F_GRAY, GRAY), (F_BLUE, BLUE), (F_GREEN, GREEN), (F_PURPLE, PURPLE)], x0=0.25, bw=1.75, gap=0.5)
    label(ax, 0.25, 1.87, "후보 제안 없이 한 번의 순전파  →  속도 빠름 (실시간), 최근에는 정확도도 근접", fs=7.2, ha="left", color=SUB)

    label(ax, 0.05, 1.4, "(c) R-CNN 계열의 발전: 병목을 하나씩 제거", fs=9, ha="left", bold=True)
    names = [("R-CNN", "후보 ~2000개마다\nCNN을 따로 실행"), ("Fast R-CNN", "이미지 전체에 CNN 1회\n+ RoI 풀링"),
             ("Faster R-CNN", "후보 제안도 신경망(RPN)\n→ 종단간 학습"), ("Mask R-CNN", "마스크 분기 + RoIAlign\n→ 인스턴스 분할")]
    x = 0.25
    for k, (n, d) in enumerate(names):
        box(ax, x, 0.15, 2.05, 0.95, "", fc=F_GRAY, ec=LINE)
        label(ax, x + 1.025, 0.88, n, fs=8, bold=True)
        label(ax, x + 1.025, 0.46, d, fs=6.8, color=SUB)
        if k < 3:
            arrow(ax, (x + 2.08, 0.62), (x + 2.47, 0.62), color=SUB, lw=0.9, ms=6)
        x += 2.5
    save(fig, "ch6_p_detectors")


# ───────────────────────── IoU ─────────────────────────
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter), inter


def fig_iou():
    gt = (2, 2, 7, 7)
    preds = [(5.2, 4.8, 10.2, 9.8), (2.9, 2.9, 7.9, 7.9), (2.2, 2.3, 7.2, 7.1)]
    titles = ["(a) 크게 어긋남", "(b) 절반가량 겹침", "(c) 거의 일치"]
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.0))
    for ax, p, t in zip(axs, preds, titles):
        ax.set_xlim(0, 11); ax.set_ylim(0, 11); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 11, 11, fc="white", ec=LINE, lw=0.6))
        ix0, iy0 = max(gt[0], p[0]), max(gt[1], p[1]); ix1, iy1 = min(gt[2], p[2]), min(gt[3], p[3])
        if ix1 > ix0 and iy1 > iy0:
            ax.add_patch(Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fc=F_BLUE, ec="none", zorder=1))
        ax.add_patch(Rectangle(gt[:2], gt[2] - gt[0], gt[3] - gt[1], fc="none", ec=GREEN, lw=1.6, zorder=3))
        ax.add_patch(Rectangle(p[:2], p[2] - p[0], p[3] - p[1], fc="none", ec=RED, lw=1.6, ls=(0, (4, 2)), zorder=3))
        v, _ = _iou(gt, p)
        ax.set_title(t, fontsize=9, fontweight="bold", pad=4)
        ax.text(5.5, -0.9, f"IoU = {v:.2f}", fontsize=9, ha="center", color=INK, fontweight="bold")
    ly = 11.9
    axs[0].plot([], [], color=GREEN, lw=1.6, label="정답 상자")
    axs[0].plot([], [], color=RED, lw=1.6, ls=(0, (4, 2)), label="예측 상자")
    axs[0].add_patch(Rectangle((0, 0), 0, 0, fc=F_BLUE, ec=BLUE, label="교집합"))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, 1.0, r"$\mathrm{IoU}=\dfrac{\mathrm{Area}(B_{pred}\cap B_{gt})}{\mathrm{Area}(B_{pred}\cup B_{gt})}$",
             ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "ch6_p_iou")


# ───────────────────────── NMS ─────────────────────────
def _nms(boxes, scores, thr):
    idx = list(np.argsort(scores)[::-1]); keep = []
    while idx:
        i = idx.pop(0); keep.append(i)
        idx = [j for j in idx if _iou(boxes[i], boxes[j])[0] < thr]
    return keep


def fig_nms():
    from matplotlib.patches import Ellipse
    boxes = [(1.2, 2.4, 5.4, 6.8), (1.8, 3.1, 6.0, 7.5), (0.6, 1.8, 4.8, 6.2), (2.0, 3.0, 6.2, 7.4),
             (6.9, 2.2, 9.9, 5.4), (7.4, 2.6, 10.4, 5.8)]
    scores = np.array([0.91, 0.84, 0.72, 0.55, 0.88, 0.67])
    thr = 0.5
    keep = _nms(boxes, scores, thr)
    assert sorted(keep) == [0, 4], keep
    corner = ["tl", "tr", "bl", "br", "tl", "tr"]
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.4))
    for ax, ttl in zip(axs, ["(a) NMS 이전: 한 물체에 상자가 여러 개", f"(b) NMS 이후 (IoU 임계값 {thr})"]):
        ax.set_xlim(0, 11); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 11, 10, fc="white", ec=LINE, lw=0.6))
        ax.add_patch(Ellipse((3.4, 4.8), 3.4, 3.8, fc="#d3d9df", ec="none"))
        ax.add_patch(Ellipse((8.7, 3.9), 2.4, 2.6, fc="#d3d9df", ec="none"))
        ax.set_title(ttl, fontsize=8.6, fontweight="bold", pad=4)

    def tag(ax, b, c, txt, col, fs, bold):
        x = b[0] if c[1] == "l" else b[2]; y = b[3] if c[0] == "t" else b[1]
        ax.text(x, y + (0.1 if c[0] == "t" else -0.1), txt, fontsize=fs, color=col, fontweight="bold" if bold else "normal",
                ha="left" if c[1] == "l" else "right", va="bottom" if c[0] == "t" else "top", zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))

    for i, (b, s_) in enumerate(zip(boxes, scores)):
        axs[0].add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fc="none", ec=BLUE, lw=1.0))
        tag(axs[0], b, corner[i], f"{s_:.2f}", BLUE, 6.5, False)
        if i in keep:
            axs[1].add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fc="none", ec=GREEN, lw=1.8))
            tag(axs[1], b, corner[i], f"{s_:.2f}", GREEN, 7.5, True)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "ch6_p_nms")
    print("NMS keep:", keep)


# ───────────────────────── YOLO 격자 · 출력 텐서 · 앵커 프리 ─────────────────────────
def _cuboid(ax, x, y, w, h, d, fc_front, fc_top, fc_side, ec=INK, lw=0.8):
    """정면(w×h)에 깊이 d를 비스듬히 붙인 육면체."""
    dx, dy = d * 0.55, d * 0.38
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=fc_top, ec=ec, lw=lw, zorder=2))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         fc=fc_side, ec=ec, lw=lw, zorder=2))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc_front, ec=ec, lw=lw, zorder=3))
    return dx, dy


def fig_yolo_grid():
    from matplotlib.patches import Ellipse
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.5), gridspec_kw={"width_ratios": [1, 1.25, 1]})

    # (a) 격자 분할과 담당 칸
    ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) 격자 분할: 중심이 속한 칸이 담당", fontsize=8.6, fontweight="bold", pad=4)
    ax.add_patch(Rectangle((0, 0), 10, 10, fc="#eef3f8", ec=LINE, lw=0.6))
    S = 7; c = 10 / S
    cx, cy, ow, oh = 4.9, 5.5, 4.2, 3.4
    ci, cj = int(cx // c), int(cy // c)
    ax.add_patch(Rectangle((ci * c, cj * c), c, c, fc=F_ORANGE, ec="none", zorder=1))
    ax.add_patch(Ellipse((cx, cy), ow, oh, fc="#b9c0c8", ec="#7d8791", lw=0.8, zorder=2))
    for k in range(S + 1):
        ax.plot([0, 10], [k * c] * 2, color=LINE, lw=0.5, zorder=3)
        ax.plot([k * c] * 2, [0, 10], color=LINE, lw=0.5, zorder=3)
    ax.add_patch(Rectangle((cx - ow / 2 - 0.1, cy - oh / 2 - 0.1), ow + 0.2, oh + 0.2, fc="none", ec=GREEN, lw=1.5, zorder=4))
    ax.plot([cx], [cy], "o", color=RED, ms=4, zorder=5)
    ax.add_patch(Rectangle((ci * c, cj * c), c, c, fc="none", ec=ORANGE, lw=1.6, zorder=5))
    ax.text(5, -0.9, r"$S\times S$ 격자 (그림은 $S=7$)", fontsize=7.8, ha="center", color=SUB)

    # (b) 출력 텐서
    ax = axs[1]; ax.set_xlim(0, 13); ax.set_ylim(-0.6, 8.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(r"(b) 출력 텐서  $S\times S\times(B\cdot 5+C)$", fontsize=8.6, fontweight="bold", pad=4)
    w = h = 3.4; d = 5.2
    dx, dy = _cuboid(ax, 0.8, 2.5, w, h, d, F_BLUE, "#eaf1fa", "#c6d8ee")
    label(ax, 0.8 + w / 2, 2.1, "S", fs=8.5); label(ax, 0.4, 2.5 + h / 2, "S", fs=8.5)
    label(ax, 0.8 + w + dx / 2 + 0.55, 2.5 + h + dy + 0.35, r"$B\cdot 5+C$", fs=8.5)
    label(ax, 0.8 + w * 0.5, 2.5 + h * 0.5, "한 칸", fs=8, color=INK)
    label(ax, 6.2, 1.0, "한 칸 = 길이 B·5+C 의 벡터", fs=8, color=SUB)
    label(ax, 6.2, 0.2, "원형 YOLO(VOC): S=7, B=2, C=20 → 7×7×30", fs=7.4, color=SUB)

    # (c) 앵커 프리: 격자점에서 네 변까지 거리
    ax = axs[2]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(c) YOLOv8: 앵커 프리 거리 예측", fontsize=8.6, fontweight="bold", pad=4)
    ax.add_patch(Rectangle((0, 0), 10, 10, fc="white", ec=LINE, lw=0.6))
    ax.add_patch(Ellipse((5.2, 5.2), 5.4, 4.2, fc="#d3d9df", ec="none"))
    bx = (2.0, 2.8, 8.2, 7.6)
    ax.add_patch(Rectangle(bx[:2], bx[2] - bx[0], bx[3] - bx[1], fc="none", ec=GREEN, lw=1.5))
    px, py = 4.6, 4.6
    for gx in np.arange(0.5, 10, 1.0):
        for gy in np.arange(0.5, 10, 1.0):
            ax.plot([gx], [gy], ".", color="#c8cfd6", ms=1.6)
    ax.plot([px], [py], "o", color=RED, ms=5, zorder=6)
    for (x1, y1, x2, y2, lb, off) in ((px, py, bx[0], py, "l", (0, 0.45)), (px, py, px, bx[3], "t", (0.5, 0)),
                                       (px, py, bx[2], py, "r", (0, 0.45)), (px, py, px, bx[1], "b", (0.5, 0))):
        arrow(ax, (x1, y1), (x2, y2), color=BLUE, lw=1.1, ms=6)
        label(ax, (x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], lb, fs=9, color=BLUE, bold=True)
    ax.text(5, -0.9, "격자점 → 상자 네 변까지의 거리 (l, t, r, b)", fontsize=7.6, ha="center", color=SUB)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "ch6_p_yolo_grid")


# ───────────────────────── YOLOv8n 구조 (ultralytics로 검증한 shape) ─────────────────────────
def fig_yolov8():
    fig, ax = canvas(12.4, 5.5, (0, 12.4), (-0.35, 5.3))
    rows = [4.0, 2.55, 1.1]
    names = ["P3", "P4", "P5"]
    shp = ["80×80×64", "40×40×128", "20×20×256"]
    strd = ["stride 8", "stride 16", "stride 32"]
    obj = ["작은 물체", "중간 물체", "큰 물체"]

    # 열 제목
    for x, t in ((2.55, "① 백본: 특징 추출"), (6.1, "② 넥: PAN-FPN 다중 스케일 결합"), (9.7, "③ 헤드: 분리형 · 앵커 프리")):
        label(ax, x, 5.05, t, fs=8.6, bold=True)
    # 입력
    box(ax, 0.1, 1.35, 1.1, 2.95, "입력\n이미지\n\n640×640\n×3", fc=F_GRAY, ec=GRAY, fs=7.4)
    arrow(ax, (1.23, 2.82), (1.65, 2.82), color=SUB, ms=6)
    # 백본
    box(ax, 1.7, 0.45, 1.75, 4.3, "", fc="white", ec=BLUE, lw=1.0, r=0.1)
    label(ax, 2.575, 4.55, "Conv · C2f 반복", fs=6.6, color=SUB)
    for y, n, s, st in zip(rows, names, shp, strd):
        box(ax, 1.85, y - 0.36, 1.45, 0.72, f"{n}  {s}\n{st}", fc=F_BLUE, ec=BLUE, fs=6.8)
    for y0, y1 in ((rows[0], rows[1]), (rows[1], rows[2])):
        arrow(ax, (2.575, y0 - 0.38), (2.575, y1 + 0.38), color=RED, lw=0.9, ms=5)
    label(ax, 3.0, 0.62, "SPPF 포함", fs=6.2, color=SUB)
    # 넥
    box(ax, 4.5, 0.45, 3.2, 4.3, "", fc="white", ec=GREEN, lw=1.0, r=0.1)
    nx = 5.55
    for y, n, s in zip(rows, names, shp):
        box(ax, nx, y - 0.36, 1.45, 0.72, f"N{n[1]}  {s}", fc=F_GREEN, ec=GREEN, fs=6.8)
        arrow(ax, (3.32, y), (nx - 0.02, y), color=GRAY, lw=1.0, ls=(0, (3, 2)), ms=5)   # 백본→넥(결합)
    for k in (1, 0):   # 하향식(위로): 업샘플 + 결합
        arrow(ax, (nx + 0.35, rows[k + 1] + 0.38), (nx + 0.35, rows[k] - 0.38), color=GREEN, lw=1.1, ms=6)
    for k in (0, 1):   # 상향식(아래로): 스트라이드 2 합성곱 + 결합
        arrow(ax, (nx + 1.1, rows[k] - 0.38), (nx + 1.1, rows[k + 1] + 0.38), color=RED, lw=1.1, ms=6)
    label(ax, 5.25, 0.62, "↑ 업샘플+결합", fs=6.2, color=GREEN, ha="center")
    label(ax, 7.0, 0.62, "↓ Conv(s2)+결합", fs=6.2, color=RED, ha="center")
    # 헤드
    for y, st, ob in zip(rows, strd, obj):
        arrow(ax, (7.05, y), (8.05, y), color=SUB, lw=0.9, ms=5)
        box(ax, 8.1, y + 0.03, 3.2, 0.42, "박스 분기  Conv·Conv·1×1 → 64 (=4×16, DFL)", fc=F_ORANGE, ec=ORANGE, fs=6.3)
        box(ax, 8.1, y - 0.45, 3.2, 0.42, "클래스 분기  Conv·Conv·1×1 → 80 (COCO)", fc=F_PURPLE, ec=PURPLE, fs=6.3)
        label(ax, 11.45, y, ob, fs=6.2, color=SUB, ha="left")
    # 출력
    arrow(ax, (11.35, 4.0), (11.35, 4.0), color=SUB)
    box(ax, 4.5, -0.3, 7.8, 0.0, "", fc="white", ec="white")
    label(ax, 6.2, -0.22, "세 스케일의 예측을 이어 붙임:  80² + 40² + 20² = 6400 + 1600 + 400 = 8400 후보  →  출력 (84, 8400) = (4 + 80, 8400)  →  NMS",
          fs=7.4, color=INK)
    save(fig, "ch6_p_yolov8")


# ───────────────────────── U-Net 스킵 연결 한 단계 (코드 6-3과 일치) ─────────────────────────
def fig_unet_skip():
    fig, ax = canvas(10.8, 4.6, (0, 10.8), (-1.5, 3.15))
    mono = dict(family="DejaVu Sans Mono")
    wd = lambda c: 0.1 + c * 0.0105
    H = 1.15

    def tensor(x, yc, c, h, fc, shape, name=None):
        slab(ax, x, yc, wd(c), h, fc)
        label(ax, x + wd(c) / 2, yc - h / 2 - 0.2, shape, fs=6.8, color=INK, **mono)
        if name:
            label(ax, x + wd(c) / 2, yc + h / 2 + 0.2, name, fs=7.2, color=SUB)

    yU, yS, yM = 2.0, 0.0, 1.0           # 업샘플, 스킵, 결합 행
    tensor(0.2, yM + 0.5, 128, 0.7, F_ORANGE, "(N,128,H,W)", "bottleneck")
    arrow(ax, (1.65, yM + 0.5), (3.0, yU), color=GREEN, lw=1.2)
    label(ax, 1.55, 2.78, "ConvTranspose2d\n(128, 64, 2, stride=2)", fs=6.2, color=GREEN, **mono)
    tensor(3.15, yU, 64, H, F_GREEN, "(N,64,2H,2W)", "x (업샘플)")
    tensor(3.15, yS, 64, H, F_BLUE, "(N,64,2H,2W)", "enc_feat (인코더 특징)")
    arrow(ax, (4.0, yU), (5.55, yM + 0.12), color=SUB, lw=1.0)
    arrow(ax, (4.0, yS), (5.55, yM - 0.12), color=SUB, lw=1.0)
    label(ax, 4.75, 2.05, "torch.cat\n(dim=1)", fs=6.4, color=SUB, **mono)
    x0 = 5.7
    slab(ax, x0, yM, wd(64), 1.3, F_GREEN); slab(ax, x0 + wd(64), yM, wd(64), 1.3, F_BLUE)
    label(ax, x0 + wd(64), yM - 0.65 - 0.2, "(N,128,2H,2W)", fs=6.8, **mono)
    label(ax, x0 + wd(64), yM + 0.65 + 0.2, "결합", fs=7.2, color=SUB)
    xe = x0 + 2 * wd(64) + 0.12
    arrow(ax, (xe, yM), (xe + 1.5, yM), color=BLUE, lw=1.2)
    label(ax, xe + 0.75, yM + 0.38, "double_conv\n(128, 64)", fs=6.4, color=BLUE, **mono)
    tensor(xe + 1.65, yM, 64, 1.3, F_BLUE, "(N,64,2H,2W)", "출력")
    label(ax, 0.2, -1.3, "업샘플(64) + 인코더 특징(64) → 결합(128) → 합성곱 → 64",
          fs=7, ha="left", color=SUB)
    save(fig, "ch6_p_unet_skip")


# ───────────────────────── 업샘플링: 보간 vs 전치 합성곱 (값은 torch로 계산) ─────────────────────────
def fig_upsample():
    import torch
    import torch.nn.functional as F
    x = torch.tensor([[1., 2.], [3., 4.]]).view(1, 1, 2, 2)
    near = F.interpolate(x, scale_factor=2, mode="nearest")[0, 0].numpy()
    bil = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)[0, 0].numpy()
    K = torch.tensor([[1.0, 0.5], [0.5, 0.2]]).view(1, 1, 2, 2)
    tr = F.conv_transpose2d(x, K, stride=2)[0, 0].numpy()
    assert tr.shape == (4, 4)
    fig, axs = plt.subplots(1, 5, figsize=(11.0, 2.6), gridspec_kw={"width_ratios": [1, 2, 2, 1, 2]})
    cmap = plt.get_cmap("Blues")

    def grid(ax, M, title, vmax, fmt="{:.2f}"):
        n = M.shape[0]
        ax.set_xlim(-0.1, n + 0.1); ax.set_ylim(n + 0.1, -0.1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=8.4, fontweight="bold", pad=4)
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                ax.add_patch(Rectangle((j, i), 1, 1, fc=cmap(0.12 + 0.55 * v / vmax), ec="white", lw=1.0))
                t = fmt.format(v).rstrip("0").rstrip(".") if fmt == "{:.2f}" else fmt.format(v)
                ax.text(j + 0.5, i + 0.5, t, ha="center", va="center", fontsize=7.6, color=INK)
        return ax

    grid(axs[0], x[0, 0].numpy(), "(a) 입력 2×2", 4.0)
    grid(axs[1], near, "(b) 최근접 보간 (학습 없음)", 4.0)
    grid(axs[2], bil, "(c) 쌍선형 보간 (학습 없음)", 4.0)
    grid(axs[3], K[0, 0].numpy(), "(d) 커널 (학습됨)", 4.0)
    grid(axs[4], tr, "(e) 전치 합성곱 (k=2, s=2)", 4.0)
    for ax, t in ((axs[3], "가중치 W"), (axs[4], "출력 = 입력값 × W 를\n칸마다 배치")):
        ax.text(1.0 if ax is axs[3] else 2.0, 4.75 if ax is axs[4] else 2.85, t, fontsize=7, ha="center", va="top", color=SUB)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "ch6_p_upsample")


# ───────────────────────── YOLO: 30개 숫자 해부 ─────────────────────────
def fig_yolo_vector():
    fig, ax = canvas(11.0, 4.3, (0, 11.0), (-0.2, 4.1))
    S, g = 7, 3.3
    gx, gy = 0.2, 0.45
    cs = g / S
    ci, cj = 4, 3                                   # 강조할 칸(열, 행: 위에서부터)
    ax.add_patch(Rectangle((gx, gy), g, g, fc="#eef3f8", ec=LINE, lw=0.8))
    ax.add_patch(Rectangle((gx + ci * cs, gy + (S - 1 - cj) * cs), cs, cs, fc=F_ORANGE, ec=ORANGE, lw=1.5, zorder=3))
    for k in range(S + 1):
        ax.plot([gx, gx + g], [gy + k * cs] * 2, color=LINE, lw=0.5, zorder=2)
        ax.plot([gx + k * cs] * 2, [gy, gy + g], color=LINE, lw=0.5, zorder=2)
    label(ax, gx + g / 2, gy - 0.22, "7×7 격자 (S=7)", fs=8, color=SUB)
    label(ax, gx + g / 2, gy + g + 0.2, "출력 텐서 7×7×30 의 한 칸", fs=8.2, bold=True)
    # 벡터 막대
    bx, by, cw, bh = 4.55, 1.65, 0.2, 0.62
    cell_x = gx + (ci + 1) * cs
    cell_y = gy + (S - 1 - cj) * cs + cs / 2
    ax.plot([cell_x, bx], [cell_y, by + bh], color=ORANGE, lw=0.8, ls=":", zorder=1)
    ax.plot([cell_x, bx], [cell_y, by], color=ORANGE, lw=0.8, ls=":", zorder=1)
    names = ["x", "y", "w", "h", "c"]
    for i in range(30):
        if i < 5:
            fc, t = F_BLUE, names[i]
        elif i < 10:
            fc, t = F_PURPLE, names[i - 5]
        else:
            fc, t = F_GREEN, ""
        ax.add_patch(Rectangle((bx + i * cw, by), cw, bh, fc=fc, ec=LINE, lw=0.5, zorder=2))
        if t:
            label(ax, bx + i * cw + cw / 2, by + bh / 2, t, fs=7)
    for i, (a, b, txt, col) in enumerate([(0, 5, "상자 1", BLUE), (5, 10, "상자 2", PURPLE), (10, 30, "클래스 확률 20개", GREEN)]):
        x0, x1 = bx + a * cw, bx + b * cw
        ax.plot([x0 + 0.02, x1 - 0.02], [by - 0.14] * 2, color=col, lw=1.3)
        label(ax, (x0 + x1) / 2, by - 0.38, txt, fs=7.6, color=INK)
        label(ax, (x0 + x1) / 2, by - 0.66, f"{a + 1}–{b}번째 값", fs=6.6, color=SUB)
    label(ax, bx + 20 * cw, by + bh + 0.22, "P(클래스 | 물체) — 칸당 한 벌", fs=6.8, color=GREEN)
    # 계산 요약
    box(ax, 4.55, 2.95, 6.2, 0.95, "", fc="white", ec=LINE, r=0.05)
    label(ax, 7.65, 3.58, "5 + 5 + 20 = 30 개의 숫자", fs=8.4, bold=True)
    label(ax, 7.65, 3.22, "49칸 × 상자 2개 = 후보 상자 98개   (칸마다 클래스 확률은 하나)", fs=7.6, color=SUB)
    save(fig, "ch6_p_yolo_vector")


# ───────────────────────── YOLO: 두 갈래 결과의 결합 ─────────────────────────
def fig_yolo_flow():
    from matplotlib.patches import Ellipse
    rng = np.random.RandomState(7)
    objs = [("고양이", ORANGE, "e", (3.0, 4.9, 3.6, 3.2)), ("자동차", BLUE, "r", (7.5, 7.2, 2.8, 2.2)),
            ("강아지", GREEN, "e", (6.9, 2.8, 2.6, 2.2))]
    S, c = 7, 10 / 7
    fig, axs = plt.subplots(1, 4, figsize=(11.2, 3.2))
    ttl = ["(a) 입력 + 7×7 격자", "(b) 상자 98개 + 확신도", "(c) 클래스 확률 지도", "(d) 결합 후 NMS"]

    def base(ax, t):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 10, 10, fc="#eef3f8", ec=LINE, lw=0.6))
        ax.set_title(t, fontsize=8.6, fontweight="bold", pad=4)

    def draw_objs(ax, a=0.8):
        for nm, col, kind, (cx, cy, w, h) in objs:
            if kind == "e":
                ax.add_patch(Ellipse((cx, cy), w, h, fc="#b9c0c8", ec="none", alpha=a, zorder=2))
            else:
                ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.3",
                                            fc="#b9c0c8", ec="none", alpha=a, zorder=2))

    def grid(ax):
        for k in range(S + 1):
            ax.plot([0, 10], [k * c] * 2, color=LINE, lw=0.4, zorder=3)
            ax.plot([k * c] * 2, [0, 10], color=LINE, lw=0.4, zorder=3)

    for ax, t in zip(axs, ttl):
        base(ax, t); draw_objs(ax); grid(ax)
    # (b) 98개 상자: 칸마다 2개, 물체 근처일수록 확신도(두께)가 큼
    for i in range(S):
        for j in range(S):
            ccx, ccy = (i + 0.5) * c, (j + 0.5) * c
            near = max(np.exp(-((ccx - o[3][0]) ** 2 + (ccy - o[3][1]) ** 2) / (2 * (o[3][2] * 0.55) ** 2)) for o in objs)
            for _ in range(2):
                ow = 0.9 + rng.rand() * 1.6 + near * 1.0
                oh = 0.9 + rng.rand() * 1.6 + near * 1.0
                conf = near * (0.5 + 0.5 * rng.rand())
                hi = conf > 0.55
                axs[1].add_patch(Rectangle((ccx - ow / 2, ccy - oh / 2), ow, oh, fc="none",
                                           ec=RED if hi else GRAY, lw=0.25 + (2.0 * conf if hi else 0.5 * conf),
                                           alpha=0.85 if hi else 0.35, zorder=4 if hi else 3.5))
    # (c) 칸마다 가장 확률 높은 클래스를 색으로 표시
    for i in range(S):
        for j in range(S):
            ccx, ccy = (i + 0.5) * c, (j + 0.5) * c
            best, bv = None, 0
            for nm, col, kind, (cx, cy, w, h) in objs:
                v = np.exp(-((ccx - cx) ** 2 + (ccy - cy) ** 2) / (2 * (max(w, h) * 0.5) ** 2))
                if v > bv:
                    best, bv = col, v
            if bv > 0.45:
                axs[2].add_patch(Rectangle((i * c, j * c), c, c, fc=best, ec="none", alpha=0.25 + 0.5 * bv, zorder=2.5))
    # (d) 최종
    for nm, col, kind, (cx, cy, w, h) in objs:
        axs[3].add_patch(Rectangle((cx - w / 2 - 0.2, cy - h / 2 - 0.2), w + 0.4, h + 0.4, fc="none", ec=col, lw=1.8, zorder=5))
        axs[3].text(cx - w / 2 - 0.2, cy + h / 2 + 0.3, nm, fontsize=6.8, color="white", va="bottom", zorder=6,
                    bbox=dict(fc=col, ec="none", pad=1.3))
    fig.subplots_adjust(wspace=0.08)
    fig.text(0.5, -0.02, "확신도 × 클래스 확률 = 클래스별 점수  →  점수가 낮은 상자 제거  →  클래스별 NMS     (모식도)",
             ha="center", fontsize=8.4, color=SUB)
    save(fig, "ch6_p_yolo_flow")


# ───────────────────────── YOLO: 박스 복원 기준 (칸 기준 x,y / 이미지 기준 w,h) ─────────────────────────
def fig_yolo_decode():
    from matplotlib.patches import Ellipse
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})
    S, c = 7, 10 / 7
    col, row = 3, 3
    tx, ty, w, h = 0.55, 0.55, 0.46, 0.46
    cx, cy = (col + tx) * c, (row + ty) * c
    ax = axs[0]
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(-2.4, 10.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) 이미지 전체", fontsize=8.8, fontweight="bold", pad=4)
    ax.add_patch(Rectangle((0, 0), 10, 10, fc="#eef3f8", ec=LINE, lw=0.7))
    ax.add_patch(Ellipse((cx, 10 - cy), w * 10 * 0.95, h * 10 * 0.82, fc="#b9c0c8", ec="none"))
    for k in range(S + 1):
        ax.plot([0, 10], [k * c] * 2, color=LINE, lw=0.4); ax.plot([k * c] * 2, [0, 10], color=LINE, lw=0.4)
    ax.add_patch(Rectangle((col * c, 10 - (row + 1) * c), c, c, fc=F_ORANGE, ec=ORANGE, lw=1.4, zorder=3))
    bw, bh = w * 10, h * 10
    ax.add_patch(Rectangle((cx - bw / 2, 10 - cy - bh / 2), bw, bh, fc="none", ec=RED, lw=1.8, zorder=4))
    ax.plot([cx], [10 - cy], "o", color=RED, ms=4, zorder=5)
    # w 표시: 상자 폭 vs 이미지 폭
    ax.annotate("", xy=(cx - bw / 2, -0.6), xytext=(cx + bw / 2, -0.6), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1))
    label(ax, cx, -1.15, r"$w=0.46$  (이미지 너비의 46%)", fs=7.8, color=RED)
    ax.annotate("", xy=(0, -1.75), xytext=(10, -1.75), arrowprops=dict(arrowstyle="<->", color=SUB, lw=0.8))
    label(ax, 5, -2.2, "이미지 너비 = 1", fs=7, color=SUB)
    # (b) 칸 확대
    ax = axs[1]
    ax.set_xlim(-0.9, 5.3); ax.set_ylim(-1.2, 5.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) 담당 칸 확대", fontsize=8.8, fontweight="bold", pad=4)
    ax.add_patch(Rectangle((0, 0), 4.4, 4.4, fc=F_ORANGE, ec=ORANGE, lw=1.6))
    px, py = tx * 4.4, (1 - ty) * 4.4
    ax.plot([px], [py], "o", color=RED, ms=6, zorder=5)
    ax.plot([0, px], [py, py], color=BLUE, lw=1.2, ls="--"); ax.plot([px, px], [4.4, py], color=BLUE, lw=1.2, ls="--")
    arrow(ax, (0, 4.4 + 0.35), (px, 4.4 + 0.35), color=BLUE, lw=1.1, style="<|-|>", ms=6)
    label(ax, px / 2, 4.4 + 0.65, r"$t_x=0.55$", fs=8, color=BLUE)
    arrow(ax, (-0.35, 4.4), (-0.35, py), color=BLUE, lw=1.1, style="<|-|>", ms=6)
    label(ax, -0.62, (4.4 + py) / 2, r"$t_y$", fs=8, color=BLUE, rotation=90)
    label(ax, 2.2, -0.4, "칸의 너비·높이 = 1 (칸 기준)", fs=7.4, color=SUB)
    label(ax, 2.2, -0.85, "중심 위치는 칸 안의 비율", fs=7.4, color=SUB)
    fig.subplots_adjust(wspace=0.04)
    fig.text(0.5, -0.06, "$c_x$ = (열 + $t_x$) / S = (3 + 0.55) / 7 = 0.507  (이미지 비율)        $c_y$ 도 같은 방식        $w, h$ 는 이미지 기준",
             ha="center", fontsize=9, color=INK)
    save(fig, "ch6_p_yolo_decode")


# ───────────────────────── YOLO 계열 타임라인 ─────────────────────────
def fig_yolo_timeline():
    nodes = [  # (이름, 연도, 핵심어, 그룹)  그룹: R=원 저자 계보, U=Ultralytics, O=그 외
        ("YOLOv1", "2015", "격자·단일 회귀", "R"), ("YOLOv2", "2016", "앵커·배치 정규화", "R"),
        ("YOLOv3", "2018", "3개 스케일", "R"), ("YOLOv4", "2020", "CSP·학습 기법", "R"),
        ("YOLOv5", "2020", "PyTorch 구현", "U"), ("YOLOv6", "2022", "산업 배포", "O"),
        ("YOLOv7", "2022", "학습 기법 최적화", "R"), ("YOLOv8", "2023", "앵커 프리", "U"),
        ("YOLOv9", "2024", "PGI·GELAN", "R"), ("YOLOv10", "2024", "NMS 없는 학습", "O"),
        ("YOLO11", "2024", "백본·넥 개선", "U"), ("YOLO12", "2025", "어텐션 중심", "U"),
        ("YOLO26", "2026", "NMS 없는 추론", "U"),
    ]
    fc = {"R": (F_BLUE, BLUE), "U": (F_ORANGE, ORANGE), "O": (F_GREEN, GREEN)}
    n = len(nodes)
    fig, ax = canvas(12.2, 4.6, (0, 12.2), (-2.45, 2.15))
    x0, dx = 0.85, 0.88
    ax.plot([0.2, 12.0], [0, 0], color=LINE, lw=1.2, zorder=1)
    arrow(ax, (11.6, 0), (12.1, 0), color=LINE, lw=1.2, ms=8)
    bw, bh = 1.56, 0.98
    for i, (nm, yr, kw, g) in enumerate(nodes):
        x = x0 + i * dx
        up = (i % 2 == 0)
        yb = 0.62 if up else -0.62 - bh
        f, e = fc[g]
        ax.plot([x, x], [0, 0.62 if up else -0.62], color=e, lw=1.0, zorder=1)
        ax.plot([x], [0], "o", color=e, ms=4.5, zorder=3)
        box(ax, x - bw / 2, yb, bw, bh, "", fc=f, ec=e, lw=0.9, r=0.07)
        label(ax, x, yb + bh - 0.22, nm, fs=8.6, bold=True)
        label(ax, x, yb + bh - 0.47, yr, fs=7.2, color=SUB)
        label(ax, x, yb + 0.21, kw, fs=7.2)
    for k, (g, txt) in enumerate([("R", "Redmon·Farhadi → Bochkovskiy·Wang·Liao"), ("U", "Ultralytics"),
                                  ("O", "그 외 (YOLOv6 Meituan, YOLOv10 칭화대)")]):
        f, e = fc[g]
        xx = [0.5, 5.6, 7.6][k]
        box(ax, xx, -2.28, 0.3, 0.2, "", fc=f, ec=e, r=0.03)
        label(ax, xx + 0.4, -2.18, txt, fs=7.2, ha="left")
    save(fig, "ch6_p_yolo_timeline")


FIGS = {"unet_num": lambda: fig_unet(numbered=True), "yolo_timeline": fig_yolo_timeline, "yolo_vector": fig_yolo_vector, "yolo_flow": fig_yolo_flow, "yolo_decode": fig_yolo_decode,
        "unet": fig_unet, "tasks": fig_tasks, "transfer": fig_transfer, "detectors": fig_detectors,
        "iou": fig_iou, "nms": fig_nms, "yolo_grid": fig_yolo_grid, "yolov8": fig_yolov8,
        "unet_skip": fig_unet_skip, "upsample": fig_upsample}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
