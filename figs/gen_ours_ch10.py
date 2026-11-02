# -*- coding: utf-8 -*-
"""10·11주차(LLM·파인튜닝·RAG) 교재용 그림. 모두 직접 작성한 그림이다.
수치 그림(10-2, 10-5, 10-10, 10-12)의 값은 이 저장소 노트북을 실행해 얻은 실제 값이다."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from _ours import new_ax, heading, BRAND, CORAL, INDIGO, AMBER, ROSE, INK, SUB, LINE, TINT
from gen_ours_ch9 import rb, tx, arr, blend, save

C4 = {"brand": BRAND, "coral": CORAL, "indigo": INDIGO, "amber": AMBER, "rose": ROSE}


def mini_mask(ax, x, y, n, cell, causal, col):
    for i in range(n):
        for j in range(n):
            on = (j <= i) if causal else True
            ax.add_patch(Rectangle((x + j * cell, y - (i + 1) * cell), cell, cell,
                         fc=blend(col, 0.75) if on else "#f1f3f5", ec="white", lw=1.2, zorder=3))


# ============ 그림 10-1: 트랜스포머의 세 가지 구성 ============
def fig_three():
    fig, ax = new_ax(13.6, 6.6, (0, 27.2), (0, 13.2))
    heading(ax, 0.4, 12.4, "트랜스포머의 세 가지 구성")
    cols = [("인코더만", "마스크 없음 · 양방향", BRAND, "BERT 계열", "분류 · 개체명 인식 · 문장 임베딩"),
            ("디코더만", "인과 마스크 · 왼쪽→오른쪽", INDIGO, "GPT 계열", "문장 생성 · 대화 · 코드 생성"),
            ("인코더 + 디코더", "교차 어텐션으로 연결", CORAL, "원 논문 · T5 계열", "번역 · 요약")]
    for k, (name, sub, col, fam, task) in enumerate(cols):
        x0 = 0.6 + k * 8.9
        rb(ax, x0, 1.0, 8.2, 10.4, "#ffffff", col, lw=2.2, r=0.3, shadow=True)
        rb(ax, x0, 10.1, 8.2, 1.3, TINT["gray"], col, lw=2.2, r=0.3)
        tx(ax, x0 + 4.1, 11.0, name, fs=14, bold=True)
        tx(ax, x0 + 4.1, 10.45, sub, fs=10, color=SUB)
        if k == 0:
            rb(ax, x0 + 1.6, 6.2, 5.0, 2.3, TINT["brand"], col, lw=1.8)
            tx(ax, x0 + 4.1, 7.35, "인코더 × N", fs=12, bold=True)
            mini_mask(ax, x0 + 3.0, 5.7, 4, 0.55, False, col)
        elif k == 1:
            rb(ax, x0 + 1.6, 6.2, 5.0, 2.3, TINT["indigo"], col, lw=1.8)
            tx(ax, x0 + 4.1, 7.35, "디코더 × N", fs=12, bold=True)
            mini_mask(ax, x0 + 3.0, 5.7, 4, 0.55, True, col)
        else:
            rb(ax, x0 + 0.5, 6.7, 3.3, 1.8, TINT["brand"], BRAND, lw=1.8)
            tx(ax, x0 + 2.15, 7.6, "인코더", fs=12, bold=True)
            rb(ax, x0 + 4.4, 6.7, 3.3, 1.8, TINT["coral"], CORAL, lw=1.8)
            tx(ax, x0 + 6.05, 7.6, "디코더", fs=12, bold=True)
            arr(ax, (x0 + 3.85, 7.6), (x0 + 4.35, 7.6), CORAL, lw=2.4)
            tx(ax, x0 + 4.1, 6.15, "교차 어텐션", fs=10, color=CORAL, bold=True)
        arr(ax, (x0 + 4.1, 8.6), (x0 + 4.1, 9.9), col, lw=2.4)
        tx(ax, x0 + 4.1, 2.9, fam, fs=12.5, bold=True)
        tx(ax, x0 + 4.1, 2.1, task, fs=10, color=SUB)
    save(fig, "ch10_three_configs.png")


# ============ 그림 10-2: 자기회귀 생성 ============
def fig_autoregress():
    fig, ax = new_ax(13.8, 6.0, (0, 27.6), (0, 12.0))
    heading(ax, 0.4, 11.3, "자기회귀 생성: 다음 토큰 확률에서 하나를 고르고 입력에 붙입니다")
    rb(ax, 0.6, 7.2, 8.2, 2.2, TINT["gray"], LINE, lw=1.8)
    tx(ax, 4.7, 8.9, "입력 토큰", fs=11, color=SUB)
    for i, t in enumerate(["Art", "ificial", " intelligence", " will"]):
        rb(ax, 0.9 + i * 1.95, 7.5, 1.85, 0.85, TINT["brand"], BRAND, lw=1.4, r=0.1)
        tx(ax, 1.82 + i * 1.95, 7.92, t, fs=8.3, bold=True)
    rb(ax, 9.9, 7.2, 5.0, 2.2, TINT["indigo"], INDIGO, lw=2)
    tx(ax, 12.4, 8.75, "트랜스포머 디코더", fs=11.5, bold=True)
    tx(ax, 12.4, 8.1, "마지막 위치의 출력 → softmax", fs=9.5, color=SUB)
    arr(ax, (8.9, 8.3), (9.8, 8.3), SUB, lw=2.2)
    arr(ax, (15.0, 8.3), (16.3, 8.3), SUB, lw=2.2)
    names = [" be", " become", " also", " help", " soon"]
    ps = [0.156, 0.043, 0.032, 0.031, 0.031]
    tx(ax, 20.6, 9.95, "다음 토큰 확률 (상위 5개)", fs=11, color=SUB)
    for i, (n, p) in enumerate(zip(names, ps)):
        y = 9.0 - i * 0.72
        tx(ax, 18.3, y, n.strip(), fs=10.5, ha="right")
        ax.add_patch(Rectangle((18.5, y - 0.25), p / 0.156 * 3.8, 0.5, fc=blend(BRAND if i == 0 else INDIGO, 0.8 if i == 0 else 0.35), ec="none", zorder=3))
        tx(ax, 18.6 + p / 0.156 * 3.8, y, f"{p:.3f}", fs=9.5, color=SUB, ha="left")
    rb(ax, 24.7, 8.2, 2.3, 1.4, TINT["amber"], AMBER, lw=1.8)
    tx(ax, 25.85, 9.15, "선택", fs=10.5, bold=True)
    tx(ax, 25.85, 8.6, "예: be", fs=10, color=SUB)
    arr(ax, (23.75, 8.9), (24.65, 8.9), AMBER, lw=2.2)
    arr(ax, (25.85, 8.1), (25.85, 5.3), AMBER, lw=2.2)
    ax.plot([25.85, 4.7], [5.3, 5.3], color=AMBER, lw=2.2, zorder=3)
    arr(ax, (4.7, 5.3), (4.7, 7.1), AMBER, lw=2.2)
    tx(ax, 15.0, 4.6, "선택한 토큰을 입력 끝에 이어 붙이고 다시 계산합니다 (종료 토큰이나 길이 한도까지 반복)", fs=10.5, color=SUB)
    for i, t in enumerate(["Art", "ificial", " intelligence", " will", " be"]):
        rb(ax, 5.0 + i * 2.7, 1.6, 2.55, 1.0, TINT["amber"] if i == 4 else TINT["brand"], AMBER if i == 4 else BRAND, lw=1.4, r=0.1)
        tx(ax, 6.27 + i * 2.7, 2.1, t, fs=9.5, bold=True)
    tx(ax, 4.5, 2.1, "다음 입력", fs=10.5, color=SUB, ha="right")
    save(fig, "ch10_autoregressive.png")


# ============ 그림 10-3: LLM 학습의 세 단계 ============
def fig_stages():
    fig, ax = new_ax(13.6, 5.8, (0, 27.2), (0, 11.6))
    heading(ax, 0.4, 10.9, "LLM이 만들어지는 세 단계")
    st = [("① 사전학습", "대규모 일반 텍스트", "다음 토큰 예측", "기반(base) 모델", BRAND),
          ("② 지도 미세조정 (SFT)", "(지시, 응답) 쌍", "응답 토큰의 다음 토큰 예측", "지시를 따르는 모델", INDIGO),
          ("③ 선호 정렬", "사람이 고른 응답 쌍", "RLHF · DPO 등", "의도·규범에 맞춘 모델", CORAL)]
    for k, (t, d, l, o, col) in enumerate(st):
        x0 = 0.6 + k * 9.0
        rb(ax, x0, 3.4, 7.6, 6.2, "#ffffff", col, lw=2.2, r=0.3, shadow=True)
        rb(ax, x0, 8.4, 7.6, 1.2, TINT["gray"], col, lw=2.2, r=0.3)
        tx(ax, x0 + 3.8, 9.0, t, fs=12.5, bold=True)
        for j, (lab, val) in enumerate([("데이터", d), ("학습 목표", l), ("결과", o)]):
            tx(ax, x0 + 0.5, 7.4 - j * 1.4, lab, fs=9.5, color=SUB, ha="left")
            tx(ax, x0 + 7.2, 7.4 - j * 1.4, val, fs=10.5, bold=True, ha="right")
        if k < 2:
            arr(ax, (x0 + 7.7, 6.5), (x0 + 8.9, 6.5), SUB, lw=2.6)
    rb(ax, 0.6, 0.6, 25.8, 1.7, TINT["gray"], LINE, lw=1.6, r=0.25)
    tx(ax, 13.5, 1.45, "가중치를 바꾸는 단계는 ①~③입니다. 프롬프트 작성과 RAG는 가중치를 바꾸지 않고 입력만 바꿉니다.", fs=11, color=SUB)
    save(fig, "ch10_llm_stages.png")


# ============ 그림 10-4: zero/one/few-shot 프롬프트 ============
def fig_incontext():
    fig, ax = new_ax(13.6, 5.8, (0, 27.2), (0, 11.6))
    heading(ax, 0.4, 10.9, "in-context 학습: 예시를 프롬프트에 넣습니다 (가중치는 그대로)")
    rows = [("zero-shot", "예시 없음", [], BRAND),
            ("one-shot", "예시 1개", [("좋은 영화였다", "긍정")], INDIGO),
            ("few-shot", "예시 여러 개", [("좋은 영화였다", "긍정"), ("시간이 아까웠다", "부정"), ("배우들이 훌륭했다", "긍정")], CORAL)]
    for k, (name, sub, ex, col) in enumerate(rows):
        x0 = 0.6 + k * 9.0
        rb(ax, x0, 1.4, 7.8, 8.2, "#ffffff", col, lw=2.2, r=0.3, shadow=True)
        rb(ax, x0, 8.5, 7.8, 1.1, TINT["gray"], col, lw=2.2, r=0.3)
        tx(ax, x0 + 3.9, 9.05, f"{name}  ({sub})", fs=12, bold=True)
        y = 7.6
        tx(ax, x0 + 0.4, y, "다음 문장의 감정을 분류하시오.", fs=10, color=SUB, ha="left")
        y -= 1.0
        for a, b in ex:
            rb(ax, x0 + 0.35, y - 0.4, 7.1, 0.8, TINT["gray"], LINE, lw=1.2, r=0.1)
            tx(ax, x0 + 0.6, y, f"{a} → {b}", fs=10, ha="left")
            y -= 1.0
        rb(ax, x0 + 0.35, y - 0.4, 7.1, 0.8, TINT["amber"], AMBER, lw=1.6, r=0.1)
        tx(ax, x0 + 0.6, y, "이야기가 지루했다 → ?", fs=10, bold=True, ha="left")
    save(fig, "ch10_incontext.png")


# ============ 그림 10-5: temperature와 확률 분포 ============
def fig_temperature():
    toks = ["be", "become", "also", "help", "soon", "not", "continue", "eventually"]
    data = {"T = 0.5": [0.686, 0.052, 0.029, 0.028, 0.028, 0.026, 0.025, 0.021],
            "T = 1.0": [0.156, 0.043, 0.032, 0.031, 0.031, 0.030, 0.030, 0.027],
            "T = 1.5": [0.040, 0.017, 0.014, 0.014, 0.014, 0.014, 0.013, 0.013]}
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.6), sharey=False)
    cols = [INDIGO, BRAND, CORAL]
    for ax, (k, v), col in zip(axs, data.items(), cols):
        ax.bar(range(8), v, color=col, alpha=0.85, width=0.7)
        for i, p in enumerate(v):
            ax.text(i, p + 0.012, f"{p:.2f}", ha="center", fontsize=8.5, color=SUB)
        ax.set_title(k, fontsize=13, fontweight="bold", color=INK, pad=10)
        ax.set_xticks(range(8)); ax.set_xticklabels(toks, rotation=40, ha="right", fontsize=9.5)
        ax.set_ylim(0, 0.78)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
    axs[0].set_ylabel("다음 토큰 확률", fontsize=10.5)
    fig.suptitle("temperature에 따른 다음 토큰 확률 (GPT-2, 'Artificial intelligence will' 다음, 상위 8개)",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "ch10_temperature.png")


# ============ 그림 10-6: 학습 메모리 ============
def fig_memory():
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    parts = [("가중치 (FP16)", 14, INDIGO), ("기울기 (FP16)", 14, BRAND), ("FP32 가중치 사본 + Adam 모멘트 2개", 84, CORAL)]
    bottom = 0
    for lab, v, col in parts:
        ax.bar(0, v, bottom=bottom, color=col, alpha=0.85, width=0.55, edgecolor="white")
        ax.text(0.33, bottom + v / 2, f"{lab}\n{v} GB", va="center", fontsize=10, color=INK)
        bottom += v
    ax.text(0, bottom + 3, f"합계 약 {bottom} GB", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.set_xlim(-0.5, 1.9); ax.set_ylim(0, 130); ax.set_xticks([])
    ax.set_ylabel("메모리 (GB)", fontsize=10.5)
    ax.set_title("전체 파인튜닝 (70억 파라미터, 혼합 정밀도 Adam)", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax = axs[1]
    ax.bar(0, 14, color=INDIGO, alpha=0.85, width=0.55, edgecolor="white")
    ax.text(0.33, 7, "가중치 (FP16, 고정)\n14 GB", va="center", fontsize=10, color=INK)
    ax.bar(0, 3.2, bottom=14, color=AMBER, alpha=0.9, width=0.55, edgecolor="white")
    ax.annotate("LoRA 파라미터·기울기·상태\n(랭크에 따라 다르며 가중치보다 훨씬 작음)", xy=(0.28, 15.6), xytext=(0.6, 34), fontsize=9, color=INK, arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6))
    ax.set_xlim(-0.5, 2.6); ax.set_ylim(0, 130); ax.set_xticks([])
    ax.set_title("LoRA (같은 모델)", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False); ax.set_yticks([])
    fig.text(0.5, -0.03, "파라미터 수 × 바이트로 계산한 대략의 값입니다. 활성값(배치·시퀀스 길이에 비례)은 포함하지 않았습니다. LoRA 쪽 주황색 구간은 존재를 보이기 위한 표시이며 축척과 맞지 않습니다.",
             ha="center", fontsize=9, color=SUB)
    fig.tight_layout()
    save(fig, "ch10_memory.png")


# ============ 그림 10-7: PEFT 세 갈래 ============
def fig_peft_types():
    fig, ax = new_ax(13.6, 6.8, (0, 27.2), (0, 13.6))
    heading(ax, 0.4, 12.8, "PEFT의 세 갈래: 변화를 넣는 위치가 다릅니다")
    rb(ax, 0.8, 1.4, 13.0, 10.2, TINT["gray"], LINE, lw=2, r=0.35)
    tx(ax, 7.3, 11.1, "트랜스포머 블록 (원래 가중치는 고정)", fs=11.5, bold=True)
    rb(ax, 3.0, 8.3, 8.6, 1.6, TINT["brand"], BRAND, lw=1.8)
    tx(ax, 7.3, 9.1, "멀티 헤드 어텐션  ( W_Q  W_K  W_V  W_O )", fs=11)
    rb(ax, 3.0, 5.5, 8.6, 1.6, TINT["indigo"], INDIGO, lw=1.8)
    tx(ax, 7.3, 6.3, "피드포워드 네트워크", fs=11)
    rb(ax, 3.0, 2.3, 8.6, 1.4, "#ffffff", LINE, lw=1.4)
    tx(ax, 7.3, 3.0, "입력 표현", fs=11, color=SUB)
    arr(ax, (7.3, 3.8), (7.3, 5.4), SUB); arr(ax, (7.3, 7.2), (7.3, 8.2), SUB)
    # 마커
    rb(ax, 11.9, 8.5, 1.7, 1.2, TINT["amber"], AMBER, lw=2); tx(ax, 12.75, 9.1, "LoRA", fs=10.5, bold=True)
    rb(ax, 11.9, 5.7, 1.7, 1.2, TINT["coral"], CORAL, lw=2); tx(ax, 12.75, 6.3, "Adapter", fs=9.5, bold=True)
    rb(ax, 0.9, 9.9, 1.9, 1.0, TINT["rose"], ROSE, lw=2); tx(ax, 1.85, 10.4, "Prefix", fs=10, bold=True)
    arr(ax, (11.85, 9.1), (11.65, 9.1), AMBER, lw=2); arr(ax, (11.85, 6.3), (11.65, 6.3), CORAL, lw=2); arr(ax, (2.85, 10.1), (3.4, 9.6), ROSE, lw=2)
    rows = [("LoRA", AMBER, "가중치 변화량 ΔW를 낮은 랭크 행렬 곱 BA로 학습", "추론 때 W에 합칠 수 있음"),
            ("Adapter", CORAL, "층 안에 작은 병목 모듈을 삽입하고 그것만 학습", "모듈이 남아 약간의 추가 계산"),
            ("Prefix Tuning", ROSE, "각 층 어텐션의 키·값 앞에 학습 가능한 벡터를 붙임", "입력 길이(문맥)를 일부 차지")]
    for k, (n, col, d, e) in enumerate(rows):
        y0 = 9.4 - k * 3.6
        rb(ax, 15.0, y0 - 1.3, 11.7, 3.0, "#ffffff", col, lw=2, r=0.25, shadow=True)
        tx(ax, 15.4, y0 + 1.0, n, fs=12.5, bold=True, ha="left")
        tx(ax, 15.4, y0 + 0.1, d, fs=10, ha="left")
        tx(ax, 15.4, y0 - 0.7, e, fs=10, color=SUB, ha="left")
    save(fig, "ch10_peft_types.png")


# ============ 그림 10-8: LoRA 구조 ============
def fig_lora():
    fig, ax = new_ax(13.6, 6.6, (0, 27.2), (0, 13.2))
    heading(ax, 0.4, 12.4, "LoRA: 고정된 W 옆에 낮은 랭크 경로를 더합니다")
    rb(ax, 1.0, 5.4, 2.6, 1.8, TINT["gray"], LINE, lw=1.8)
    tx(ax, 2.3, 6.3, "입력 x", fs=12, bold=True); tx(ax, 2.3, 4.9, "(768)", fs=9.5, color=SUB)
    rb(ax, 7.0, 8.6, 7.0, 2.2, TINT["indigo"], INDIGO, lw=2.2)
    tx(ax, 10.5, 10.0, "W  (고정)", fs=13, bold=True); tx(ax, 10.5, 9.2, "768 × 2304 = 1,769,472", fs=10, color=SUB)
    rb(ax, 6.0, 2.4, 3.6, 2.0, TINT["amber"], AMBER, lw=2.2)
    tx(ax, 7.8, 3.7, "A  (학습)", fs=12, bold=True); tx(ax, 7.8, 2.9, "8 × 768", fs=10, color=SUB)
    rb(ax, 11.4, 2.4, 3.6, 2.0, TINT["amber"], AMBER, lw=2.2)
    tx(ax, 13.2, 3.7, "B  (학습)", fs=12, bold=True); tx(ax, 13.2, 2.9, "2304 × 8", fs=10, color=SUB)
    tx(ax, 9.9, 1.35, "랭크 r = 8: 두 행렬 합계 6,144 + 18,432 = 24,576", fs=10.5, color=SUB)
    arr(ax, (3.7, 6.7), (6.9, 9.5), INDIGO, lw=2.4); arr(ax, (3.7, 5.9), (5.9, 3.6), AMBER, lw=2.4)
    arr(ax, (9.7, 3.4), (11.3, 3.4), AMBER, lw=2.4)
    tx(ax, 10.5, 4.1, "× α/r", fs=10.5, color=AMBER, bold=True)
    ax.add_patch(plt.Circle((18.4, 6.4), 0.85, fc="#ffffff", ec=INK, lw=2, zorder=4))
    tx(ax, 18.4, 6.4, "+", fs=20, bold=True)
    arr(ax, (14.1, 9.5), (17.8, 6.9), INDIGO, lw=2.4); arr(ax, (15.1, 3.4), (18.0, 5.9), AMBER, lw=2.4)
    arr(ax, (19.3, 6.4), (21.3, 6.4), SUB, lw=2.4)
    rb(ax, 21.4, 5.4, 4.8, 2.0, TINT["gray"], LINE, lw=1.8)
    tx(ax, 23.8, 6.7, "출력 h", fs=12, bold=True); tx(ax, 23.8, 5.9, "(2304)", fs=9.5, color=SUB)
    tx(ax, 21.3, 10.6, "h = Wx + (α/r)·B A x", fs=13, bold=True, ha="center")
    tx(ax, 21.3, 9.6, "B = 0 으로 시작 → 학습 전에는 ΔW = 0", fs=10.5, color=SUB)
    save(fig, "ch10_lora.png")


# ============ 그림 10-9: LoRA 파라미터 수 ============
def fig_lora_count():
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    ax.bar([0, 1], [1769472, 24576], color=[INDIGO, AMBER], alpha=0.85, width=0.55)
    ax.text(0, 1769472 + 40000, "1,769,472", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(1, 24576 + 40000, "24,576\n(약 1.4%)", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["전체 학습\nW (768×2304)", "LoRA (r=8)\nA + B"], fontsize=10.5)
    ax.set_ylim(0, 2.1e6); ax.set_ylabel("학습 파라미터 수", fontsize=10.5)
    ax.set_title("c_attn 한 층", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(style="plain", axis="y")
    ax = axs[1]
    total, tr = 124734720, 294912
    ax.barh([0], [total - tr], color=TINT["gray"], edgecolor=LINE, height=0.5)
    ax.barh([0], [tr], left=[total - tr], color=AMBER, height=0.5)
    ax.text(total / 2, 0, "고정된 원래 가중치", ha="center", va="center", fontsize=11, color=SUB)
    ax.annotate("학습: 294,912개\n(0.24%)", xy=(total - tr / 2, 0.27), xytext=(total * 0.78, 0.78), fontsize=11, fontweight="bold",
                color=INK, arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2))
    ax.set_xlim(0, total * 1.02); ax.set_ylim(-0.5, 1.0); ax.set_yticks([])
    ax.set_xticks([0, total]); ax.set_xticklabels(["0", "124,734,720"], fontsize=9.5)
    ax.set_title("GPT-2 전체 (12개 층에 적용)", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save(fig, "ch10_lora_count.png")


# ============ 그림 10-10: LoRA 학습 손실 ============
def fig_loss():
    ep = [1, 10, 20, 30, 40, 50, 60, 70, 80]
    ls = [5.1888, 4.3559, 3.3794, 2.5407, 1.7466, 1.1576, 0.7330, 0.5937, 0.4872]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(ep, ls, color=INDIGO, lw=2.6, marker="o", ms=7, zorder=3)
    for e, l in zip(ep, ls):
        ax.text(e, l + 0.18, f"{l:.2f}", ha="center", fontsize=9, color=SUB)
    ax.set_xlabel("에폭", fontsize=10.5); ax.set_ylabel("학습 손실", fontsize=10.5)
    ax.set_ylim(0, 6); ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("LoRA 파인튜닝의 손실 곡선 (GPT-2, 12문장, 80에폭)", fontsize=12, fontweight="bold", color=INK, pad=10)
    fig.tight_layout()
    save(fig, "ch10_lora_loss.png")


# ============ 그림 10-11: RAG 전체 흐름 ============
def fig_rag_flow():
    fig, ax = new_ax(13.6, 7.4, (0, 27.2), (0, 14.8))
    heading(ax, 0.4, 14.0, "RAG의 전체 흐름")
    rb(ax, 0.6, 7.7, 26.0, 5.2, "#fbfcfd", LINE, lw=1.6, r=0.3)
    tx(ax, 1.0, 12.35, "색인 (미리 한 번)", fs=11.5, bold=True, color=BRAND, ha="left")
    steps = [("문서", "PDF · 웹 · 메모", TINT["gray"], LINE), ("청킹", "의미 단위로 분할", TINT["brand"], BRAND),
             ("임베딩", "청크 → 벡터", TINT["indigo"], INDIGO), ("벡터 저장소", "벡터 + 원문", TINT["amber"], AMBER)]
    for k, (a, b, fc, ec) in enumerate(steps):
        x0 = 1.2 + k * 6.3
        rb(ax, x0, 8.4, 4.6, 2.9, fc, ec, lw=1.9)
        tx(ax, x0 + 2.3, 10.4, a, fs=12.5, bold=True); tx(ax, x0 + 2.3, 9.2, b, fs=9.5, color=SUB)
        if k < 3: arr(ax, (x0 + 4.7, 9.85), (x0 + 6.2, 9.85), SUB, lw=2.4)
    rb(ax, 0.6, 0.5, 26.0, 6.6, "#fbfcfd", LINE, lw=1.6, r=0.3)
    tx(ax, 1.0, 6.5, "질의 (질문마다)", fs=11.5, bold=True, color=CORAL, ha="left")
    q = [("질문", "사용자 입력", TINT["gray"], LINE), ("질문 임베딩", "같은 임베딩 모델", TINT["indigo"], INDIGO),
         ("검색", "코사인 유사도 top-k", TINT["amber"], AMBER), ("프롬프트 구성", "지시 + 청크 + 질문", TINT["brand"], BRAND),
         ("LLM 생성", "문맥을 근거로 답변", TINT["coral"], CORAL)]
    for k, (a, b, fc, ec) in enumerate(q):
        x0 = 0.9 + k * 5.2
        rb(ax, x0, 1.4, 4.4, 3.6, fc, ec, lw=1.9)
        tx(ax, x0 + 2.2, 3.7, a, fs=11.5, bold=True); tx(ax, x0 + 2.2, 2.5, b, fs=8.8, color=SUB)
        if k < 4: arr(ax, (x0 + 4.45, 3.2), (x0 + 5.15, 3.2), SUB, lw=2.2)
    arr(ax, (19.4, 8.3), (15.9, 5.1), AMBER, lw=2.6, rad=0.0)
    tx(ax, 21.0, 6.9, "저장된 벡터와 비교", fs=9.5, color=AMBER, bold=True, ha="left")
    save(fig, "ch10_rag_flow.png")


# ============ 그림 10-12: 임베딩 모델에 따른 검색 점수 ============
def fig_retrieval():
    docs = ["오목 게임", "차를 선호", "6시 헬스장", "AI 프로젝트"]
    eng = [0.45, 0.70, 0.53, 0.37]
    mul = [0.11, 0.03, 0.31, 0.04]
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, v, ttl, col, res in [(axs[0], eng, "영어 중심 모델 (all-MiniLM-L6-v2)", CORAL, "1순위: '차를 선호' (오답)"),
                                 (axs[1], mul, "다국어 모델 (paraphrase-multilingual-MiniLM-L12-v2)", BRAND, "1순위: '6시 헬스장' (정답)")]:
        best = int(np.argmax(v))
        ax.bar(range(4), v, color=[col if i == best else "#c9ced6" for i in range(4)], width=0.6)
        for i, p in enumerate(v):
            ax.text(i, p + 0.015, f"{p:.2f}", ha="center", fontsize=10, color=INK)
        ax.set_xticks(range(4)); ax.set_xticklabels(docs, fontsize=10)
        ax.set_ylim(0, 0.85); ax.set_ylabel("코사인 유사도", fontsize=10)
        ax.set_title(ttl, fontsize=10.5, fontweight="bold", color=INK, pad=22)
        ax.text(0.5, 1.03, res, transform=ax.transAxes, ha="center", fontsize=10.5, color=col, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("질문 '나는 몇 시에 운동해?'에 대한 문서 4개의 유사도", fontsize=12.5, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "ch10_retrieval_scores.png")


# ============ 그림 10-13: 무엇을 쓸 것인가 ============
def fig_choice():
    fig, ax = new_ax(13.6, 6.6, (0, 27.2), (-0.8, 12.8))
    heading(ax, 0.4, 12.0, "목적에 따른 선택: 프롬프트, RAG, 파인튜닝")
    rb(ax, 9.8, 8.2, 7.6, 2.2, TINT["gray"], LINE, lw=1.8)
    tx(ax, 13.6, 9.3, "모델이 바꿔야 하는 것은?", fs=13, bold=True)
    opts = [("모르는 정보를 알려 주기", "최신 · 내부 문서 · 출처 필요", "RAG", BRAND, 1.2),
            ("행동·형식·말투를 바꾸기", "일정한 출력 형식, 도메인 문체", "파인튜닝", INDIGO, 9.8),
            ("간단한 과제 지시", "예시 몇 개로 충분한 경우", "프롬프트(in-context)", AMBER, 18.4)]
    for lab, sub, res, col, x0 in opts:
        rb(ax, x0, 3.2, 7.6, 3.2, "#ffffff", col, lw=2.2, r=0.25, shadow=True)
        tx(ax, x0 + 3.8, 5.4, lab, fs=11.5, bold=True); tx(ax, x0 + 3.8, 4.4, sub, fs=9.5, color=SUB)
        rb(ax, x0 + 0.9, 0.8, 5.8, 1.5, TINT["gray"], col, lw=2, r=0.3)
        tx(ax, x0 + 3.8, 1.55, res, fs=12.5, bold=True)
        arr(ax, (x0 + 3.8, 3.1), (x0 + 3.8, 2.4), col, lw=2.2)
    arr(ax, (11.8, 8.1), (5.0, 6.5), BRAND, lw=2.2); arr(ax, (13.6, 8.1), (13.6, 6.5), INDIGO, lw=2.2); arr(ax, (15.4, 8.1), (22.2, 6.5), AMBER, lw=2.2)
    tx(ax, 13.6, -0.35, "하나만 고르는 것이 아닙니다. 둘 다 필요하면 함께 사용합니다.", fs=10, color=SUB)
    save(fig, "ch10_choice.png")


if __name__ == "__main__":
    for f in (fig_three, fig_autoregress, fig_stages, fig_incontext, fig_temperature, fig_memory, fig_peft_types,
              fig_lora, fig_lora_count, fig_loss, fig_rag_flow, fig_retrieval, fig_choice):
        f()
