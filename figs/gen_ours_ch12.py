# -*- coding: utf-8 -*-
"""13주차(프롬프트 엔지니어링·RAG) 교재용 그림. 모두 직접 작성한 도식이며 색을 쓰지 않는다."""
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.patches import Rectangle
from _ours import new_ax, heading, INK, SUB
from gen_ours_ch9 import rb, tx, arr, save

mpl.rcParams["mathtext.fontset"] = "dejavusans"
MONO = "#3a3f45"
WHITE = "#ffffff"
GRAY = "#f1f3f5"


# ============ 그림 12-1: RAG의 전체 흐름 ============
def fig_rag():
    fig, ax = new_ax(14.0, 6.9, (0, 28.0), (1.2, 14.8))
    heading(ax, 0.4, 14.0, "RAG의 전체 흐름: 미리 색인하고, 질문이 올 때 검색해서 프롬프트에 넣습니다")
    # 위: 색인 (미리 한 번)
    tx(ax, 0.6, 11.7, "색인 (미리 한 번)", fs=12, bold=True, color=INK, ha="left")
    idx = [("문서 모음", "개인 메모, 사내 문서"), ("청킹", "작은 조각으로 나눔"),
           ("임베딩", "조각마다 벡터로 변환"), ("벡터 저장소", "벡터와 원문을 저장")]
    w = 5.6
    xs = [0.6 + 6.9 * k for k in range(4)]
    for k, (a, b) in enumerate(idx):
        rb(ax, xs[k], 8.0, w, 3.0, WHITE, MONO, lw=2.0, shadow=True)
        tx(ax, xs[k] + w / 2, 9.9, a, fs=12, bold=True)
        tx(ax, xs[k] + w / 2, 8.8, b, fs=9.5, color=SUB)
        if k < 3:
            arr(ax, (xs[k] + w + 0.05, 9.5), (xs[k + 1] - 0.05, 9.5), MONO, lw=2.2)
    # 아래: 질의 (질문마다)
    tx(ax, 0.6, 6.2, "질의 (질문마다)", fs=12, bold=True, color=INK, ha="left")
    q = [("질문", "사용자의 입력"), ("질문 임베딩", "같은 모델로 벡터화"),
         ("유사도 검색", "가까운 조각 k개 선택"), ("프롬프트 구성", "질문 + 검색한 문맥"),
         ("LLM 생성", "문맥을 근거로 답변")]
    w2 = 4.7
    xs2 = [0.6 + 5.5 * k for k in range(5)]
    for k, (a, b) in enumerate(q):
        rb(ax, xs2[k], 2.4, w2, 3.0, WHITE if k != 4 else GRAY, MONO, lw=2.0, shadow=True)
        tx(ax, xs2[k] + w2 / 2, 4.3, a, fs=11.5, bold=True)
        tx(ax, xs2[k] + w2 / 2, 3.2, b, fs=9, color=SUB)
        if k < 4:
            arr(ax, (xs2[k] + w2 + 0.05, 3.9), (xs2[k + 1] - 0.05, 3.9), MONO, lw=2.2)
    # 저장소 -> 유사도 검색 연결
    arr(ax, (xs[3] + w / 2, 7.9), (xs2[2] + w2 / 2, 5.5), MONO, lw=2.0, rad=-0.15)
    tx(ax, 24.0, 7.0, "저장된 벡터와 비교", fs=9.5, color=SUB, ha="left")
    save(fig, "ch12_rag.png")


# ============ 그림 12-2: 자기 일관성 (실제 실행 결과 사용) ============
def fig_selfcons():
    answers = [40, 40, 40, 40, 40, 7, 40, 40]        # 코드 12-3의 시드 0 결과
    fig, ax = new_ax(14.0, 5.8, (0, 28.0), (0, 11.6))
    heading(ax, 0.4, 10.8, "자기 일관성: 여러 번 샘플링한 풀이에서 가장 많이 나온 답을 고릅니다")
    rb(ax, 0.6, 6.5, 4.4, 2.8, WHITE, MONO, lw=2.0, shadow=True)
    tx(ax, 2.8, 8.2, "CoT 프롬프트", fs=11.5, bold=True)
    tx(ax, 2.8, 7.2, "같은 문제를 8번 샘플링", fs=8.5, color=SUB)
    arr(ax, (5.1, 7.9), (6.1, 7.9), MONO, lw=2.2)
    for k, a in enumerate(answers):
        x = 6.3 + 2.7 * k
        win = (a == 40)
        rb(ax, x, 6.8, 2.4, 2.2, GRAY if win else WHITE, MONO, lw=2.4 if win else 1.4)
        tx(ax, x + 1.2, 8.1, f"{a}", fs=13, bold=True)
        tx(ax, x + 1.2, 7.3, f"경로 {k + 1}", fs=7.5, color=SUB)
    ax.plot([6.3, 27.6], [6.4, 6.4], color=MONO, lw=1.6)
    arr(ax, (16.95, 6.4), (16.95, 5.2), MONO, lw=2.2)
    rb(ax, 11.4, 2.2, 11.1, 3.0, WHITE, MONO, lw=2.0, shadow=True)
    tx(ax, 16.95, 4.3, "다수결", fs=12, bold=True)
    tx(ax, 16.95, 3.1, "7은 1번, 40은 7번 나왔습니다", fs=9.5, color=SUB)
    arr(ax, (22.6, 3.7), (23.5, 3.7), MONO, lw=2.2)
    rb(ax, 23.6, 2.2, 4.0, 3.0, GRAY, MONO, lw=2.4, shadow=True)
    tx(ax, 25.6, 4.3, "답: 40", fs=12.5, bold=True)
    tx(ax, 25.6, 3.1, "7/8 득표", fs=9.5, color=SUB)
    tx(ax, 14.0, 0.8, "온도 0.7, top-p 0.9, 시드 0으로 실제 실행한 결과입니다. 7은 풀이가 아니라 FINAL: 40 뒤에 이어진 글의 숫자를 읽은 것입니다.", fs=9.5, color=INK)
    save(fig, "ch12_selfcons.png")


if __name__ == "__main__":
    fig_rag(); fig_selfcons()
