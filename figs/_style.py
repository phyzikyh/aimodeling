# -*- coding: utf-8 -*-
"""여러 장의 그림 생성 스크립트가 공유하는 스타일·헬퍼."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#4a5568",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

C_BLUE = "#2c7be5"
C_RED = "#e5533c"
C_GREEN = "#2f9e44"
C_PURPLE = "#8957e5"
C_ORANGE = "#e8912d"
C_GRAY = "#6b7280"

FIGDIR = os.path.dirname(__file__)


def save(fig, name):
    p = os.path.join(FIGDIR, name)
    fig.savefig(p)
    plt.close(fig)
    print("saved", name, os.path.getsize(p), "bytes")
