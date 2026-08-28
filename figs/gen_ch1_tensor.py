# -*- coding: utf-8 -*-
"""1장: 텐서 랭크(0D~6D) 큐브 시각화."""
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from _style import save

FRONT, TOP, RIGHT = "#dbe4f0", "#eef3f9", "#c2cee0"
EC, GRID = "#6b7c93", "#9fb0c6"
DX, DY = 0.52, 0.34   # 깊이 단위 벡터(iso)


def cuboid(ax, ox, oy, nx, ny, nz, u):
    """nx×ny×nz 로 분할된 등축 직육면체를 (ox,oy)에 그린다."""
    W, H = nx*u, ny*u
    Dx, Dy = nz*u*DX, nz*u*DY
    top = [(ox, oy+H), (ox+W, oy+H), (ox+W+Dx, oy+H+Dy), (ox+Dx, oy+H+Dy)]
    right = [(ox+W, oy), (ox+W+Dx, oy+Dy), (ox+W+Dx, oy+H+Dy), (ox+W, oy+H)]
    front = [(ox, oy), (ox+W, oy), (ox+W, oy+H), (ox, oy+H)]
    ax.add_patch(Polygon(top, fc=TOP, ec=EC, lw=1.0))
    ax.add_patch(Polygon(right, fc=RIGHT, ec=EC, lw=1.0))
    ax.add_patch(Polygon(front, fc=FRONT, ec=EC, lw=1.0))
    # 격자선
    for i in range(1, nx):
        ax.plot([ox+i*u, ox+i*u], [oy, oy+H], color=GRID, lw=0.6)
        ax.plot([ox+i*u, ox+i*u+Dx], [oy+H, oy+H+Dy], color=GRID, lw=0.6)
    for j in range(1, ny):
        ax.plot([ox, ox+W], [oy+j*u, oy+j*u], color=GRID, lw=0.6)
        ax.plot([ox+W, ox+W+Dx], [oy+j*u, oy+j*u+Dy], color=GRID, lw=0.6)
    for k in range(1, nz):
        ax.plot([ox+k*u*DX, ox+W+k*u*DX], [oy+H+k*u*DY, oy+H+k*u*DY], color=GRID, lw=0.6)
        ax.plot([ox+W+k*u*DX, ox+W+k*u*DX], [oy+k*u*DY, oy+H+k*u*DY], color=GRID, lw=0.6)


def block_extent(nx, ny, nz, u):
    return (nx*u + nz*u*DX, ny*u + nz*u*DY)


def meta(ax, ox, oy, mx, my, mz, u, gap):
    """(3,3,3) 블록을 mx×my×mz 로 배열(고차원 텐서)."""
    bw, bh = block_extent(3, 3, 3, u)
    sx, sy = bw + gap, bh + gap
    mdx, mdy = bw*0.45, bh*0.45      # 블록 z(깊이) 오프셋
    order = []
    for k in range(mz):
        for j in range(my-1, -1, -1):
            for i in range(mx):
                order.append((k, j, i))
    for (k, j, i) in order:
        bx = ox + i*sx + k*mdx
        by = oy + j*sy + k*mdy
        cuboid(ax, bx, by, 3, 3, 3, u)


def fig_tensor_ranks():
    fig, ax = plt.subplots(figsize=(12.2, 8.4), constrained_layout=True)
    ax.set_xlim(0, 17); ax.set_ylim(0, 12); ax.set_aspect("equal"); ax.axis("off")

    def label(cx, y, name, rank, shape):
        ax.text(cx, y, name, ha="center", fontsize=10.5, fontweight="bold")
        ax.text(cx, y-0.55, f"Rank: {rank}", ha="center", fontsize=9.5, color="#334")
        ax.text(cx, y-1.1, f"Shape: {shape}", ha="center", fontsize=9.5, color="#334")

    # --- 윗줄: 0D~3D (y≈8.4) ---
    U = 0.62; oy1 = 8.4
    cuboid(ax, 0.7, oy1, 1, 1, 1, U);      label(1.2, 7.4, "0D Tensor(Scalar)", 0, "()")
    cuboid(ax, 2.9, oy1, 3, 1, 1, U);      label(3.9, 7.4, "1D Tensor(Vector)", 1, "(3, )")
    cuboid(ax, 6.4, oy1, 3, 3, 1, U);      label(7.5, 7.4, "2D Tensor(Matrix)", 2, "(3, 3)")
    cuboid(ax, 10.9, oy1, 3, 3, 3, U);     label(12.2, 7.4, "3D Tensor", 3, "(3, 3, 3)")

    # --- 아랫줄: 4D~6D (아래에서 위로 자람, y0≈1.9) ---
    oy2 = 1.9
    meta(ax, 0.8, oy2, 3, 1, 1, 0.30, 0.35);   label(3.3, 1.1, "4D Tensor", 4, "(3, 3, 3, 3)")
    meta(ax, 6.4, oy2, 3, 3, 1, 0.24, 0.30);   label(8.4, 0.7, "5D Tensor", 5, "(3, 3, 3, 3, 3)")
    meta(ax, 11.9, oy2, 3, 3, 3, 0.185, 0.22); label(14.2, 0.7, "6D Tensor", 6, "(3, 3, 3, 3, 3, 3)")

    ax.set_title("텐서의 랭크(차원)와 모양", fontsize=14, fontweight="bold")
    save(fig, "ch1_tensor_ranks.png")


if __name__ == "__main__":
    fig_tensor_ranks()
    print("done tensor")
