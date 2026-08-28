# -*- coding: utf-8 -*-
"""5장(CNN) 개념 그림. 합성 이미지를 numpy로 만들어 사용."""
import numpy as np
import matplotlib.pyplot as plt
from _style import save

rng = np.random.default_rng(5)


def synth_image(n=72):
    """도형이 있는 합성 흑백 이미지."""
    yy, xx = np.mgrid[0:n, 0:n]
    img = 0.15 + 0.0*xx
    circle = (xx - 24)**2 + (yy - 26)**2 < 14**2
    img[circle] = 0.9
    img[44:62, 40:62] = 0.6                 # 사각형
    tri = (yy > 44) & (yy < 64) & (np.abs(xx - 20) < (yy - 44) * 0.7)
    img[tri] = 0.35                         # 삼각형
    img += 0.03 * rng.standard_normal((n, n))
    return np.clip(img, 0, 1)


def conv2d(img, k):
    kh, kw = k.shape; ph, pw = kh // 2, kw // 2
    p = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(p[i:i+kh, j:j+kw] * k)
    return out


# 5.1 합성곱 엣지 검출
def fig_convolution():
    img = synth_image()
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    Ky = Kx.T
    gx, gy = conv2d(img, Kx), conv2d(img, Ky)
    edge = np.sqrt(gx**2 + gy**2)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for ax in axes: ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("(a) 원본 이미지")
    axes[1].imshow(Kx, cmap="RdBu", vmin=-2, vmax=2)
    axes[1].set_title("(b) 수직 모서리 커널")
    for (i, j), v in np.ndenumerate(Kx):
        axes[1].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=12)
    axes[2].imshow(edge, cmap="magma"); axes[2].set_title("(c) 합성곱 결과(특징 맵)")
    save(fig, "ch5_convolution.png")


# 5.2 데이터 증강
def fig_augmentation():
    img = synth_image()
    hflip = img[:, ::-1]
    rot = np.rot90(img)
    bright = np.clip(img * 1.5, 0, 1)
    sub = img[10:46, 8:44]
    crop = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)   # 확대(자르기+줌)
    panels = [(img, "원본"), (hflip, "좌우 뒤집기"), (rot, "90° 회전"),
              (bright, "밝기 증가"), (crop, "무작위 자르기")]
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.7), constrained_layout=True)
    for ax, (im, t) in zip(axes, panels):
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=11)
    save(fig, "ch5_augmentation.png")


# 5.3 최대 풀링 다운샘플
def fig_pooling():
    rng2 = np.random.default_rng(11)
    fm = rng2.random((8, 8))
    pooled = fm.reshape(4, 2, 4, 2).max(axis=(1, 3))       # 2x2 max pool
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    for ax in axes: ax.grid(False)
    im0 = axes[0].imshow(fm, cmap="viridis")
    axes[0].set_title("(a) 입력 특징 맵 (8×8)")
    im1 = axes[1].imshow(pooled, cmap="viridis")
    axes[1].set_title("(b) 2×2 최대 풀링 후 (4×4)")
    for ax, mat in [(axes[0], fm), (axes[1], pooled)]:
        ax.set_xticks(range(mat.shape[1])); ax.set_yticks(range(mat.shape[0]))
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white", fontsize=7)
    save(fig, "ch5_pooling.png")


if __name__ == "__main__":
    fig_convolution()
    fig_augmentation()
    fig_pooling()
    print("done ch5")
