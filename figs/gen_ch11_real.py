# -*- coding: utf-8 -*-
"""12·13주차 교재용 그림 중 스테이블 디퓨전으로 직접 생성한 실제 이미지 그림.
실행: PYTHONPATH=<diffusers 설치 경로> python gen_ch11_real.py   (GPU 필요, 모델 최초 다운로드 약 5GB)
이미지는 모두 이 스크립트가 만든 것이며, 수치는 실행 결과를 출력한다."""
import time, os
import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["mathtext.fontset"] = "dejavusans"
OUT = os.path.dirname(os.path.abspath(__file__))
INK = "#1f2937"

device = torch.device("cuda")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
pipe.set_progress_bar_config(disable=True)

BASE = "a red apple on a wooden table"


def gen(prompt, seed=0, **kw):
    g = torch.Generator(device).manual_seed(seed)
    return pipe(prompt, generator=g, **kw).images[0]


def grid(images, titles, fname, ncols, size=3.1, title_fs=11):
    nrows = (len(images) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows + 0.55 * nrows))
    axs = np.atleast_1d(axs).flatten()
    for a in axs:
        a.axis("off")
    for a, im, t in zip(axs, images, titles):
        a.imshow(im)
        a.set_title(t, fontsize=title_fs, color=INK, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", fname, flush=True)


print("== 모델 구성 ==")
cnt = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
print(f"unet {cnt(pipe.unet):.1f}M  vae {cnt(pipe.vae):.1f}M  text_encoder {cnt(pipe.text_encoder):.1f}M")
print("scheduler", type(pipe.scheduler).__name__, "| vae scaling", pipe.vae.config.scaling_factor,
      "| cross_attention_dim", pipe.unet.config.cross_attention_dim, "| unet in_channels", pipe.unet.config.in_channels,
      "| tokenizer max", pipe.tokenizer.model_max_length, "| text_encoder", type(pipe.text_encoder).__name__,
      "| hidden", pipe.text_encoder.config.hidden_size, flush=True)

# ---- 기본 이미지와 시간 ----
t0 = time.time(); base = gen(BASE, 0); torch.cuda.synchronize(); print(f"기본 생성(50단계) {time.time()-t0:.1f}s")
t0 = time.time(); base = gen(BASE, 0); torch.cuda.synchronize(); print(f"기본 생성(50단계, 2회째) {time.time()-t0:.1f}s")
print("최대 GPU 메모리(GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
base.save(os.path.join(OUT, "ch11_base_apple.png"))

# ---- 그림: 전방향 과정 (실제 이미지에 노이즈를 섞음) ----
T = 1000
beta = np.linspace(1e-4, 0.02, T); abar = np.cumprod(1 - beta)
x0 = np.asarray(base).astype(np.float32) / 127.5 - 1
rng = np.random.default_rng(0)
eps = rng.standard_normal(x0.shape).astype(np.float32)
ts = [0, 100, 250, 500, 750, 999]
ims, tl = [], []
for t in ts:
    if t == 0:
        xt = x0
        tl.append("원본 $x_0$")
    else:
        a = abar[t - 1]
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
        tl.append(f"$t={t}$  (신호 {np.sqrt(a):.2f})")
    ims.append(np.clip((xt + 1) / 2, 0, 1))
grid(ims, tl, "ch11_forward_real.png", 6, size=2.6, title_fs=10)

# ---- 그림: 생성 중간 단계 (잠재를 디코딩) ----
want = {0: "1단계 후", 4: "5단계 후", 9: "10단계 후", 19: "20단계 후", 34: "35단계 후", 49: "50단계 후"}
cap = {}

def cb(p, i, t, kw):
    if i in want:
        lat = kw["latents"]
        with torch.no_grad():
            im = p.vae.decode(lat / p.vae.config.scaling_factor).sample
        im = ((im[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
        cap[i] = im
    return kw

g = torch.Generator(device).manual_seed(0)
pipe(BASE, generator=g, callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
ks = sorted(cap)
grid([cap[k] for k in ks], [want[k] for k in ks], "ch11_denoise_steps.png", 6, size=2.6, title_fs=10)

# ---- 그림: guidance scale ----
gs = [1.0, 3.0, 7.5, 15.0]
grid([gen(BASE, 0, guidance_scale=w) for w in gs], [f"guidance_scale = {w:g}" for w in gs], "ch11_guidance.png", 4, size=2.9)

# ---- 그림: 추론 단계 수 ----
st = [5, 10, 20, 50]
grid([gen(BASE, 0, num_inference_steps=n) for n in st], [f"{n}단계" for n in st], "ch11_steps.png", 4, size=2.9)

# ---- 그림: 시드 ----
a0 = gen(BASE, 0); b0 = gen(BASE, 0); c1 = gen(BASE, 1); d2 = gen(BASE, 2)
same = np.array_equal(np.asarray(a0), np.asarray(b0))
print("seed 0 두 번 생성한 이미지가 완전히 같은가:", same, "| seed0 vs seed1 평균 픽셀 차이:",
      round(float(np.abs(np.asarray(a0).astype(float) - np.asarray(c1).astype(float)).mean()), 1), flush=True)
grid([a0, b0, c1, d2], ["seed 0", "seed 0 (다시)", "seed 1", "seed 2"], "ch11_seed.png", 4, size=2.9)

# ---- 그림: 부정 프롬프트 ----
P = "a cozy reading room with warm lighting"
N = "blurry, distorted, low quality"
grid([gen(P, 3), gen(P, 3, negative_prompt=N)], ["부정 프롬프트 없음", "negative_prompt = 'blurry, distorted, low quality'"],
     "ch11_negative.png", 2, size=4.6, title_fs=10)

# ---- 그림: 스타일 프롬프트 ----
S1 = "a small cottage in a quiet forest"
styles = [S1, S1 + ", watercolor painting", S1 + ", cinematic lighting, 3d rendering"]
grid([gen(s, 5) for s in styles], ["기본", "+ watercolor painting", "+ cinematic lighting, 3d rendering"], "ch11_style.png", 3, size=3.3, title_fs=10)

# ---- 그림: 프롬프트 민감도 ----
pm = [BASE, BASE + ", soft studio lighting", BASE + ", dramatic lighting", BASE + ", oil painting"]
grid([gen(p, 0) for p in pm], ["기본", "+ soft studio lighting", "+ dramatic lighting", "+ oil painting"], "ch11_prompt.png", 4, size=2.9, title_fs=10)

# ---- 그림: VAE 압축과 복원 ----
x = torch.from_numpy(np.asarray(base).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device, torch.float16)
with torch.no_grad():
    z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    rec = pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample
print("입력", tuple(x.shape), "-> 잠재", tuple(z.shape), "| 원소 수", x.numel(), "->", z.numel(), "| 비율", x.numel() / z.numel(), flush=True)
rec_im = ((rec[0].float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
orig = np.asarray(base).astype(np.float32) / 255
mse = float(((rec_im - orig) ** 2).mean()); psnr = 10 * np.log10(1 / mse)
print(f"VAE 복원 PSNR {psnr:.1f} dB", flush=True)
zz = z[0].float().cpu().numpy()
ims = [orig] + [zz[c] for c in range(4)] + [rec_im]
fig, axs = plt.subplots(1, 6, figsize=(15.6, 3.3))
titles = ["원본 512×512×3", "잠재 채널 1", "잠재 채널 2", "잠재 채널 3", "잠재 채널 4", f"복원 (PSNR {psnr:.1f} dB)"]
for k, (a, im, t) in enumerate(zip(axs, ims, titles)):
    a.axis("off")
    a.imshow(im, cmap=None if k in (0, 5) else "viridis")
    a.set_title(t, fontsize=10, color=INK, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "ch11_vae_real.png"), dpi=130, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved ch11_vae_real.png")

# ---- 텍스트 인코더 출력 크기 ----
tok = pipe.tokenizer(BASE, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
with torch.no_grad():
    emb = pipe.text_encoder(tok.input_ids.to(device))[0]
print("프롬프트 토큰 수(패딩 전):", int(tok.attention_mask.sum()), "| 텍스트 임베딩 모양:", tuple(emb.shape), flush=True)
print("토큰:", pipe.tokenizer.convert_ids_to_tokens(tok.input_ids[0][:int(tok.attention_mask.sum())]))
print("ALL DONE")
