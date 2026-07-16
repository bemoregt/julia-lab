# Julia Global Filter Network (GFNet)

A from-scratch Julia implementation of the core mechanism from **Global Filter
Networks for Image Classification** (Rao, Zhao, Zhu, Lu, Zhou — *NeurIPS
2021*), [arXiv:2107.00645](https://arxiv.org/abs/2107.00645).

GFNet replaces the self-attention layer of a Vision Transformer with a
**global filter layer**: a 2D FFT, an element-wise multiplication with a
learnable frequency-domain filter, and a 2D inverse FFT. This mixes
information across *all* spatial tokens with `O(L log L)` complexity instead
of the `O(L²)` complexity of self-attention or spatial MLPs (where `L = H×W`
is the number of tokens).

## Algorithm

Given token features `x ∈ R^{H×W×D}`, the global filter layer computes:

```
X  = F_r[x]        # 2D real FFT              (eq. 3.4 / 3.7)
X̃  = K ⊙ X          # element-wise (Hadamard) product with a learnable
                     # complex filter K ∈ C^{H×Ŵ×D}, Ŵ = W÷2+1   (eq. 3.5)
x' = F_r^{-1}[X̃]    # 2D inverse real FFT                        (eq. 3.6)
```

Because a real signal's DFT is conjugate symmetric, only the half-spectrum
`rfft2` (`Ŵ = W÷2+1`) is needed to represent the full information — this is
exactly `Algorithm 1` in the paper and halves both compute and parameter
count relative to a full complex FFT.

A full GFNet block is `x = x + GlobalFilter(LN(x)); x = x + FFN(LN(x))`,
stacked after a ViT-style non-overlapping patch embedding, followed by global
average pooling and a linear classification head (Figure 1 of the paper).

The paper also proves (Appendix A) that the global filter is mathematically
equivalent to a **depthwise global circular convolution** in the spatial
domain — just computed in `O(L log L)` via FFT instead of `O(L²)` via direct
convolution.

## What this script demonstrates

`global_filter_network.jl` implements and verifies the architecture end to
end, without any deep learning framework (only `FFTW.jl` for the transforms):

1. **Identity-filter round-trip check** — with `K = 1`, `rfft2 → ⊙ → irfft2`
   exactly reconstructs the input (verifies the FFT plumbing).
2. **Global filter ↔ circular convolution equivalence** (Appendix A) — the
   FFT-domain path is checked numerically against a brute-force depthwise
   circular convolution computed directly in the spatial domain.
3. **Complexity benchmark** — measures wall-clock time of the global filter
   layer (`O(L log L)`) against a dense spatial-mixing MLP (`O(L²)`) as the
   token grid grows, reproducing the trend in the paper's Table 1.
4. **Frequency-domain filtering on a real image** — applies hand-crafted
   low-pass/high-pass global filters (built directly in the `(H, Ŵ)`
   half-spectrum layout used by the layer) to a test image, visualizing what
   a "global filter" does to spatial structure.
5. **End-to-end mini GFNet forward pass** — patch embedding → 4×
   `GFNetBlock` → global average pooling → linear head, run on a real image
   with randomly initialized (untrained) weights, to confirm the full
   architecture pipeline is shape-correct and runnable.

## Usage

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. global_filter_network.jl
```

## Example result

```
[1] 항등 필터(K=1) 왕복 재구성 오차
  max|x - GlobalFilter(x; K=1)| = 6.94e-16

[2] 전역 필터(FFT 경로) ↔ 전역 원형 합성곱(공간 경로) 동치성 검증
  max|FFT 경로 - 원형합성곱 경로| = 2.66e-15

[3] 계산 복잡도 비교: Global Filter O(L log L)  vs  Spatial MLP O(L²)
  H×W       tokens L  GlobalFilter(ms)    SpatialMLP(ms)
  8×8       64        0.023               0.004
  16×16     256       0.047               0.086
  32×32     1024      0.127               0.606
  64×64     4096      0.526               15.511
  512×512   262144    190.462             OOM (혼합행렬 이론상 549.8GB 필요)

[4] 메모리 점유 비교: Global Filter 파라미터 O(HWD)  vs  Spatial MLP 파라미터 O(H²W²)
  H×W     GF params   GF weight   GF/call alloc  MLP params      MLP weight  MLP/call alloc
  8×8     1280        10.0KB      49.1KB         4096            32.0KB  8.1KB
  16×16   4608        36.0KB      225.1KB        65536           512.0KB  32.1KB
  32×32   17408       136.0KB     705.1KB        1048576         8.0MB  128.1KB
  64×64   67584       528.0KB     2.56MB         16777216        128.0MB  512.1KB
  512×512 4210688     32.12MB     160.5MB        68719476736     512.0GB (이론값, 미할당)  N/A
  -> 64×64(D=16) 기준 Spatial MLP 가중치가 Global Filter보다 248.2배 더 많은 메모리 사용
  -> 512×512(D=16, 실제 샘플 이미지 해상도) 기준 Spatial MLP 혼합행렬은 512.0GB가 필요해
     이 시스템(16GB RAM)에서 실제 할당 자체가 불가능함 (반면 Global Filter는 32.12MB면 충분)

[5] 실제 영상에 대한 전역 필터(저역/고역) 시각화
  이미지 크기(H×W)      = (512, 512) (D=1 채널)
  1회 순전파 소요시간    = 6.295ms  (실제 512×512 해상도, rfft2->K⊙->irfft2)
  1회 순전파 런타임 할당 = 10.06MB
  edge energy(원본)     = 308.48
  edge energy(저역필터)  = 55.29
  edge energy(고역필터)  = 164.73

[6] 미니 GFNet 순전파 (patch embedding -> 4×GFNetBlock -> GAP -> Linear)
  입력 이미지 크기: (64, 64)  패치 크기: 8×8 -> 토큰 격자 8×8, D=32, depth=4
  출력 로짓(10-way, 미학습 가중치): [-0.0261, -0.3862, -0.019, 0.2847, 0.2885, -0.0554, -0.2947, -0.0459, -0.0996, 0.429]
```

At a 64×64 token grid, the global filter layer is already ~30× faster than
the equivalent spatial-mixing MLP and its learnable-weight memory footprint
(528KB) is **~248× smaller** than the spatial MLP's mixing matrix (128MB) —
because the filter has `O(HWD)` parameters versus `O(H²W²)` for a full
token-mixing matrix. Note the *runtime* allocation-per-call (`GF/call alloc`)
is actually higher for the global filter than the MLP matmul at small sizes,
since each FFT call allocates temporary plan/complex-buffer memory; this is a
fixed-size working-set cost, not a parameter-count cost, and is amortizable
by reusing FFTW plans (not done in this minimal reference implementation).

**At the sample image's native 512×512 resolution**, the gap becomes
qualitative rather than quantitative: a spatial-MLP-style `L×L` token-mixing
matrix (`L = 512×512 = 262144` tokens) would need **~550GB** just to hold its
weights — infeasible on a 16GB machine, so the benchmark skips actually
allocating it (guarded by `max_mlp_bytes`) and reports the theoretical size
instead. The global filter layer, by contrast, runs the real 512×512 image
through `rfft2 → K⊙ → irfft2` directly in **~6-190ms** (depending on channel
count) using **32MB or less** of parameter memory. This reproduces the
paper's own stated motivation: "the self-attention model quickly runs out of
memory when feature resolution exceeds 56²" (Section 4.3) — GFNet's
log-linear complexity is precisely what makes native high-resolution
processing possible at all.

| Source | Low-pass global filter | High-pass global filter |
|---|---|---|
| ![source](gfnet_source.png) | ![lowpass](gfnet_filter_lowpass.png) | ![highpass](gfnet_filter_highpass.png) |

## API

```julia
GlobalFilterLayer(H, W, D)           # random learnable filter, or
GlobalFilterLayer(K)                 # from a given (H, W÷2+1, D) complex filter
layer(x)                             # applies rfft2 -> K⊙ -> irfft2 to x ∈ R^{H×W×D}

circular_conv2_depthwise(x, kernel)  # reference O(H²W²D) depthwise circular convolution

GFNetBlock(H, W, D, hidden)          # LN -> GlobalFilterLayer -> residual -> LN -> FFN -> residual
GFNet(; img_size, patch, D, hidden, depth, num_classes)  # full model
model(img)                           # patch embed -> blocks -> GAP -> linear head -> logits

benchmark_complexity(sides; D, reps) # O(L log L) vs O(L²) timing comparison
benchmark_memory(sides; D)           # parameter-memory (O(HWD) vs O(H²W²)) and per-call allocation comparison
```

## Dependencies

- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)
- [Images.jl](https://github.com/JuliaImages/Images.jl)
- [TestImages.jl](https://github.com/JuliaImages/TestImages.jl)
- [ImageIO.jl](https://github.com/JuliaIO/ImageIO.jl)
- `Random`, `Statistics` (standard library)

## Reference

- Y. Rao, W. Zhao, Z. Zhu, J. Lu, J. Zhou, "Global Filter Networks for Image
  Classification," *NeurIPS 2021*. [arXiv:2107.00645](https://arxiv.org/abs/2107.00645)
- Official PyTorch implementation: <https://github.com/raoyongming/GFNet>
