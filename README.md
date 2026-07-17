# julia_fft

Julia/Flux re-implementations of two "classify directly from JPEG DCT
coefficients" papers, plus benchmark scripts to test pipeline
correctness and measure inference/training speed on a sample image.

These are from-scratch reproductions of the *architectural ideas*
described in each paper, written independently in Julia. They are not
a port of the papers' original code (Keras/TensorFlow with a modified
`libjpeg`), are not trained on the papers' datasets, and do not
reproduce the papers' accuracy numbers. See "Limitations" below.

## Papers reproduced

1. **DCT-CompCNN** — Bulla Rajesh, Mohammed Javed, Ratnesh, Shubham
   Srivastava, *"DCT-CompCNN: A Novel Image Classification Network
   Using JPEG Compressed DCT Coefficients"*, IEEE CICT 2019.
   ([arXiv:1907.11503](https://arxiv.org/abs/1907.11503))

2. **Faster Neural Networks Straight from JPEG** — Lionel Gueguen,
   Alex Sergeev, Ben Kadlec, Rosanne Liu, Jason Yosinski, NeurIPS 2018.
   ([PDF](https://yosinski.com/media/papers/Gueguen__2018__Faster_neural_networks_straight_from_JPEG__.pdf))

## Requirements

- Julia 1.12+ (developed against 1.12.6)
- macOS on Apple Silicon for the MPS/Metal benchmarks (optional —
  scripts fall back to CPU-only if `Metal.jl` isn't functional)

All Julia dependencies are pinned in `Project.toml` / `Manifest.toml`.
Install them once with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Project layout

```
src/
  DCTCompCNN.jl          # Rajesh/Javed et al. reproduction
  FastJPEGResNet.jl       # Gueguen et al. reproduction
scripts/
  run_test.jl              # DCT-CompCNN: pipeline check + inference speed
  run_fastjpeg_test.jl      # FastJPEGResNet: pipeline check + inference speed (CPU + MPS)
  run_fastjpeg_train_test.jl  # FastJPEGResNet: training-step speed (CPU + MPS)
```

## 1. DCT-CompCNN

`src/DCTCompCNN.jl` implements the paper's Fig. 1–4 pipeline:
RGB → YCbCr → 4:2:0 chroma subsampling → non-decimating 8×8 block DCT
(FFTW) → optional JPEG-table quantization → channel-size matching by
either upsampling the chroma DCT maps or downsampling the luma DCT
map → concatenation into a `(32,32,3)` tensor → an 8-layer CNN
(4× conv+maxpool, 3→64→128→256→512 channels, FC 2048→128→256→512→
num_classes), matching the paper's Fig. 4 architecture.

Run:

```sh
julia --project=. scripts/run_test.jl
```

The script:
- Verifies the DCT/quantize/dequantize/IDCT round trip is numerically
  correct (PSNR check).
- Builds all four input variants from the paper (quantized/
  un-quantized × up-/down-sampling) from a sample image
  (`TestImages.testimage("mandrill")`).
- Times input-tensor construction and CNN forward-pass (inference)
  for each variant against a plain-RGB baseline.

## 2. Faster Neural Networks Straight from JPEG

`src/FastJPEGResNet.jl` implements the paper's *decimating* block DCT
(each 8×8 pixel block → 64 channels at 1/8 spatial resolution, i.e.
a stride-8 depthwise convolution with the DCT-II basis as fixed
filters — Sec. 2.2 of the paper) plus two ResNet-50-based networks:

- `build_resnet50_rgb`: standard ResNet-50 (7×7/2 stem + maxpool +
  conv2_x…conv5_x).
- `build_resnet50_dct_upsampling`: the paper's "UpSampling" variant —
  stem and conv2_x are removed entirely; the DCT tensor
  (`Y: 28×28×64`, `Cb/Cr: 14×14×64` upsampled ×2 to `28×28×64`,
  concatenated to `28×28×192`) feeds directly into conv3_x with its
  first block's stride reduced from 2 to 1, then standard conv4_x,
  conv5_x.

Run inference benchmark (CPU, and MPS/Metal if available):

```sh
julia --project=. scripts/run_fastjpeg_test.jl
```

Run training-step benchmark (forward + backward + Adam update; CPU
and MPS/Metal if available):

```sh
julia --project=. scripts/run_fastjpeg_train_test.jl
```

### Measured results (Apple M1, sample image `mandrill`, batch 4)

| Metric | CPU | MPS (Metal) |
|---|---|---|
| Inference speedup, DCT vs RGB | 1.88x | 1.75x (1.87x at batch 32) |
| Training-step speedup, DCT vs RGB | 1.44x | 1.61x |
| MPS vs CPU speedup, RGB model | — | 1.74x (infer), 1.51x (train) |
| MPS vs CPU speedup, DCT model | — | 1.61x (infer), 1.69x (train) |

For reference, the paper's own GPU numbers (NVIDIA Pascal, batch
1024, Table 1) report **1.90x** inference speedup for this same
"UpSampling" architecture, and **1.77x** for their best
accuracy-matched variant ("Late-Concat-RFA-Thinner", not
implemented here). The paper does not publish a training-speed
ratio.

## Limitations (read before citing any number from this repo)

- **No accuracy is measured.** All networks use randomly-initialized
  weights; there is no labelled dataset (ImageNet / CIFAR-10 / Dogs
  vs Cats) or trained checkpoint in this repo. Only pipeline
  correctness (shapes, a round-trip PSNR check, a forward pass that
  runs) and step/inference timing are tested.
- **DCT extraction is not real JPEG partial decoding.** Both modules
  compute the forward DCT in Julia (via FFTW / a precomputed DCT-II
  basis matrix) starting from an already-decoded pixel array. Neither
  uses `libjpeg`'s raw-coefficient API (`jpeg_read_coefficients`), so
  savings from skipping the *actual* JPEG entropy/Huffman decode and
  IDCT are not captured — only the downstream neural-network compute
  savings are.
- **MPS ≠ the paper's hardware.** The paper benchmarks an NVIDIA
  Pascal GPU with cuDNN across 128 GPUs; this repo benchmarks a single
  Apple Silicon (M1) GPU via Metal.jl. Absolute FPS numbers are not
  comparable between the two; only the qualitative, hardware-agnostic
  speedup from skipping high-resolution early layers is expected to
  transfer, and that is what was checked.
- **Training benchmark has no data-augmentation cost.** Real JPEG-
  domain training pipelines must decode → augment (random crop/flip)
  → re-encode to DCT every step, since augmentation isn't directly
  expressible on DCT coefficients. This repo's training benchmark
  reuses one fixed pre-built DCT tensor per step, so the measured DCT-
  path training speedup is optimistic relative to a real training
  loop.
- Small sample size: all timings come from a single 224×224 (or
  32×32) sample image repeated into a batch, on one machine, over a
  modest number of runs (reported as min/median to reduce GC-pause
  noise). Treat all ratios as order-of-magnitude sanity checks, not
  publication-grade benchmarks.
