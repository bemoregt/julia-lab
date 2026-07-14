# 2D Fractional Fourier Transform (FRFT) Saliency in Julia

A from-scratch Julia implementation of the discrete fractional Fourier
transform (DFRFT) and a 2D-FRFT phase-spectrum saliency detector, inspired
by:

- Niu, Chen, Qi & Guo, *"Image registration based on Fractional Fourier
  Transform"*, Optik 126(23), 2015.
- Di, Long & Liang, *"Fabric Defect Detection Based on Illumination
  Correction and Visual Salient Features"*, Sensors 20(18), 2020
  (PMC7571081) — uses a 2D-FRFT-domain phase spectrum (with `α=β=0.9`) as
  part of a quaternion-image saliency pipeline for detecting fabric defects.

This repo implements the core saliency mechanism (2D-FRFT → phase-only
reconstruction → saliency map) on a single (grayscale) channel, as a
self-contained, testable building block.

## How it works

1. **Discrete fractional Fourier transform** (`src/FRFT.jl`)
   The ordinary DFT matrix `F` (normalized, unitary) satisfies `F^4 = I`, so
   its eigenvalues are 4th roots of unity `exp(-i*pi*k/2)`, `k=0,1,2,3`,
   each with high multiplicity (~N/4 each). Diagonalizing `F = V·D·V'` lets
   us *define* a fractional power `F^a`. Naively giving every eigenvector in
   a degenerate class the same phase is unitary and matches `F` exactly at
   integer orders, but produces large, spatially non-smooth artifacts at
   fractional orders (verified and rejected during development — see
   `FRFT.jl` comments). Instead, each eigenvector is assigned its own
   integer Hermite order `n = 0,...,N-1` (`n mod 4 == k`, so integer-order
   behavior is untouched) by ranking eigenvectors within each class using
   the eigenvalues of the **Dickinson & Steiglitz (1982) commuting matrix**
   `S` — a real symmetric, tridiagonal-with-corner matrix that provably
   commutes with `F` but (unlike `F`) has simple eigenvalues, recovering the
   smooth discrete-Hermite-Gauss ordering used in the standard DFRFT
   literature (Pei & Yeh 1997; Candan, Kutay & Ozaktas 2000). The final
   transform is
   ```
   F^a := sum_n exp(-i*pi*a*n/2) * v_n v_n^H
   ```
   which is unitary for every real `a`, and satisfies `F^0=I`, `F^1=F`,
   `F^2` = exact time reversal, `F^3=F^{-1}`, and `F^a·F^b = F^{a+b}`
   exactly. The 2D transform (`frft2d`) applies this separably along rows
   (order `a`) and columns (order `b`).

2. **Phase-spectrum saliency** (`src/Saliency.jl`)
   Generalizes the classical PFT / spectral-residual saliency idea (Hou &
   Zhang 2007; Guo, Ma & Zhang 2008) from the ordinary/quaternion Fourier
   transform to the FRFT:
   - Take the 2D-FRFT of the image at orders `(a,b)`.
   - Discard the magnitude, keep only the phase (`exp(i·angle(F))`).
   - Invert the 2D-FRFT at orders `(-a,-b)`.
   - Square the magnitude of the reconstruction, smooth with a Gaussian,
     and normalize (percentile-clipped, to stay robust to the occasional
     extreme boundary pixel that these Fourier-domain methods can produce).

   Regions that are *not* well predicted by the image's dominant
   periodic/self-similar structure — e.g. a defect breaking an otherwise
   regular weave — light up in the resulting map.

## Project layout

```
src/FRFT.jl        1D/2D discrete fractional Fourier transform
src/Saliency.jl     2D-FRFT phase-spectrum saliency map
test_frft.jl        Algebraic correctness checks for FRFT.jl
main.jl              Demo: synthetic fabric-defect image + a natural image
output/              PNGs written by main.jl
```

## Requirements

Julia 1.12 (developed/tested on this version). Dependencies are pinned in
`Project.toml` / `Manifest.toml`:
`FFTW`, `Images`, `ImageFiltering`, `ImageIO`, `TestImages`, `FileIO`,
`PNGFiles`, `ColorTypes`.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Run the correctness tests for the FRFT implementation:

```bash
julia --project=. test_frft.jl
```

Run the saliency demo (writes PNGs to `output/`):

```bash
julia --project=. main.jl
```

Use it on your own image:

```julia
include("src/Saliency.jl")
using .Saliency
using Images

img = Float64.(Gray.(load("your_image.png")))
sal = frft_saliency(img; a=0.9, b=0.9, sigma=6.0)   # sal ∈ [0,1], same size as img
save("saliency.png", Gray.(sal))
```

## Test results

`test_frft.jl` verifies, to machine precision (~1e-14):

- `F^0 = I` (identity)
- `F^2` = exact discrete time-reversal (parity) operator
- Energy (Parseval) preservation for arbitrary fractional orders
- Inverse property: `F^{-a}(F^a(x)) = x`
- Additivity: `F^a(F^b(x)) = F^{a+b}(x)`

`main.jl` builds a synthetic 256×256 grid-pattern "fabric" image with an
embedded circular defect (mimicking the paper's grid/star/dot pattern test
images) and compares FRFT-domain saliency (`a=b=0.9`, the paper's setting)
against the classical FFT-domain phase-spectrum saliency (`a=b=1.0`):

| Method | Detected pixels | IoU vs. ground truth |
|---|---|---|
| FRFT (a=b=0.9) | 1769 | 0.347 |
| FFT (a=b=1.0)  | 1217 | 0.504 |

Both methods clearly localize the defect (see `output/06_montage_fabric.png`:
input \| ground truth \| FRFT saliency \| FFT saliency). On this
*perfectly regular* synthetic grid, plain FFT-based phase saliency actually
scores higher: at `a=0.9` the transform is not purely frequency-domain, so
some of the grid's spatial-domain periodicity leaks into the saliency map
as background texture, whereas `a=1` (pure phase spectrum) fully suppresses
strictly periodic structure. This is reported as-is rather than tuned away
— the paper's reported advantage of FRFT over FFT likely shows up more on
real (noisy, imperfectly periodic) fabric photographs and with the full
quaternion (color + intensity + edge) representation, neither of which this
single-channel grayscale demo implements.

A natural-image sanity check (`output/09_montage_natural.png`, the standard
"mandrill" test image) shows the saliency map reasonably highlighting the
eyes, consistent with typical phase-spectrum saliency behavior in the
literature.

## Known simplifications vs. the paper

- Operates on a single grayscale channel; the paper builds a **quaternion**
  image from color + intensity + Canny edge features. A simple per-channel
  extension is sketched in `frft_saliency_color` (`src/Saliency.jl`) but is
  not a true quaternion Fourier transform.
- No illumination correction (MS-BOX filter + gamma correction) or L0
  gradient-minimization background-texture removal stage is implemented —
  this repo isolates and tests only the FRFT-saliency core.
