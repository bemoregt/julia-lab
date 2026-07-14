module Saliency

include("FRFT.jl")
using .FRFT
using ImageFiltering: imfilter, Kernel
using Statistics: quantile

export frft_saliency

"""
    frft_saliency(img_gray; a=0.9, b=0.9, sigma=8.0)

Compute a 2D-FRFT-domain phase-spectrum saliency map, following the
"phase spectrum saliency" idea (Hou & Zhang 2007; Guo, Ma & Zhang 2008
PQFT) generalized from the ordinary/quaternion Fourier transform to the
*fractional* Fourier transform (as in Niu, Chen, Qi, Guo 2015 and its
fabric-defect-detection descendant, Di, Long & Liang 2020):

  1. Take the 2D-FRFT of the (grayscale) image with orders `(a,b)`.
  2. Keep only the *phase* of the spectrum (discard magnitude -> unit
     modulus), which is where most of a scene's "surprising" structure
     lives.
  3. Invert the 2D-FRFT (orders `(-a,-b)`) of that phase-only spectrum.
  4. Take the squared magnitude of the reconstruction and smooth it with a
     Gaussian -> the saliency map. Regions that are *not* well predicted by
     the dominant periodic/self-similar structure of the image (e.g. a
     fabric defect breaking an otherwise regular weave pattern) light up.

`a=b=1` reduces this to the classical FFT-based phase-spectrum saliency
(PFT). The paper's fabric-defect method uses `a=b=0.9` and a quaternion
image (color+intensity+edge); here we operate on a single (grayscale)
channel for clarity -- see `frft_saliency_color` for a simple per-channel
extension.
"""
function frft_saliency(img_gray::AbstractMatrix{<:Real}; a::Real=0.9, b::Real=0.9,
                        sigma::Real=8.0, clip_pct::Real=0.99)
    x = ComplexF64.(img_gray)
    F = frft2d(x, a, b)
    phase_only = exp.(1im .* angle.(F))
    g = frft2d(phase_only, -a, -b)
    sal = abs2.(g)
    sal = imfilter(sal, Kernel.gaussian(sigma))

    # Robust (percentile) normalization: the abrupt, non-periodic image
    # border routinely produces one or two extreme corner/edge pixels in
    # phase-only reconstructions (a well-known artifact of FFT/FRFT-domain
    # saliency methods). A plain min-max normalization lets such an outlier
    # swamp the [0,1] range and erase the genuine saliency contrast, so we
    # clip at the `clip_pct` quantile before rescaling.
    lo = minimum(sal)
    hi = quantile(vec(sal), clip_pct)
    sal = clamp.(sal, lo, hi)
    return hi > lo ? (sal .- lo) ./ (hi - lo) : sal
end

"""
    frft_saliency_color(img_rgb; a=0.9, b=0.9, sigma=8.0)

Applies `frft_saliency` independently to each of the R, G, B channels (plus
the grayscale intensity) and combines them by taking the pointwise maximum,
as a lightweight stand-in for the paper's full quaternion (color+intensity+
edge) representation.
"""
function frft_saliency_color(r::AbstractMatrix{<:Real}, g::AbstractMatrix{<:Real},
                              b_::AbstractMatrix{<:Real}; a::Real=0.9, b::Real=0.9,
                              sigma::Real=8.0)
    intensity = (r .+ g .+ b_) ./ 3
    maps = [frft_saliency(c; a=a, b=b, sigma=sigma) for c in (r, g, b_, intensity)]
    return reduce((x, y) -> max.(x, y), maps)
end

end # module
