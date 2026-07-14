module FNOModel

using Flux
using FFTW
using LinearAlgebra

export SpectralConv2d, PointwiseConv, FNOBlock, FNOSeg

# ---------------------------------------------------------------------------
# All layers operate on (C, H, W, N) arrays: channels, height, width, batch.
# This is the key to resolution invariance: nothing in any layer's shape
# depends on H or W except through the *fixed* number of retained Fourier
# modes (modes1, modes2), so the exact same trained weights can be applied
# to inputs of any spatial size >= modes.
# ---------------------------------------------------------------------------

"""
    SpectralConv2d(cin, cout, modes1, modes2)

The core FNO layer (Li et al., 2020, "Fourier Neural Operator for
Parametric PDEs"): 2D FFT, multiply only the lowest `modes1 x modes2`
frequencies by a learnable complex channel-mixing weight, zero out
everything else, inverse FFT. Implemented without any in-place array
mutation (`cat`/`zeros`/broadcasting only) so it is compatible with
Zygote's reverse-mode autodiff.
"""
struct SpectralConv2d
    weight::Array{ComplexF64,4}   # (Cout, Cin, modes1, modes2)
    modes1::Int
    modes2::Int
end
Flux.@layer SpectralConv2d

function SpectralConv2d(cin::Int, cout::Int, modes1::Int, modes2::Int)
    scale = 1.0 / (cin * cout)
    w = scale .* randn(ComplexF64, cout, cin, modes1, modes2)
    return SpectralConv2d(w, modes1, modes2)
end

function (m::SpectralConv2d)(x::AbstractArray{<:Real,4})
    Cin, H, W, N = size(x)
    Cout = size(m.weight, 1)
    m1, m2 = m.modes1, m.modes2
    @assert H >= m1 && W >= m2 "input resolution ($H,$W) smaller than retained modes ($m1,$m2)"

    X = fft(x, (2, 3))
    Xlow = X[:, 1:m1, 1:m2, :]                                  # (Cin,m1,m2,N)
    Wr = reshape(m.weight, Cout, Cin, m1, m2, 1)
    Xr = reshape(Xlow, 1, Cin, m1, m2, N)
    Ylow = dropdims(sum(Wr .* Xr, dims=2), dims=2)                # (Cout,m1,m2,N)

    top = cat(Ylow, zeros(ComplexF64, Cout, m1, W - m2, N), dims=3)
    full = cat(top, zeros(ComplexF64, Cout, H - m1, W, N), dims=2)

    y = real.(ifft(full, (2, 3)))
    return y
end

"""
    PointwiseConv(cin, cout)

A 1x1 convolution (per-pixel linear channel mixing), implemented as a
`Dense` layer applied identically at every spatial location -- also
resolution-independent by construction.
"""
struct PointwiseConv
    dense::Flux.Dense
end
Flux.@layer PointwiseConv

function PointwiseConv(cin::Int, cout::Int)
    PointwiseConv(Flux.f64(Flux.Dense(cin => cout)))
end

function (m::PointwiseConv)(x::AbstractArray{<:Real,4})
    C, H, W, N = size(x)
    xf = reshape(x, C, H * W * N)
    yf = m.dense(xf)
    Cout = size(yf, 1)
    return reshape(yf, Cout, H, W, N)
end

"""
    FNOBlock(width, modes1, modes2)

One FNO block: global spectral-domain convolution (SpectralConv2d) plus a
local pointwise (1x1 conv) skip connection, combined and passed through a
nonlinearity -- exactly the residual structure from the original FNO paper.
"""
struct FNOBlock
    spectral::SpectralConv2d
    pointwise::PointwiseConv
end
Flux.@layer FNOBlock

FNOBlock(width::Int, modes1::Int, modes2::Int) =
    FNOBlock(SpectralConv2d(width, width, modes1, modes2), PointwiseConv(width, width))

(m::FNOBlock)(x) = gelu.(m.spectral(x) .+ m.pointwise(x))

"""
    FNOSeg(; cin=1, width=20, modes1=12, modes2=12, depth=4, cout=1)

Resolution-invariant segmentation network: lift the input to `width`
channels, apply `depth` FNO blocks, then project down to `cout` output
segmentation channels (sigmoid activation for binary masks). Structurally
mirrors the FNO-for-segmentation idea in Liu, Veldhuis & Brune (MIUA 2024)
and the general FNO architecture of Li et al. (2020).
"""
struct FNOSeg
    lift::PointwiseConv
    blocks::Vector{FNOBlock}
    proj1::PointwiseConv
    proj2::PointwiseConv
end
Flux.@layer FNOSeg

function FNOSeg(; cin::Int=1, width::Int=20, modes1::Int=12, modes2::Int=12,
                 depth::Int=4, cout::Int=1)
    lift = PointwiseConv(cin, width)
    blocks = [FNOBlock(width, modes1, modes2) for _ in 1:depth]
    proj1 = PointwiseConv(width, 2 * width)
    proj2 = PointwiseConv(2 * width, cout)
    return FNOSeg(lift, blocks, proj1, proj2)
end

function (m::FNOSeg)(x::AbstractArray{<:Real,4})
    h = m.lift(x)
    for b in m.blocks
        h = b(h)
    end
    h = gelu.(m.proj1(h))
    h = m.proj2(h)
    return sigmoid.(h)
end

end # module
