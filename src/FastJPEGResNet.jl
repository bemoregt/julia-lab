module FastJPEGResNet
# Reproduction of the CNN designs from:
#   Gueguen, Sergeev, Kadlec, Liu, Yosinski,
#   "Faster Neural Networks Straight from JPEG", NeurIPS 2018.
#
# We implement:
#   - a standard ResNet-50 (RGB input, 224x224x3) as the baseline
#   - the "UpSampling" DCT-input variant: decimating 8x8 block DCT on
#     Y/Cb/Cr (each 8x8 pixel block -> 64 "channels" at 1/8 spatial
#     resolution, exactly as described in Sec 2.2 of the paper --
#     "The DCT can be thought of as convolution with filter size 8x8,
#     stride of 8x8, one input channel, 64 output channels"), chroma
#     upsampled x2 to match luma, concatenated, then fed directly into
#     conv3_x of ResNet-50 (skipping the stem and conv2_x) with the
#     first conv3_x block's stride reduced from 2 to 1, per Sec 3
#     ("UpSampling ... go through a batch normalization layer before
#     going into ResNet ConvBlock 3 (CB3) but with reduced stride 1,
#     then standard CB4 and CB5").
#
# This is a from-scratch Flux re-implementation (the paper's own code
# uses Keras/TensorFlow + a modified libjpeg); it is meant to test
# whether the same *architectural* idea (skip the early high-resolution
# stem by consuming block-DCT coefficients directly) yields a
# comparable relative inference speedup, not to bit-reproduce their
# ImageNet accuracy numbers (no training data/labels are used here).

using Flux
using FFTW
using Statistics

export dct_feature_map, build_resnet50_rgb, build_resnet50_dct_upsampling,
       ycbcr_planes, chroma_subsample2, upsample2x

# ---------------------------------------------------------------------
# Colour space + chroma subsampling (same conventions as DCTCompCNN.jl)
# ---------------------------------------------------------------------
function ycbcr_planes(rgb::AbstractArray{Float32,3})  # (H,W,3) in [0,1]
    r = @view(rgb[:, :, 1]) .* 255f0
    g = @view(rgb[:, :, 2]) .* 255f0
    b = @view(rgb[:, :, 3]) .* 255f0
    Y  =  0.299f0    .* r .+ 0.587f0    .* g .+ 0.114f0    .* b .- 128f0
    Cb = -0.168736f0 .* r .- 0.331264f0 .* g .+ 0.5f0      .* b
    Cr =  0.5f0      .* r .- 0.418688f0 .* g .- 0.081312f0 .* b
    return Y, Cb, Cr
end

function chroma_subsample2(C::AbstractMatrix{Float32})
    h, w = size(C)
    out = Array{Float32}(undef, h ÷ 2, w ÷ 2)
    @inbounds for j in 1:(w ÷ 2), i in 1:(h ÷ 2)
        i0, j0 = 2i - 1, 2j - 1
        out[i, j] = 0.25f0 * (C[i0,j0] + C[i0+1,j0] + C[i0,j0+1] + C[i0+1,j0+1])
    end
    return out
end

# ---------------------------------------------------------------------
# Decimating 8x8 block DCT: (H,W) -> (H/8, W/8, 64).
# Equivalent to depthwise convolution, filter 8x8, stride 8, 64 output
# channels, with the (non-learned, orthonormal) DCT-II basis as
# filters -- see Sec 2.2 / Fig. 2a of the paper.
# ---------------------------------------------------------------------
const DCT8 = let
    D = Array{Float32}(undef, 8, 8)
    for k in 0:7, n in 0:7
        ck = k == 0 ? sqrt(1f0/8f0) : sqrt(2f0/8f0)
        D[k+1, n+1] = ck * cos(Float32(pi) * (2n + 1) * k / 16f0)
    end
    D
end

function dct_feature_map(plane::AbstractMatrix{Float32})
    h, w = size(plane)
    @assert h % 8 == 0 && w % 8 == 0
    gh, gw = h ÷ 8, w ÷ 8
    out = Array{Float32}(undef, gh, gw, 64)
    @inbounds for bj in 1:gw, bi in 1:gh
        blk = plane[(bi-1)*8+1:bi*8, (bj-1)*8+1:bj*8]
        coeffs = DCT8 * blk * DCT8'          # 8x8 2D DCT-II
        out[bi, bj, :] = reshape(coeffs, 64)
    end
    return out
end

function upsample2x(A::AbstractArray{Float32,3})   # (H,W,C) -> (2H,2W,C)
    h, w, c = size(A)
    out = Array{Float32}(undef, 2h, 2w, c)
    @inbounds for k in 1:c, j in 1:w, i in 1:h
        v = A[i, j, k]
        out[2i-1,2j-1,k]=v; out[2i,2j-1,k]=v; out[2i-1,2j,k]=v; out[2i,2j,k]=v
    end
    return out
end

"""
    build_dct_tensor(rgb01_224) -> Array Float32 (28,28,192)

Full pipeline for one 224x224x3 RGB image (values in [0,1]):
YCbCr -> 4:2:0 chroma subsample -> decimating block DCT on each plane
-> upsample Cb/Cr DCT maps x2 -> concat channel-wise. Matches Fig.1a
dimensions: D_Y is 28x28x64, D_Cb/D_Cr are 14x14x64.
"""
function build_dct_tensor(rgb01_224::AbstractArray{Float32,3})
    @assert size(rgb01_224) == (224, 224, 3)
    Y, Cb, Cr = ycbcr_planes(rgb01_224)
    Cb, Cr = chroma_subsample2(Cb), chroma_subsample2(Cr)
    DY  = dct_feature_map(Y)     # 28x28x64
    DCb = dct_feature_map(Cb)    # 14x14x64
    DCr = dct_feature_map(Cr)    # 14x14x64
    DCb = upsample2x(DCb)        # 28x28x64
    DCr = upsample2x(DCr)        # 28x28x64
    return cat(DY, DCb, DCr; dims=3)   # 28x28x192
end

# ---------------------------------------------------------------------
# Standard ResNet-50 bottleneck block (post-activation, v1 style)
# ---------------------------------------------------------------------
function bottleneck(in_ch::Int, mid_ch::Int, out_ch::Int; stride::Int=1)
    main = Chain(
        Conv((1,1), in_ch => mid_ch; stride=1, pad=0, bias=false), BatchNorm(mid_ch, relu),
        Conv((3,3), mid_ch => mid_ch; stride=stride, pad=1, bias=false), BatchNorm(mid_ch, relu),
        Conv((1,1), mid_ch => out_ch; stride=1, pad=0, bias=false), BatchNorm(out_ch),
    )
    shortcut = (stride != 1 || in_ch != out_ch) ?
        Chain(Conv((1,1), in_ch => out_ch; stride=stride, pad=0, bias=false), BatchNorm(out_ch)) :
        identity
    return Chain(Parallel(+, main, shortcut), x -> relu.(x))
end

function res_stage(in_ch::Int, mid_ch::Int, out_ch::Int, nblocks::Int; first_stride::Int=2)
    layers = Any[bottleneck(in_ch, mid_ch, out_ch; stride=first_stride)]
    for _ in 2:nblocks
        push!(layers, bottleneck(out_ch, mid_ch, out_ch; stride=1))
    end
    return Chain(layers...)
end

# ResNet-50 stage config: (blocks, mid_ch, out_ch)
const STAGE_CFG = [(3, 64, 256), (4, 128, 512), (6, 256, 1024), (3, 512, 2048)]

"""
    build_resnet50_rgb(; num_classes=1000)

Full ResNet-50: 7x7/2 stem + maxpool + conv2_x..conv5_x + global avg
pool + FC. Input (224,224,3,N).
"""
function build_resnet50_rgb(; num_classes::Int=1000)
    stem = Chain(
        Conv((7,7), 3 => 64; stride=2, pad=3, bias=false), BatchNorm(64, relu),
        MaxPool((3,3); stride=2, pad=1),
    )
    conv2 = res_stage(64, 64, 256, STAGE_CFG[1][1]; first_stride=1)
    conv3 = res_stage(256, 128, 512, STAGE_CFG[2][1]; first_stride=2)
    conv4 = res_stage(512, 256, 1024, STAGE_CFG[3][1]; first_stride=2)
    conv5 = res_stage(1024, 512, 2048, STAGE_CFG[4][1]; first_stride=2)
    return Chain(stem, conv2, conv3, conv4, conv5,
                 AdaptiveMeanPool((1,1)), Flux.flatten, Dense(2048, num_classes))
end

"""
    build_resnet50_dct_upsampling(; num_classes=1000)

The "UpSampling" DCT variant (Sec 3): stem + conv2_x are REMOVED.
Input is the (28,28,192,N) DCT tensor built by `build_dct_tensor`.
A BatchNorm "entry" layer normalises it, then it feeds conv3_x (first
block stride reduced 2->1, since input is already at conv3_x's usual
OUTPUT resolution), followed by standard conv4_x, conv5_x.
"""
function build_resnet50_dct_upsampling(; num_classes::Int=1000)
    entry = BatchNorm(192)
    conv3 = res_stage(192, 128, 512, STAGE_CFG[2][1]; first_stride=1)   # reduced-stride CB3
    conv4 = res_stage(512, 256, 1024, STAGE_CFG[3][1]; first_stride=2)
    conv5 = res_stage(1024, 512, 2048, STAGE_CFG[4][1]; first_stride=2)
    return Chain(entry, conv3, conv4, conv5,
                 AdaptiveMeanPool((1,1)), Flux.flatten, Dense(2048, num_classes))
end

end # module
