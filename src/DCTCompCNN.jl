module DCTCompCNN

using FFTW
using Images
using ImageTransformations
using Flux
using Statistics
using Random

export rgb_to_ycbcr, chroma_subsample, block_dct, block_idct,
       quantize_blocks, dequantize_blocks,
       upsample2, downsample2,
       build_dct_input, build_baseline_input,
       build_dct_compcnn, build_baseline_cnn,
       LUMA_QTABLE, CHROMA_QTABLE

# ---------------------------------------------------------------------
# Standard JPEG (Annex K, ITU T.81 / Wallace 1992) quantization tables
# ---------------------------------------------------------------------
const LUMA_QTABLE = Float32[
 16 11 10 16 24  40  51  61
 12 12 14 19 26  58  60  55
 14 13 16 24 40  57  69  56
 14 17 22 29 51  87  80  62
 18 22 37 56 68 109 103  77
 24 35 55 64 81 104 113  92
 49 64 78 87 103 121 120 101
 72 92 95 98 112 100 103  99
]

const CHROMA_QTABLE = Float32[
 17 18 24 47 99 99 99 99
 18 21 26 66 99 99 99 99
 24 26 56 99 99 99 99 99
 47 66 99 99 99 99 99 99
 99 99 99 99 99 99 99 99
 99 99 99 99 99 99 99 99
 99 99 99 99 99 99 99 99
 99 99 99 99 99 99 99 99
]

# ---------------------------------------------------------------------
# Colour space conversion (JPEG uses BT.601 full-range YCbCr)
# ---------------------------------------------------------------------
"""
    rgb_to_ycbcr(img) -> (Y, Cb, Cr)

Convert an RGB image (values in [0,1]) to JPEG-style YCbCr planes,
each returned as a Float32 matrix in the range [0,255] (level-shifted,
i.e. already centred so that 128 maps to 0 for the chroma planes and
DCT can be applied directly, matching how a JPEG encoder feeds 8x8
blocks into the forward DCT).
"""
function rgb_to_ycbcr(img::AbstractMatrix{<:Colorant})
    r = Float32.(red.(img))   .* 255f0
    g = Float32.(green.(img)) .* 255f0
    b = Float32.(blue.(img))  .* 255f0

    Y  =  0.299f0  .* r .+ 0.587f0  .* g .+ 0.114f0  .* b .- 128f0
    Cb = -0.168736f0 .* r .- 0.331264f0 .* g .+ 0.5f0 .* b
    Cr =  0.5f0 .* r .- 0.418688f0 .* g .- 0.081312f0 .* b
    return Y, Cb, Cr
end

"""
    chroma_subsample(C) -> C_sub

4:2:0 chroma subsampling: average non-overlapping 2x2 blocks.
Input dimensions must be even.
"""
function chroma_subsample(C::AbstractMatrix{Float32})
    h, w = size(C)
    @assert iseven(h) && iseven(w) "dimensions must be even for 4:2:0 subsampling"
    out = Array{Float32}(undef, h ÷ 2, w ÷ 2)
    @inbounds for j in 1:(w ÷ 2), i in 1:(h ÷ 2)
        i0, j0 = 2i - 1, 2j - 1
        out[i, j] = 0.25f0 * (C[i0, j0] + C[i0+1, j0] + C[i0, j0+1] + C[i0+1, j0+1])
    end
    return out
end

# ---------------------------------------------------------------------
# Block-wise (8x8) forward / inverse DCT-II, applied "in place" so the
# output has the SAME spatial size as the input (each 8x8 pixel block
# is replaced by its 8x8 DCT-coefficient block) -- this is the
# "DCT coefficient image" representation used for compressed-domain
# CNN input in the paper (partial decoding stops right after the
# (de)quantization stage of Fig. 1, before the IDCT).
# ---------------------------------------------------------------------
function block_dct(plane::AbstractMatrix{Float32})
    h, w = size(plane)
    @assert h % 8 == 0 && w % 8 == 0 "plane dimensions must be multiples of 8"
    out = similar(plane)
    @inbounds for bj in 0:8:(w-8), bi in 0:8:(h-8)
        blk = plane[bi+1:bi+8, bj+1:bj+8]
        out[bi+1:bi+8, bj+1:bj+8] = FFTW.dct(blk)
    end
    return out
end

function block_idct(planeDCT::AbstractMatrix{Float32})
    h, w = size(planeDCT)
    out = similar(planeDCT)
    @inbounds for bj in 0:8:(w-8), bi in 0:8:(h-8)
        blk = planeDCT[bi+1:bi+8, bj+1:bj+8]
        out[bi+1:bi+8, bj+1:bj+8] = FFTW.idct(blk)
    end
    return out
end

function quantize_blocks(planeDCT::AbstractMatrix{Float32}, qtable::AbstractMatrix{Float32})
    h, w = size(planeDCT)
    out = similar(planeDCT)
    @inbounds for bj in 0:8:(w-8), bi in 0:8:(h-8)
        out[bi+1:bi+8, bj+1:bj+8] = round.(planeDCT[bi+1:bi+8, bj+1:bj+8] ./ qtable)
    end
    return out
end

function dequantize_blocks(planeQ::AbstractMatrix{Float32}, qtable::AbstractMatrix{Float32})
    h, w = size(planeQ)
    out = similar(planeQ)
    @inbounds for bj in 0:8:(w-8), bi in 0:8:(h-8)
        out[bi+1:bi+8, bj+1:bj+8] = planeQ[bi+1:bi+8, bj+1:bj+8] .* qtable
    end
    return out
end

# ---------------------------------------------------------------------
# Dimension-matching transformations (Section II of the paper): the
# luma plane is full resolution while chroma planes are half
# resolution after 4:2:0 subsampling, so before the three planes can
# be concatenated into one tensor either the chroma planes must be
# up-sampled by 2, or the luma plane must be down-sampled by 2.
# ---------------------------------------------------------------------
function upsample2(C::AbstractMatrix{Float32})
    h, w = size(C)
    out = Array{Float32}(undef, 2h, 2w)
    @inbounds for j in 1:w, i in 1:h
        v = C[i, j]
        out[2i-1, 2j-1] = v; out[2i, 2j-1] = v
        out[2i-1, 2j]   = v; out[2i, 2j]   = v
    end
    return out
end

function downsample2(Y::AbstractMatrix{Float32})
    h, w = size(Y)
    out = Array{Float32}(undef, h ÷ 2, w ÷ 2)
    @inbounds for j in 1:(w÷2), i in 1:(h÷2)
        i0, j0 = 2i - 1, 2j - 1
        out[i, j] = 0.25f0 * (Y[i0, j0] + Y[i0+1, j0] + Y[i0, j0+1] + Y[i0+1, j0+1])
    end
    return out
end

# ---------------------------------------------------------------------
# Full pipeline: RGB image -> (H,W,3) Float32 tensor ready for the CNN,
# reproducing Fig.2 (quantized) / Fig.3 (un-quantized) of the paper.
# ---------------------------------------------------------------------
"""
    build_dct_input(img; quantized::Bool, sampling::Symbol, target_hw::Int)

`sampling` is `:up` (up-sample chroma to match luma) or `:down`
(down-sample luma to match chroma). `target_hw` is the side length the
CNN expects for its LUMA-resolution grid (e.g. 32 for CIFAR-10-sized
input); the source image is resized so the pipeline lands exactly on
that grid for the chosen sampling mode.
"""
function build_dct_input(img; quantized::Bool, sampling::Symbol, target_hw::Int=32)
    @assert sampling in (:up, :down)
    # Resize so the final concatenated tensor is target_hw x target_hw x 3
    src_hw = sampling === :up ? target_hw : target_hw * 2
    resized = imresize(img, (src_hw, src_hw))

    Y, Cb, Cr = rgb_to_ycbcr(resized)
    Cb = chroma_subsample(Cb)
    Cr = chroma_subsample(Cr)

    Ydct  = block_dct(Y)
    Cbdct = block_dct(Cb)
    Crdct = block_dct(Cr)

    if quantized
        Ydct  = quantize_blocks(Ydct,  LUMA_QTABLE)
        Cbdct = quantize_blocks(Cbdct, CHROMA_QTABLE)
        Crdct = quantize_blocks(Crdct, CHROMA_QTABLE)
    end

    if sampling === :up
        Cbdct = upsample2(Cbdct)
        Crdct = upsample2(Crdct)
    else
        Ydct = downsample2(Ydct)
    end

    @assert size(Ydct) == size(Cbdct) == size(Crdct) == (target_hw, target_hw)

    tensor = Array{Float32}(undef, target_hw, target_hw, 3)
    tensor[:, :, 1] = Ydct
    tensor[:, :, 2] = Cbdct
    tensor[:, :, 3] = Crdct
    return tensor
end

"""
    build_baseline_input(img; target_hw)

Conventional path: full JPEG-equivalent decode (resize + RGB pixels
in [0,1]) as fed to a standard pixel-domain CNN / ResNet-50 style
model.
"""
function build_baseline_input(img; target_hw::Int=32)
    resized = imresize(img, (target_hw, target_hw))
    tensor = Array{Float32}(undef, target_hw, target_hw, 3)
    tensor[:, :, 1] = Float32.(red.(resized))
    tensor[:, :, 2] = Float32.(green.(resized))
    tensor[:, :, 3] = Float32.(blue.(resized))
    return tensor
end

# ---------------------------------------------------------------------
# DCT-CompCNN architecture (Fig. 4 of the paper): 4 conv+maxpool
# stages (3->64->128->256->512 channels, 32->16->8->4->2 spatial),
# flatten (2*2*512 = 2048), then FC 2048->128->256->512->num_classes.
# ---------------------------------------------------------------------
function build_dct_compcnn(; in_hw::Int=32, in_ch::Int=3, num_classes::Int=10)
    @assert in_hw == 32 "reference architecture assumes a 32x32 input grid (as in the paper's CIFAR-10 setup)"
    return Flux.Chain(
        Conv((3, 3), in_ch => 64, pad = SamePad()), BatchNorm(64, relu),
        MaxPool((2, 2)),                                            # 16x16x64
        Conv((3, 3), 64 => 128, pad = SamePad()), BatchNorm(128, relu),
        MaxPool((2, 2)),                                            # 8x8x128
        Conv((3, 3), 128 => 256, pad = SamePad()), BatchNorm(256, relu),
        MaxPool((2, 2)),                                            # 4x4x256
        Conv((3, 3), 256 => 512, pad = SamePad()), BatchNorm(512, relu),
        MaxPool((2, 2)),                                            # 2x2x512
        Flux.flatten,                                               # 2048
        Dense(2048, 128, relu),
        Dense(128, 256, relu),
        Dense(256, 512, relu),
        Dense(512, num_classes),
    )
end

"""
    build_baseline_cnn(; in_hw, in_ch, num_classes)

A conventional pixel-domain CNN of matching depth/capacity, used as
the RGB-pixel-input reference point for the inference-speed
comparison (stands in for "decode JPEG -> RGB -> ResNet-50-style
classifier").
"""
function build_baseline_cnn(; in_hw::Int=32, in_ch::Int=3, num_classes::Int=10)
    return build_dct_compcnn(; in_hw=in_hw, in_ch=in_ch, num_classes=num_classes)
end

end # module
