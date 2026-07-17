#!/usr/bin/env julia
# Reproduction / benchmark of DCT-CompCNN (Rajesh, Javed et al., 2019)
# on a single sample image: build the 4 input-representation variants
# (quantized/un-quantized x up-/down-sampling) plus a conventional RGB
# baseline, then measure preprocessing time and CNN forward-pass
# (inference) time for each, mirroring Table I / Table II of the paper.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "DCTCompCNN.jl"))
using .DCTCompCNN
using TestImages
using ImageTransformations
using Flux
using Statistics
using Random
using Printf

Random.seed!(42)

const TARGET_HW = 32   # CIFAR-10-sized CNN input grid, as in the paper's DCT-CompCNN

# ---------------------------------------------------------------------
# timing helper: warm up (JIT-compile), then take the median of N runs
# ---------------------------------------------------------------------
function timeit(f::Function; nruns::Int=30, warmup::Int=5)
    for _ in 1:warmup
        f()
    end
    times = Vector{Float64}(undef, nruns)
    for i in 1:nruns
        t0 = time_ns()
        f()
        times[i] = (time_ns() - t0) / 1e6   # ms
    end
    return minimum(times), median(times)
end

println("="^72)
println("DCT-CompCNN reproduction -- single sample image benchmark")
println("="^72)

img = testimage("mandrill")   # 512x512 RGB standard test image
println("Sample image: mandrill.png, size = ", size(img))

model = build_dct_compcnn(; in_hw=TARGET_HW, num_classes=10)
Flux.testmode!(model)   # inference mode (fixed BatchNorm stats)

baseline_model = build_baseline_cnn(; in_hw=TARGET_HW, num_classes=10)
Flux.testmode!(baseline_model)

variants = [
    ("Quantized   + Downsampling", true,  :down),
    ("Quantized   + Upsampling",   true,  :up),
    ("Unquantized + Downsampling", false, :down),
    ("Unquantized + Upsampling",   false, :up),
]

println()
println("--- Stage 0: DCT/quantization round-trip sanity check ---")
let
    Y, _, _ = rgb_to_ycbcr(imresize(img, (256, 256)))
    Ydct = block_dct(Y)
    Yq   = quantize_blocks(Ydct, LUMA_QTABLE)
    Ydq  = dequantize_blocks(Yq, LUMA_QTABLE)
    Yrec = block_idct(Ydq)
    mse  = mean((Y .- Yrec) .^ 2)
    psnr = 10 * log10(255f0^2 / mse)
    @printf("Luma channel forward-DCT -> quantize -> dequantize -> inverse-DCT\n")
    @printf("round-trip PSNR vs original: %.2f dB (typical JPEG quality ~30-45 dB)\n", psnr)

    Yrec_noq = block_idct(Ydct)
    mse2 = mean((Y .- Yrec_noq) .^ 2)
    @printf("Un-quantized round-trip max abs error: %.2e (should be ~0, confirms DCT/IDCT are exact inverses)\n",
            maximum(abs.(Y .- Yrec_noq)))
end
println("(cost of computing the block-DCT representation from a decoded")
println(" pixel array -- NOT a libjpeg raw-coefficient extraction, see")
println(" note at the bottom. 'Baseline' is a plain resize, i.e. the")
println(" tail end of a conventional decode pipeline.)")
@printf("%-30s | %-16s | %-18s\n", "Variant", "min / median ms", "Output shape")
println("-"^72)

built_inputs = Dict{String,Array{Float32,3}}()
for (name, quantized, sampling) in variants
    mn, med = timeit(() -> build_dct_input(img; quantized=quantized, sampling=sampling, target_hw=TARGET_HW))
    t = build_dct_input(img; quantized=quantized, sampling=sampling, target_hw=TARGET_HW)
    built_inputs[name] = t
    @printf("%-30s | %5.3f / %5.3f | %-18s\n", name, mn, med, string(size(t)))
end

mn_base, med_base = timeit(() -> build_baseline_input(img; target_hw=TARGET_HW))
baseline_input = build_baseline_input(img; target_hw=TARGET_HW)
@printf("%-30s | %5.3f / %5.3f | %-18s\n", "Baseline RGB (resize only)", mn_base, med_base, string(size(baseline_input)))

println()
println("--- Stage 2: CNN forward-pass ('inference') time, batch size 1 ---")
println("(identical architecture on both sides, only input content differs")
println(" -- this isolates whether the DCT-domain representation adds any")
println(" cost on the network side; differences here are measurement noise.)")
@printf("%-30s | %-16s | %-10s\n", "Variant", "min / median ms", "Pred class")
println("-"^72)

for (name, _, _) in variants
    x = reshape(built_inputs[name], TARGET_HW, TARGET_HW, 3, 1)
    mn, med = timeit(() -> model(x); nruns=50, warmup=10)
    pred = Flux.onecold(model(x)) .- 1
    @printf("%-30s | %5.3f / %5.3f | %-10s\n", name, mn, med, string(pred[1]))
end

xb = reshape(baseline_input, TARGET_HW, TARGET_HW, 3, 1)
mn_b, med_b = timeit(() -> baseline_model(xb); nruns=50, warmup=10)
pred_b = Flux.onecold(baseline_model(xb)) .- 1
@printf("%-30s | %5.3f / %5.3f | %-10s\n", "Baseline RGB", mn_b, med_b, string(pred_b[1]))

println()
println("--- Stage 3: end-to-end (tensor build + inference) time ---")
@printf("%-30s | %-16s\n", "Variant", "min / median ms")
println("-"^72)
for (name, quantized, sampling) in variants
    f = () -> begin
        t = build_dct_input(img; quantized=quantized, sampling=sampling, target_hw=TARGET_HW)
        model(reshape(t, TARGET_HW, TARGET_HW, 3, 1))
    end
    mn, med = timeit(f; nruns=20, warmup=5)
    @printf("%-30s | %5.3f / %5.3f\n", name, mn, med)
end
fb = () -> begin
    t = build_baseline_input(img; target_hw=TARGET_HW)
    baseline_model(reshape(t, TARGET_HW, TARGET_HW, 3, 1))
end
mn_bt, med_bt = timeit(fb; nruns=20, warmup=5)
@printf("%-30s | %5.3f / %5.3f\n", "Baseline RGB (resize + infer)", mn_bt, med_bt)

println()
println("="^72)
println("NOTES")
println("-"^72)
println("1) Accuracy: the paper's 81-86% figures come from training on the")
println("   FULL Dogs-vs-Cats / CIFAR-10 datasets for 45-100 epochs. A")
println("   single sample image has no label to train against, so this run")
println("   uses randomly-initialised weights and reports a (meaningless)")
println("   predicted class purely to prove the forward pass executes.")
println("   Classification accuracy was not and cannot be measured here.")
println()
println("2) Speed: Stage 1 measures computing the block-DCT/quantization")
println("   transform from an already-decoded pixel array (FFTW dct per")
println("   8x8 block), not a true libjpeg partial-decode of a compressed")
println("   bitstream (jpeg_read_coefficients), so it does NOT reproduce")
println("   the paper's IDCT-skipping speed advantage -- it only shows the")
println("   relative cost of the up-/down-sampling and (de)quantization")
println("   choices. Stage 2 (identical CNN, different input content) is")
println("   the fair apples-to-apples inference-speed comparison.")
println("="^72)
