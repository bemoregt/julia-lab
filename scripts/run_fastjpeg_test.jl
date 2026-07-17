#!/usr/bin/env julia
# Reproduction / benchmark of "Faster Neural Networks Straight from JPEG"
# (Gueguen et al., NeurIPS 2018) on a sample image: build a standard
# ResNet-50 (RGB, 224x224x3) and the "UpSampling" DCT-input variant
# (stem + conv2_x replaced by decimating block-DCT features feeding
# directly into conv3_x), then compare inference throughput (images/sec),
# mirroring the FPS column of the paper's Table 1.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "FastJPEGResNet.jl"))
using .FastJPEGResNet
using TestImages
using ImageTransformations
using Images
using Flux
using Statistics
using Printf
using Random

const HAS_METAL = try
    @eval using Metal
    Metal.functional()
catch e
    @warn "Metal.jl not available/functional, falling back to CPU only" exception=e
    false
end

Random.seed!(42)

# `sync`: pass Metal.synchronize for GPU timing (Metal dispatches
# asynchronously, so wall-clock time around the call alone would only
# measure kernel-launch/queueing overhead, not actual compute time).
function timeit(f::Function; nruns::Int, warmup::Int, sync::Function=() -> nothing)
    for _ in 1:warmup
        f(); sync()
    end
    times = Vector{Float64}(undef, nruns)
    for i in 1:nruns
        t0 = time_ns()
        f()
        sync()
        times[i] = (time_ns() - t0) / 1e6   # ms
    end
    return minimum(times), median(times)
end

println("="^72)
println("Faster-Neural-Networks-Straight-from-JPEG reproduction (Julia/Flux)")
println("="^72)

# ---------------------------------------------------------------------
# Sample image -> batch
# ---------------------------------------------------------------------
img = testimage("mandrill")
img224 = imresize(img, (224, 224))
rgb01 = Array{Float32}(undef, 224, 224, 3)
rgb01[:, :, 1] = Float32.(red.(img224))
rgb01[:, :, 2] = Float32.(green.(img224))
rgb01[:, :, 3] = Float32.(blue.(img224))

BATCH = 4
println("Sample image: mandrill.png resized to 224x224, batch size = $BATCH (CPU)")

rgb_batch = Array{Float32}(undef, 224, 224, 3, BATCH)
for n in 1:BATCH
    rgb_batch[:, :, :, n] = rgb01
end

println()
println("--- Building DCT input tensor ---")
dct_single = FastJPEGResNet.build_dct_tensor(rgb01)   # warmup / JIT compile
println("D_Y/D_Cb/D_Cr -> concat shape: ", size(dct_single), " (expect (28,28,192))")
let
    mn, med = timeit(() -> FastJPEGResNet.build_dct_tensor(rgb01); nruns=30, warmup=5)
    @printf("Single-image DCT-tensor build time: min %.4f ms / median %.4f ms\n", mn, med)
end

dct_batch = Array{Float32}(undef, 28, 28, 192, BATCH)
for n in 1:BATCH
    dct_batch[:, :, :, n] = dct_single
end

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
println()
println("--- Building models ---")
model_rgb = build_resnet50_rgb(; num_classes=1000)
model_dct = build_resnet50_dct_upsampling(; num_classes=1000)
Flux.testmode!(model_rgb)
Flux.testmode!(model_dct)

nparams(m) = sum(length, Flux.trainables(m))
println("ResNet-50 RGB baseline params:        ", nparams(model_rgb))
println("ResNet-50 DCT-UpSampling variant params: ", nparams(model_dct))

println()
println("--- Sanity check: forward pass output shape (CPU) ---")
out_rgb = model_rgb(rgb_batch)
out_dct = model_dct(dct_batch)
println("RGB branch output: ", size(out_rgb), "   DCT branch output: ", size(out_dct))

# ---------------------------------------------------------------------
# Benchmark helper: run both models on a given device (CPU array /
# Metal MtlArray), return timing results.
# ---------------------------------------------------------------------
function bench_device(name::String, to_dev::Function, sync::Function; batch::Int, nruns::Int, warmup::Int)
    mrgb = to_dev(model_rgb)
    mdct = to_dev(model_dct)
    xrgb = to_dev(rgb01_tensor_batch(batch))
    xdct = to_dev(repeat_dct_batch(batch))

    mn_rgb, med_rgb = timeit(() -> mrgb(xrgb); nruns=nruns, warmup=warmup, sync=sync)
    mn_dct, med_dct = timeit(() -> mdct(xdct); nruns=nruns, warmup=warmup, sync=sync)

    return (; name, batch, mn_rgb, med_rgb, mn_dct, med_dct,
              fps_rgb = batch / (med_rgb/1000), fps_dct = batch / (med_dct/1000))
end

rgb01_tensor_batch(b) = (a = Array{Float32}(undef, 224,224,3,b); for n in 1:b; a[:,:,:,n]=rgb01; end; a)
repeat_dct_batch(b) = (a = Array{Float32}(undef, 28,28,192,b); for n in 1:b; a[:,:,:,n]=dct_single; end; a)

results = Any[]

println()
println("--- Inference speed: CPU, batch=$BATCH, warmup=5, nruns=20 ---")
r_cpu = bench_device("CPU", identity, () -> nothing; batch=BATCH, nruns=20, warmup=5)
push!(results, r_cpu)
@printf("%-30s | %10s | %10s | %10s\n", "Model", "min ms", "median ms", "FPS (med)")
println("-"^72)
@printf("%-30s | %10.2f | %10.2f | %10.2f\n", "ResNet-50 RGB (full stem)", r_cpu.mn_rgb, r_cpu.med_rgb, r_cpu.fps_rgb)
@printf("%-30s | %10.2f | %10.2f | %10.2f\n", "ResNet-50 DCT-UpSampling", r_cpu.mn_dct, r_cpu.med_dct, r_cpu.fps_dct)

r_mps = nothing
if HAS_METAL
    println()
    println("--- Inference speed: MPS/Metal GPU (Apple M1), batch=$BATCH, warmup=5, nruns=20 ---")
    r_mps = bench_device("MPS", Flux.gpu, Metal.synchronize; batch=BATCH, nruns=20, warmup=5)
    push!(results, r_mps)
    @printf("%-30s | %10s | %10s | %10s\n", "Model", "min ms", "median ms", "FPS (med)")
    println("-"^72)
    @printf("%-30s | %10.2f | %10.2f | %10.2f\n", "ResNet-50 RGB (full stem)", r_mps.mn_rgb, r_mps.med_rgb, r_mps.fps_rgb)
    @printf("%-30s | %10.2f | %10.2f | %10.2f\n", "ResNet-50 DCT-UpSampling", r_mps.mn_dct, r_mps.med_dct, r_mps.fps_dct)

    println()
    println("--- Inference speed: MPS/Metal GPU, larger batch=32 (throughput scaling) ---")
    r_mps32 = bench_device("MPS batch32", Flux.gpu, Metal.synchronize; batch=32, nruns=15, warmup=5)
    @printf("%-30s | %10.2f | %10.2f | %10.2f\n", "ResNet-50 RGB (full stem)", r_mps32.mn_rgb, r_mps32.med_rgb, r_mps32.fps_rgb)
    @printf("%-30s | %10.2f | %10.2f | %10.2f\n", "ResNet-50 DCT-UpSampling", r_mps32.mn_dct, r_mps32.med_dct, r_mps32.fps_dct)
else
    println()
    println("Metal.jl not functional on this machine -- MPS benchmark skipped.")
end

println()
println("="^72)
println("SUMMARY")
println("-"^72)
@printf("%-14s | %-26s | %10s | %10s\n", "Device", "Model", "med ms", "FPS")
println("-"^72)
for r in results
    @printf("%-14s | %-26s | %10.2f | %10.2f\n", r.name, "ResNet-50 RGB", r.med_rgb, r.fps_rgb)
    @printf("%-14s | %-26s | %10.2f | %10.2f\n", r.name, "ResNet-50 DCT-UpSampling", r.med_dct, r.fps_dct)
end
println()
if r_mps !== nothing
    @printf("DCT-vs-RGB speedup   -- CPU: %.2fx   MPS: %.2fx\n",
            r_cpu.med_rgb / r_cpu.med_dct, r_mps.med_rgb / r_mps.med_dct)
    @printf("MPS-vs-CPU speedup   -- RGB model: %.2fx   DCT model: %.2fx\n",
            r_cpu.med_rgb / r_mps.med_rgb, r_cpu.med_dct / r_mps.med_dct)
end

println()
println("="^72)
println("NOTES")
println("-"^72)
println("1) Paper reference numbers (Table 1, NVIDIA Pascal GPU, batch 1024,")
println("   averaged over 128 GPUs): ResNet-50 RGB = 208 FPS; DCT UpSampling")
println("   = 396 FPS (1.90x); Late-Concat-RFA-Thinner = 369 FPS (1.77x, the")
println("   architecture the paper highlights as 'same accuracy, 1.77x').")
println("   This script implements the simpler 'UpSampling' variant.")
println()
println("2) MPS here is Apple Silicon's unified-memory Metal GPU via Metal.jl,")
println("   NOT an NVIDIA Pascal GPU with cuDNN, and no libjpeg raw-coefficient")
println("   decode is used -- absolute FPS is NOT comparable to the paper's")
println("   numbers. What IS comparable is the qualitative mechanism: the DCT")
println("   branch skips the stem (7x7/2 conv + maxpool) and conv2_x, which")
println("   run at 112x112 and 56x56 resolution in the RGB path, vs the DCT")
println("   branch starting directly at 28x28.")
println()
println("3) No accuracy/training comparison: both models have random")
println("   (untrained) weights -- there is no ImageNet-trained checkpoint")
println("   or labelled sample to evaluate top-1/top-5 error against.")
println("="^72)
