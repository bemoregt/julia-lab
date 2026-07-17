#!/usr/bin/env julia
# Training-speed benchmark for the ResNet-50 RGB baseline vs the DCT
# "UpSampling" variant (Gueguen et al., NeurIPS 2018), on CPU and
# MPS/Metal (Apple Silicon GPU).
#
# NOTE: the paper itself only publishes an INFERENCE speed table
# (Table 1). Training is only described narratively ("each experiment
# trains for 90 epochs, corresponding to 2-3 hours" on 128 Pascal
# GPUs) -- no RGB-vs-DCT training-speed ratio is reported. So there is
# no paper number to reproduce here; this script measures our own
# full training-step time (forward + backward + optimizer update) as
# an independent data point, using a synthetic classification loss
# (random one-hot labels) since no labelled dataset is used.

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
println("Training-speed benchmark: ResNet-50 RGB vs DCT-UpSampling (Julia/Flux)")
println("="^72)

# ---------------------------------------------------------------------
# Sample image -> batch tensors (RGB pixel + DCT-coefficient forms)
# ---------------------------------------------------------------------
img = testimage("mandrill")
img224 = imresize(img, (224, 224))
rgb01 = Array{Float32}(undef, 224, 224, 3)
rgb01[:, :, 1] = Float32.(red.(img224))
rgb01[:, :, 2] = Float32.(green.(img224))
rgb01[:, :, 3] = Float32.(blue.(img224))
dct_single = FastJPEGResNet.build_dct_tensor(rgb01)

rgb_batch_fn(b) = (a = Array{Float32}(undef, 224,224,3,b); for n in 1:b; a[:,:,:,n] = rgb01; end; a)
dct_batch_fn(b) = (a = Array{Float32}(undef, 28,28,192,b); for n in 1:b; a[:,:,:,n] = dct_single; end; a)

const NCLASSES = 1000
labels_fn(b) = Flux.onehotbatch(rand(1:NCLASSES, b), 1:NCLASSES)

loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

# One full training step: forward + backward (gradient) + optimizer update
function make_train_step(model, opt_state, x, y)
    return function ()
        gs = Flux.gradient(m -> loss(m, x, y), model)
        Flux.update!(opt_state, model, gs[1])
        return nothing
    end
end

function bench_train(name::String, to_dev::Function, sync::Function; batch::Int, nruns::Int, warmup::Int)
    m_rgb = build_resnet50_rgb(; num_classes=NCLASSES); Flux.trainmode!(m_rgb); m_rgb = to_dev(m_rgb)
    m_dct = build_resnet50_dct_upsampling(; num_classes=NCLASSES); Flux.trainmode!(m_dct); m_dct = to_dev(m_dct)

    x_rgb, y_rgb = to_dev(rgb_batch_fn(batch)), to_dev(labels_fn(batch))
    x_dct, y_dct = to_dev(dct_batch_fn(batch)), to_dev(labels_fn(batch))

    opt_rgb = Flux.setup(Flux.Adam(1f-3), m_rgb)
    opt_dct = Flux.setup(Flux.Adam(1f-3), m_dct)

    step_rgb = make_train_step(m_rgb, opt_rgb, x_rgb, y_rgb)
    step_dct = make_train_step(m_dct, opt_dct, x_dct, y_dct)

    mn_rgb, med_rgb = timeit(step_rgb; nruns=nruns, warmup=warmup, sync=sync)
    mn_dct, med_dct = timeit(step_dct; nruns=nruns, warmup=warmup, sync=sync)

    return (; name, batch, mn_rgb, med_rgb, mn_dct, med_dct,
              ips_rgb = batch / (med_rgb/1000), ips_dct = batch / (med_dct/1000))
end

BATCH = 4
results = Any[]

println()
println("--- Training-step speed: CPU, batch=$BATCH, warmup=3, nruns=10 ---")
println("(forward + backward + Adam update, synthetic random labels)")
r_cpu = bench_train("CPU", identity, () -> nothing; batch=BATCH, nruns=10, warmup=3)
push!(results, r_cpu)
@printf("%-30s | %10s | %10s | %14s\n", "Model", "min ms", "median ms", "images/sec")
println("-"^72)
@printf("%-30s | %10.2f | %10.2f | %14.2f\n", "ResNet-50 RGB (full stem)", r_cpu.mn_rgb, r_cpu.med_rgb, r_cpu.ips_rgb)
@printf("%-30s | %10.2f | %10.2f | %14.2f\n", "ResNet-50 DCT-UpSampling", r_cpu.mn_dct, r_cpu.med_dct, r_cpu.ips_dct)

r_mps = nothing
if HAS_METAL
    println()
    println("--- Training-step speed: MPS/Metal GPU (Apple M1), batch=$BATCH, warmup=3, nruns=10 ---")
    r_mps = bench_train("MPS", Flux.gpu, Metal.synchronize; batch=BATCH, nruns=10, warmup=3)
    push!(results, r_mps)
    @printf("%-30s | %10s | %10s | %14s\n", "Model", "min ms", "median ms", "images/sec")
    println("-"^72)
    @printf("%-30s | %10.2f | %10.2f | %14.2f\n", "ResNet-50 RGB (full stem)", r_mps.mn_rgb, r_mps.med_rgb, r_mps.ips_rgb)
    @printf("%-30s | %10.2f | %10.2f | %14.2f\n", "ResNet-50 DCT-UpSampling", r_mps.mn_dct, r_mps.med_dct, r_mps.ips_dct)
else
    println("\nMetal.jl not functional on this machine -- MPS training benchmark skipped.")
end

println()
println("="^72)
println("SUMMARY")
println("-"^72)
@printf("%-14s | %-26s | %10s | %14s\n", "Device", "Model", "med ms/step", "images/sec")
println("-"^72)
for r in results
    @printf("%-14s | %-26s | %10.2f | %14.2f\n", r.name, "ResNet-50 RGB", r.med_rgb, r.ips_rgb)
    @printf("%-14s | %-26s | %10.2f | %14.2f\n", r.name, "ResNet-50 DCT-UpSampling", r.med_dct, r.ips_dct)
end
println()
@printf("DCT-vs-RGB training-step speedup -- CPU: %.2fx", r_cpu.med_rgb / r_cpu.med_dct)
if r_mps !== nothing
    @printf("   MPS: %.2fx\n", r_mps.med_rgb / r_mps.med_dct)
    @printf("MPS-vs-CPU training speedup -- RGB model: %.2fx   DCT model: %.2fx\n",
            r_cpu.med_rgb / r_mps.med_rgb, r_cpu.med_dct / r_mps.med_dct)
else
    println()
end

println()
println("="^72)
println("NOTES")
println("-"^72)
println("1) The paper reports NO train-time RGB-vs-DCT speed ratio -- Table 1")
println("   is inference-only. It only states each 90-epoch ImageNet run took")
println("   2-3 hours on 128 Pascal GPUs via Horovod, without a breakdown per")
println("   architecture. So there is no published number to compare against")
println("   here; these results are an independent measurement only.")
println()
println("2) The paper also notes a real dataloader-level caveat that applies")
println("   more to training than inference: JPEG-domain training normally")
println("   can't use spatial data augmentation (random crop/flip) directly")
println("   on DCT coefficients, so in practice images are decoded, augmented,")
println("   and RE-ENCODED to DCT every step -- an extra cost NOT modelled in")
println("   this benchmark (we feed a fixed pre-built DCT tensor repeatedly).")
println("   So this training-step number is optimistic for the DCT path")
println("   relative to a real training pipeline with augmentation.")
println()
println("3) Same architecture/hardware caveats as the inference benchmark:")
println("   CPU/MPS on one Apple Silicon machine, random weights, no real")
println("   labels -- only step-time (images/sec), not convergence, accuracy,")
println("   or wall-clock-to-90-epochs, is measured.")
println("="^72)
