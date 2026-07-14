include("src/Saliency.jl")
using .Saliency
using Images
using FileIO
using TestImages
using Random

Random.seed!(42)

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)

# ---------------------------------------------------------------------------
# 1) Synthetic fabric-like image with an embedded "defect", mimicking the
#    grid-pattern fabric test images used in the paper (Di, Long & Liang
#    2020, Table 2).
# ---------------------------------------------------------------------------

N = 256
T = 16  # grid period in pixels

fabric = zeros(Float64, N, N)
for y in 1:N, x in 1:N
    online = (mod(x - 1, T) < 2) || (mod(y - 1, T) < 2)
    fabric[y, x] = online ? 0.30 : 0.82
end
fabric .+= 0.03 .* randn(N, N)   # sensor / weave noise
fabric = clamp.(fabric, 0.0, 1.0)

# Embed a localized defect: a patch where the regular weave is disrupted
# (grid lines erased + brightness pulled down), like a snag/hole in fabric.
cx, cy, r = 180, 95, 14
for y in 1:N, x in 1:N
    if (x - cx)^2 + (y - cy)^2 <= r^2
        fabric[y, x] = clamp(0.45 + 0.08 * randn(), 0.0, 1.0)
    end
end

save(joinpath(outdir, "01_fabric_input.png"), Gray.(fabric))

# ---------------------------------------------------------------------------
# 2) Saliency maps: FRFT-domain (paper's alpha=beta=0.9) vs. classical
#    FFT-domain phase-spectrum saliency (alpha=beta=1.0), for comparison.
# ---------------------------------------------------------------------------

sal_frft = frft_saliency(fabric; a=0.9, b=0.9, sigma=6.0)
sal_fft = frft_saliency(fabric; a=1.0, b=1.0, sigma=6.0)

save(joinpath(outdir, "02_saliency_frft_a0.9.png"), Gray.(sal_frft))
save(joinpath(outdir, "03_saliency_fft_a1.0.png"), Gray.(sal_fft))

# Simple detection: threshold the saliency map (mean + k*std, a common rule
# of thumb for phase-spectrum saliency maps) and compare against the true
# defect disk.
function threshold_mask(sal; k=3.0)
    mu, sd = sum(sal) / length(sal), sqrt(sum((sal .- sum(sal) / length(sal)) .^ 2) / length(sal))
    return sal .> (mu + k * sd)
end

gt = falses(N, N)
for y in 1:N, x in 1:N
    gt[y, x] = (x - cx)^2 + (y - cy)^2 <= r^2
end

function iou(mask, gt)
    inter = count(mask .& gt)
    union_ = count(mask .| gt)
    return union_ == 0 ? 0.0 : inter / union_
end

mask_frft = threshold_mask(sal_frft)
mask_fft = threshold_mask(sal_fft)

println("Detected-pixel count  | FRFT(a=0.9): ", count(mask_frft), "   FFT(a=1.0): ", count(mask_fft))
println("IoU vs. ground truth  | FRFT(a=0.9): ", round(iou(mask_frft, gt), digits=3),
        "   FFT(a=1.0): ", round(iou(mask_fft, gt), digits=3))

save(joinpath(outdir, "04_mask_frft.png"), Gray.(Float64.(mask_frft)))
save(joinpath(outdir, "05_mask_fft.png"), Gray.(Float64.(mask_fft)))

# Side-by-side comparison montage: input | ground truth | FRFT saliency | FFT saliency
gt_img = Gray.(Float64.(gt))
montage = hcat(Gray.(fabric), gt_img, Gray.(sal_frft), Gray.(sal_fft))
save(joinpath(outdir, "06_montage_fabric.png"), montage)

# ---------------------------------------------------------------------------
# 3) Generalization check on a natural test image (not fabric-specific).
# ---------------------------------------------------------------------------

natural = Float64.(Gray.(testimage("mandrill")))
sal_nat = frft_saliency(natural; a=0.9, b=0.9, sigma=8.0)
save(joinpath(outdir, "07_natural_input.png"), Gray.(natural))
save(joinpath(outdir, "08_natural_saliency.png"), Gray.(sal_nat))
save(joinpath(outdir, "09_montage_natural.png"), hcat(Gray.(natural), Gray.(sal_nat)))

println("\nAll outputs written to: ", outdir)
