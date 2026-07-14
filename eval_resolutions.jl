include("src/FNO.jl")
include("src/SyntheticData.jl")
using .FNOModel
using .SyntheticData
using Flux
using Random
using Images
using FileIO
using BSON: @load

@load "checkpoints/fno_seg.bson" model_cpu
model = model_cpu

const TRAIN_RES = 64
const TEST_RESOLUTIONS = (32, 64, 96, 128, 200, 256)
const N_TEST = 30
const SEED = 999   # different seed from training -> unseen shapes

rng = MersenneTwister(SEED)
test_shapes = [sample_shape(rng) for _ in 1:N_TEST]

function dice_score(pred, target; eps=1.0)
    predb = pred .> 0.5
    inter = sum(predb .& (target .> 0.5))
    return (2 * inter + eps) / (sum(predb) + sum(target .> 0.5) + eps)
end

mkpath("output")

println("Model trained ONLY at $(TRAIN_RES)x$(TRAIN_RES). Evaluating (no retraining) at:")
println(rpad("resolution", 12), rpad("mean Dice", 12), "std Dice")

results = Dict{Int,Float64}()
for res in TEST_RESOLUTIONS
    dices = Float64[]
    for s in test_shapes
        img, mask = rasterize(s, res; noise=0.03, rng=rng)
        x = reshape(img, 1, res, res, 1)
        y = reshape(mask, 1, res, res, 1)
        pred = model(x)
        push!(dices, dice_score(pred, y))
    end
    mu = sum(dices) / length(dices)
    sd = sqrt(sum((dices .- mu) .^ 2) / length(dices))
    results[res] = mu
    println(rpad(string(res, "x", res), 12), rpad(string(round(mu, digits=4)), 12), round(sd, digits=4))
end

# ---------------------------------------------------------------------------
# Visualization: one held-out shape rendered + segmented at every test
# resolution, arranged as (image | ground truth | prediction) rows.
# ---------------------------------------------------------------------------
vis_shape = test_shapes[1]
for res in TEST_RESOLUTIONS
    img, mask = rasterize(vis_shape, res; noise=0.03, rng=MersenneTwister(123))
    x = reshape(img, 1, res, res, 1)
    pred = model(x)[1, :, :, 1]
    row = hcat(Gray.(img), Gray.(mask), Gray.(clamp.(pred, 0, 1)))
    save("output/eval_res$(res)_img_gt_pred.png", row)
end

println("\nSaved per-resolution (image | ground truth | prediction) panels to output/eval_res*_img_gt_pred.png")
println("\nSummary — trained once at $(TRAIN_RES)x$(TRAIN_RES), evaluated with NO retraining:")
for res in TEST_RESOLUTIONS
    println("  $(res)x$(res): Dice = $(round(results[res], digits=4))")
end
