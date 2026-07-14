include("src/FNO.jl")
include("src/SyntheticData.jl")
using .FNOModel
using .SyntheticData
using Flux
using Random
using Statistics
using BSON: @save

# ---------------------------------------------------------------------------
# Config (mirrors the paper's protocol: train at a single low resolution,
# 500 training images).
# ---------------------------------------------------------------------------
const TRAIN_RES = 64
const N_TRAIN = 500
const N_VAL = 40
const BATCH = 10
const EPOCHS = parse(Int, get(ENV, "FNO_EPOCHS", "40"))
const SEED = 42

rng = MersenneTwister(SEED)

println("Generating $N_TRAIN training shapes + $N_VAL validation shapes at $(TRAIN_RES)x$(TRAIN_RES) ...")
train_shapes = [sample_shape(rng) for _ in 1:N_TRAIN]
val_shapes = [sample_shape(rng) for _ in 1:N_VAL]

function make_batch(shapes, res; rng=Random.default_rng())
    n = length(shapes)
    X = Array{Float64}(undef, 1, res, res, n)
    Y = Array{Float64}(undef, 1, res, res, n)
    for (i, s) in enumerate(shapes)
        img, mask = rasterize(s, res; noise=0.03, rng=rng)
        X[1, :, :, i] = img
        Y[1, :, :, i] = mask
    end
    return X, Y
end

Xval, Yval = make_batch(val_shapes, TRAIN_RES; rng=rng)

# ---------------------------------------------------------------------------
# Model + loss (BCE + soft Dice, standard for segmentation with class
# imbalance between small foreground blob and background).
# ---------------------------------------------------------------------------
model = FNOSeg(; cin=1, width=20, modes1=12, modes2=12, depth=4, cout=1)
opt_state = Flux.setup(Flux.Adam(1e-3), model)

function dice_loss(pred, target; eps=1.0)
    inter = sum(pred .* target)
    return 1 - (2 * inter + eps) / (sum(pred) + sum(target) + eps)
end

function loss_fn(m, x, y)
    pred = m(x)
    bce = Flux.Losses.binarycrossentropy(pred, y)
    return bce + dice_loss(pred, y)
end

function dice_score(pred, target; eps=1.0)
    predb = pred .> 0.5
    inter = sum(predb .& (target .> 0.5))
    return (2 * inter + eps) / (sum(predb) + sum(target .> 0.5) + eps)
end

println("Training FNOSeg for $EPOCHS epochs (batch=$BATCH, train_res=$TRAIN_RES) ...")
t0 = time()
for epoch in 1:EPOCHS
    perm = randperm(rng, N_TRAIN)
    epoch_loss = 0.0
    nbatches = 0
    for start in 1:BATCH:N_TRAIN
        idx = perm[start:min(start + BATCH - 1, N_TRAIN)]
        X, Y = make_batch(train_shapes[idx], TRAIN_RES; rng=rng)
        l, grads = Flux.withgradient(m -> loss_fn(m, X, Y), model)
        Flux.update!(opt_state, model, grads[1])
        epoch_loss += l
        nbatches += 1
    end
    if epoch % 5 == 0 || epoch == 1 || epoch == EPOCHS
        predval = model(Xval)
        vloss = loss_fn(model, Xval, Yval)
        vdice = dice_score(predval, Yval)
        elapsed = round(time() - t0, digits=1)
        println("epoch $epoch/$EPOCHS  train_loss=$(round(epoch_loss/nbatches,digits=4))  " *
                "val_loss=$(round(vloss,digits=4))  val_dice=$(round(vdice,digits=4))  " *
                "elapsed=$(elapsed)s")
    end
end

mkpath("checkpoints")
model_cpu = model
@save "checkpoints/fno_seg.bson" model_cpu
println("Saved trained model to checkpoints/fno_seg.bson")
