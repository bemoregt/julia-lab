using Images
using Random
using Statistics: mean
using Zygote

include(joinpath(@__DIR__, "jpeg_domain_resnet.jl"))

# =============================================================================
# Hymenoptera(개미/꿀벌) 이진 분류: 작은 잔차 CNN을 "공간 영역"에서 표준
# 역전파(Zygote 자동미분)로 학습한 뒤, jpeg_domain_resnet.jl에서 이미 검증한
# JPEG 변환 영역 연산(jpeg_domain_conv, dct_domain_batchnorm, asm_relu_block,
# global_avg_from_dc)으로 그대로 "모델 변환"(논문 Section 4.6)하여 같은
# 가중치로 압축 영역에서 분류 추론을 수행하고, 두 도메인의 정확도를 비교한다.
#
# 아키텍처 (ResNet 기본 블록 1개):
#   Conv(3x3,1->C) -> BN -> ReLU -> Conv(3x3,C->C) -> BN -> (+skip) -> ReLU
#   -> GlobalAveragePool -> Linear(C->2)
# =============================================================================

# ---------------------------------------------------------------------------
# 1. 데이터 로딩
# ---------------------------------------------------------------------------
function load_split(root::String, split::String; sz=(64, 64))
    classes = ["ants", "bees"]
    imgs = Matrix{Float64}[]
    labels = Int[]
    for (ci, c) in enumerate(classes)
        dir = joinpath(root, split, c)
        for f in sort(readdir(dir))
            lf = lowercase(f)
            (endswith(lf, ".jpg") || endswith(lf, ".jpeg") || endswith(lf, ".png")) || continue
            img = Float64.(Gray.(load(joinpath(dir, f))))
            img = Float64.(imresize(img, sz))
            push!(imgs, img)
            push!(labels, ci)   # 1=ants, 2=bees
        end
    end
    X = zeros(Float64, sz[1], sz[2], 1, length(imgs))
    for (i, im) in enumerate(imgs)
        X[:, :, 1, i] = im
    end
    return X, labels
end

onehot(y::Vector{Int}, K::Int) = begin
    Y = zeros(K, length(y))
    for (i, yi) in enumerate(y)
        Y[yi, i] = 1.0
    end
    Y
end

# ---------------------------------------------------------------------------
# 2. 공간 영역 모델 (벡터화된 순수 배열 연산 -> Zygote 자동미분 가능)
# ---------------------------------------------------------------------------
function pad_hw(x::Array{Float64,4}, p::Int)
    H, W, C, N = size(x)
    top = zeros(Float64, p, W, C, N)
    xp = cat(top, x, top; dims=1)
    left = zeros(Float64, H + 2p, p, C, N)
    xp = cat(left, xp, left; dims=2)
    return xp
end

"""3x3 'same' 합성곱(Cin->Cout). 9개의 이동된(shifted) 슬라이스에 대한 행렬곱의
합으로 구현하여 im2col 없이도 BLAS 행렬곱 속도와 Zygote 미분가능성을 모두 얻는다."""
function conv3x3(x::Array{Float64,4}, W::Array{Float64,4}, b::Vector{Float64})
    H, Wd, Cin, N = size(x)
    Cout = size(W, 4)
    xp = pad_hw(x, 1)
    out = zeros(Float64, H, Wd, Cout, N)
    for di in 0:2, dj in 0:2
        shifted = xp[1+di:H+di, 1+dj:Wd+dj, :, :]
        shifted2 = reshape(permutedims(shifted, (1, 2, 4, 3)), H * Wd * N, Cin)
        wslice = W[di+1, dj+1, :, :]
        contrib = shifted2 * wslice
        contrib = permutedims(reshape(contrib, H, Wd, N, Cout), (1, 2, 4, 3))
        out = out .+ contrib
    end
    return out .+ reshape(b, 1, 1, Cout, 1)
end

"""배치+공간 차원에 대해 채널별 평균/분산을 구해 정규화(학습 모드 BatchNorm)."""
function batchnorm_stats(x::Array{Float64,4}, γ::Vector{Float64}, β::Vector{Float64}; eps::Float64=1e-5)
    μ = mean(x, dims=(1, 2, 4))
    σ2 = mean((x .- μ) .^ 2, dims=(1, 2, 4))
    x̂ = (x .- μ) ./ sqrt.(σ2 .+ eps)
    out = x̂ .* reshape(γ, 1, 1, :, 1) .+ reshape(β, 1, 1, :, 1)
    return out, dropdims(μ, dims=(1, 2, 4)), dropdims(σ2, dims=(1, 2, 4))
end

"""추론 모드 BatchNorm: 학습 중 집계한 고정된 러닝 평균/분산 사용."""
function apply_bn_fixed(x::Array{Float64,4}, γ::Vector{Float64}, β::Vector{Float64},
    μ::Vector{Float64}, σ2::Vector{Float64}; eps::Float64=1e-5)
    return (x .- reshape(μ, 1, 1, :, 1)) ./ sqrt.(reshape(σ2, 1, 1, :, 1) .+ eps) .*
           reshape(γ, 1, 1, :, 1) .+ reshape(β, 1, 1, :, 1)
end

function init_params(C::Int; rng=Random.default_rng())
    W1 = 0.15 .* randn(rng, 3, 3, 1, C)
    W2 = (0.15 / sqrt(C)) .* randn(rng, 3, 3, C, C)
    Wfc = 0.1 .* randn(rng, 2, C)
    return (W1=W1, b1=zeros(C), γ1=ones(C), β1=zeros(C),
        W2=W2, b2=zeros(C), γ2=ones(C), β2=zeros(C),
        Wfc=Wfc, bfc=zeros(2))
end

"""학습 모드 순전파 (배치 통계로 BN, loss 계산/역전파용)."""
function model_forward(p, x::Array{Float64,4})
    h1_pre = conv3x3(x, p.W1, p.b1)
    h1_bn, = batchnorm_stats(h1_pre, p.γ1, p.β1)
    h1 = max.(h1_bn, 0.0)
    h2_pre = conv3x3(h1, p.W2, p.b2)
    h2_bn, = batchnorm_stats(h2_pre, p.γ2, p.β2)
    h3 = max.(h2_bn .+ h1, 0.0)
    pooled = dropdims(mean(h3, dims=(1, 2)), dims=(1, 2))
    return p.Wfc * pooled .+ p.bfc
end

"""학습 완료 후 최종 BN 러닝 통계(전체 학습셋 기준)를 한 번에 추출."""
function compute_bn_stats(p, x::Array{Float64,4})
    h1_pre = conv3x3(x, p.W1, p.b1)
    h1_bn, μ1, σ1 = batchnorm_stats(h1_pre, p.γ1, p.β1)
    h1 = max.(h1_bn, 0.0)
    h2_pre = conv3x3(h1, p.W2, p.b2)
    _, μ2, σ2 = batchnorm_stats(h2_pre, p.γ2, p.β2)
    return (μ1=μ1, σ1=σ1, μ2=μ2, σ2=σ2)
end

"""추론 모드 순전파 (고정 러닝 통계 사용) - 공간 영역 검증/평가용."""
function model_forward_eval(p, x::Array{Float64,4}, bn)
    h1_pre = conv3x3(x, p.W1, p.b1)
    h1 = max.(apply_bn_fixed(h1_pre, p.γ1, p.β1, bn.μ1, bn.σ1), 0.0)
    h2_pre = conv3x3(h1, p.W2, p.b2)
    h3 = max.(apply_bn_fixed(h2_pre, p.γ2, p.β2, bn.μ2, bn.σ2) .+ h1, 0.0)
    pooled = dropdims(mean(h3, dims=(1, 2)), dims=(1, 2))
    return p.Wfc * pooled .+ p.bfc
end

function softmax_ce_loss(logits::Matrix{Float64}, Y::Matrix{Float64})
    ls = logits .- maximum(logits, dims=1)
    exps = exp.(ls)
    probs = exps ./ sum(exps, dims=1)
    return -sum(Y .* log.(probs .+ 1e-12)) / size(logits, 2)
end

function predictions(logits::Matrix{Float64})
    [argmax(logits[:, i]) for i in 1:size(logits, 2)]
end
accuracy(logits::Matrix{Float64}, y::Vector{Int}) = mean(predictions(logits) .== y)

# ---------------------------------------------------------------------------
# 3. Adam 옵티마이저 (NamedTuple 파라미터를 그대로 다룸)
# ---------------------------------------------------------------------------
function adam_step(p::NamedTuple, g::NamedTuple, state; lr::Float64=1e-2, β1::Float64=0.9, β2::Float64=0.999, eps::Float64=1e-8)
    t = state.t + 1
    ks = keys(p)
    newm = NamedTuple{ks}(Tuple(β1 .* state.m[k] .+ (1 - β1) .* g[k] for k in ks))
    newv = NamedTuple{ks}(Tuple(β2 .* state.v[k] .+ (1 - β2) .* (g[k] .^ 2) for k in ks))
    mhat = NamedTuple{ks}(Tuple(newm[k] ./ (1 - β1^t) for k in ks))
    vhat = NamedTuple{ks}(Tuple(newv[k] ./ (1 - β2^t) for k in ks))
    newp = NamedTuple{ks}(Tuple(p[k] .- lr .* mhat[k] ./ (sqrt.(vhat[k]) .+ eps) for k in ks))
    return newp, (m=newm, v=newv, t=t)
end

function train!(p, Xtr::Array{Float64,4}, Ytr::Matrix{Float64}, ytr::Vector{Int}; epochs::Int=150, lr::Float64=1e-2)
    ks = keys(p)
    state = (m=NamedTuple{ks}(Tuple(zero.(values(p)))), v=NamedTuple{ks}(Tuple(zero.(values(p)))), t=0)
    for epoch in 1:epochs
        loss, back = Zygote.pullback(pp -> softmax_ce_loss(model_forward(pp, Xtr), Ytr), p)
        grads = back(1.0)[1]
        p, state = adam_step(p, grads, state; lr=lr)
        if epoch % 25 == 0 || epoch == 1
            acc = accuracy(model_forward(p, Xtr), ytr)
            println("  epoch $(rpad(epoch,4)) loss=$(round(loss,digits=4))  train acc=$(round(acc,digits=3))")
        end
    end
    return p
end

# ---------------------------------------------------------------------------
# 4. JPEG 변환 영역으로 모델 변환 (Section 4.6 Model Conversion) 및 순전파
#    jpeg_domain_resnet.jl에서 이미 수치 검증된 연산들(jpeg_domain_conv,
#    dct_domain_batchnorm, asm_relu_block, global_avg_from_dc)을 그대로 재사용.
# ---------------------------------------------------------------------------
function jpeg_conv_multichannel(Js::Vector{Matrix{Float64}}, W::Array{Float64,4}, b::Vector{Float64}; bs::Int=8)
    Cin = length(Js)
    Cout = size(W, 4)
    rows, cols = size(Js[1])
    outs = Vector{Matrix{Float64}}(undef, Cout)
    for o in 1:Cout
        acc = zeros(Float64, rows, cols)
        for i in 1:Cin
            acc .+= jpeg_domain_conv(Js[i], W[:, :, i, o]; bs=bs)
        end
        for by in 0:bs:rows-1, bx in 0:bs:cols-1
            acc[by+1, bx+1] += b[o] * bs   # 편향은 모든 픽셀에 상수를 더하는 것 -> DC 계수만 이동
        end
        outs[o] = acc
    end
    return outs
end

function jpeg_bn_multichannel(Js::Vector{Matrix{Float64}}, γ::Vector{Float64}, β::Vector{Float64},
    μ::Vector{Float64}, σ2::Vector{Float64}; bs::Int=8)
    [dct_domain_batchnorm(Js[c], γ[c], β[c], μ[c], σ2[c]; bs=bs) for c in eachindex(Js)]
end

function jpeg_relu_multichannel(Js::Vector{Matrix{Float64}}, m::Int; bs::Int=8)
    zz = zigzag_order(bs)
    out = Vector{Matrix{Float64}}(undef, length(Js))
    for c in eachindex(Js)
        J = Js[c]
        rows, cols = size(J)
        Jr = similar(J)
        for by in 0:bs:rows-1, bx in 0:bs:cols-1
            Jr[by+1:by+bs, bx+1:bx+bs] = asm_relu_block(J[by+1:by+bs, bx+1:bx+bs], m; zz=zz)
        end
        out[c] = Jr
    end
    return out
end

jpeg_add_multichannel(As::Vector{Matrix{Float64}}, Bs::Vector{Matrix{Float64}}) = [As[c] .+ Bs[c] for c in eachindex(As)]

function jpeg_gap_multichannel(Js::Vector{Matrix{Float64}}; bs::Int=8)
    [global_avg_from_dc(J; bs=bs) for J in Js]
end

"""단일 그레이스케일 영상에 대해 학습된 가중치를 그대로 JPEG 변환 영역에서
순전파한다 (Conv->BN->ASM-ReLU->+skip->Conv... 는 이미 위에서 GAP까지 완료,
여기서는 한 번의 잔차 블록 구조를 그대로 따른다)."""
function jpeg_domain_forward_single(p, bn, img::Matrix{Float64}; bs::Int=8, m::Int=15)
    J0 = [dct2_blocks(img, bs)]                                   # 입력 1채널
    J1 = jpeg_conv_multichannel(J0, p.W1, p.b1; bs=bs)              # Conv1: 1->C
    J1 = jpeg_bn_multichannel(J1, p.γ1, p.β1, bn.μ1, bn.σ1; bs=bs)  # BN1
    J1 = jpeg_relu_multichannel(J1, m; bs=bs)                        # ASM-ReLU1  (h1)
    J2 = jpeg_conv_multichannel(J1, p.W2, p.b2; bs=bs)              # Conv2: C->C
    J2 = jpeg_bn_multichannel(J2, p.γ2, p.β2, bn.μ2, bn.σ2; bs=bs)  # BN2
    J3 = jpeg_add_multichannel(J2, J1)                                # 잔차 덧셈 (식 25)
    J3 = jpeg_relu_multichannel(J3, m; bs=bs)                         # 마지막 ReLU
    pooled = jpeg_gap_multichannel(J3; bs=bs)                         # GAP (DC 계수 평균, Section 4.5)
    return p.Wfc * pooled .+ p.bfc
end

function jpeg_domain_accuracy(p, bn, X::Array{Float64,4}, y::Vector{Int}; bs::Int=8, m::Int=15)
    N = size(X, 4)
    preds = Vector{Int}(undef, N)
    for i in 1:N
        logits = jpeg_domain_forward_single(p, bn, X[:, :, 1, i]; bs=bs, m=m)
        preds[i] = argmax(logits)
    end
    return mean(preds .== y), preds
end

# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------
function main()
    println("="^76)
    println("Hymenoptera(개미/꿀벌) 분류: 공간 영역 학습 -> JPEG 변환 영역 모델 변환")
    println("(Ehrlich & Davis, ICCV 2019, Section 4.6 Model Conversion 재현)")
    println("="^76)

    rng = Random.MersenneTwister(42)
    root = joinpath(@__DIR__, "..", "hymenoptera_data")
    sz = (64, 64)
    C = 8

    println("\n[1] 데이터 로딩 (", sz, " 그레이스케일로 리사이즈)")
    Xtr, ytr = load_split(root, "train"; sz=sz)
    Xval, yval = load_split(root, "val"; sz=sz)
    println("  train: ", size(Xtr, 4), "장 (ants=", count(==(1), ytr), ", bees=", count(==(2), ytr), ")")
    println("  val:   ", size(Xval, 4), "장 (ants=", count(==(1), yval), ", bees=", count(==(2), yval), ")")
    Ytr = onehot(ytr, 2)

    println("\n[2] 공간 영역에서 잔차 블록(Conv-BN-ReLU-Conv-BN-Add-ReLU) 학습 (Zygote 자동미분, Adam)")
    p = init_params(C; rng=rng)
    p = train!(p, Xtr, Ytr, ytr; epochs=150, lr=1e-2)

    bn = compute_bn_stats(p, Xtr)   # 학습셋 전체로 최종 BatchNorm 러닝 통계 확정

    println("\n[3] 공간 영역(spatial) 최종 정확도")
    train_acc = accuracy(model_forward_eval(p, Xtr, bn), ytr)
    val_acc = accuracy(model_forward_eval(p, Xval, bn), yval)
    println("  train accuracy = ", round(train_acc, digits=4))
    println("  val   accuracy = ", round(val_acc, digits=4))

    println("\n[4] 모델 변환(Section 4.6): 동일 가중치를 JPEG 변환 영역에서 순전파, m별 정확도 비교")
    println("  ", rpad("m (ASM 주파수 수)", 20), rpad("JPEG val acc", 16), "spatial과의 편차")
    for m in (4, 8, 15, 32, 64)
        jacc, jpreds = jpeg_domain_accuracy(p, bn, Xval, yval; m=m)
        println("  ", rpad(string(m), 20), rpad(string(round(jacc, digits=4)), 16),
            round(abs(jacc - val_acc), digits=4))
    end

    println("\n[5] 개별 검증 영상 분류 예시 (JPEG 변환 영역, m=15 추론)")
    jacc15, jpreds15 = jpeg_domain_accuracy(p, bn, Xval, yval; m=15)
    class_names = ["ant", "bee"]
    nshow = min(10, length(yval))
    idxs = round.(Int, range(1, length(yval), length=nshow))
    for i in idxs
        mark = jpreds15[i] == yval[i] ? "O" : "X"
        println("  [$mark] 정답=", rpad(class_names[yval[i]], 4), " JPEG영역예측=", class_names[jpreds15[i]])
    end
    println("  (JPEG 영역, m=15) val accuracy = ", round(jacc15, digits=4),
        "   / 총 ", length(yval), "장 중 ", count(jpreds15 .== yval), "장 정답")

    # 예측 결과 몇 장을 이미지로 저장 (정답 O/X 라벨과 함께 그리드)
    println("\n[6] 예측 결과 시각화 저장")
    tile = 72
    grid_imgs = RGB{Float64}[]
    ncols = 5
    rows_grid = Matrix{RGB{Float64}}[]
    for r in 0:1
        row_tiles = Matrix{RGB{Float64}}[]
        for c in 1:ncols
            i = idxs[r*ncols+c]
            base = Gray.(Xval[:, :, 1, i])
            rgb = RGB.(base)
            correct = jpreds15[i] == yval[i]
            border = correct ? RGB(0.0, 0.9, 0.0) : RGB(0.9, 0.0, 0.0)
            canvas = fill(border, sz[1] + 8, sz[2] + 8)
            canvas[5:4+sz[1], 5:4+sz[2]] = rgb
            push!(row_tiles, canvas)
        end
        push!(rows_grid, hcat(row_tiles...))
    end
    mosaic = vcat(rows_grid...)
    save("jpegresnet_bees_ants_predictions.png", mosaic)
    println("  저장됨: jpegresnet_bees_ants_predictions.png (녹색 테두리=정답, 빨강 테두리=오답)")

    println("\n완료.")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
