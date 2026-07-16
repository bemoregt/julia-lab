using Images
using TestImages
using FFTW
using Random
using Statistics: mean

# =============================================================================
# Global Filter Networks (GFNet)
#   Y. Rao, W. Zhao, Z. Zhu, J. Lu, J. Zhou, "Global Filter Networks for Image
#   Classification," NeurIPS 2021.  arXiv:2107.00645
#
# 비전 트랜스포머(ViT)의 자기주의(self-attention, O(L²)) 계층을 다음 세 연산으로
# 대체한 아키텍처:
#   1) 2D 실수 FFT       X  = F_r[x]        (식 3.4, 3.7)
#   2) 주파수 영역 필터링  X̃  = K ⊙ X        (식 3.5, element-wise/Hadamard 곱)
#   3) 2D 역 실수 FFT     x' = F_r^{-1}[X̃]   (식 3.6)
# 여기서 K ∈ C^{H×Ŵ×D} (Ŵ=W÷2+1) 는 채널별 학습 가능한 "전역 필터"이다.
# FFT의 O(L log L) 복잡도 덕분에 self-attention/spatial-MLP의 O(L²)보다
# 훨씬 효율적이면서도, 필터가 전체 스펙트럼을 덮으므로 토큰 간 장거리
# 상호작용을 원본 데이터로부터 직접 학습할 수 있다 (Table 1, Algorithm 1).
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Global Filter Layer (Algorithm 1)
# ---------------------------------------------------------------------------

"""
GlobalFilterLayer

토큰 특징 x ∈ R^{H×W×D} 에 대한 전역 필터 계층. `K`는 (H, W÷2+1, D) 크기의
복소 텐서로, 채널마다 독립적인 주파수 영역 필터를 담는다. 실수 신호의 DFT는
켤레대칭(conjugate symmetric)이므로 절반의 스펙트럼(rfft2)만으로 전체 정보를
표현할 수 있어(식 3.7), 파라미터 수와 연산량을 절반으로 줄인다.
"""
struct GlobalFilterLayer
    K::Array{ComplexF64,3}   # (H, W÷2+1, D)
end

function GlobalFilterLayer(H::Int, W::Int, D::Int; rng=Random.default_rng(), scale::Float64=0.02)
    Ŵ = W ÷ 2 + 1
    K = complex.(scale .* randn(rng, H, Ŵ, D), scale .* randn(rng, H, Ŵ, D))
    return GlobalFilterLayer(K)
end

"""
    (layer::GlobalFilterLayer)(x) -> x'

Algorithm 1 그대로: rfft2 -> element-wise 곱(K) -> irfft2.
Julia FFTW의 `rfft(x, (2,1))`은 두 번째 축(W)을 W÷2+1로 절반화하여 논문의
(H, Ŵ, D) 레이아웃과 일치시킨다.
"""
function (layer::GlobalFilterLayer)(x::AbstractArray{<:Real,3})
    H, W, D = size(x)
    X = rfft(x, (2, 1))          # 식 (3.4)/(3.7)
    X̃ = layer.K .* X             # 식 (3.5)
    return irfft(X̃, W, (2, 1))   # 식 (3.6)
end

# ---------------------------------------------------------------------------
# 2. 전역 필터 ↔ 깊이별 전역 원형 합성곱(circular convolution) 동치성 (Appendix A)
#    주파수 영역에서의 곱셈은 공간 영역에서 필터의 역FFT 커널과의 원형 합성곱과
#    같다. 아래는 그 동치성을 직접(직접 합성곱, O(H²W²D))으로 검증하기 위한
#    참조 구현이며, GlobalFilterLayer가 실제로 사용하는 O(L log L) 경로와
#    독립적으로 계산한다.
# ---------------------------------------------------------------------------
function circular_conv2_depthwise(x::AbstractArray{<:Real,3}, kernel::AbstractArray{<:Real,3})
    H, W, D = size(x)
    y = zeros(Float64, H, W, D)
    @inbounds for d in 1:D, i in 0:H-1, j in 0:W-1
        s = 0.0
        for p in 0:H-1, q in 0:W-1
            s += kernel[p+1, q+1, d] * x[mod(i - p, H)+1, mod(j - q, W)+1, d]
        end
        y[i+1, j+1, d] = s
    end
    return y
end

# ---------------------------------------------------------------------------
# 3. LayerNorm / FeedForward / GFNetBlock (Figure 1: LN -> GlobalFilter -> FFN)
# ---------------------------------------------------------------------------

function layernorm(x::AbstractArray{<:Real,3}; eps::Float64=1e-6)
    μ = mean(x, dims=3)
    σ2 = mean((x .- μ) .^ 2, dims=3)
    return (x .- μ) ./ sqrt.(σ2 .+ eps)
end

gelu(x) = 0.5 .* x .* (1 .+ tanh.(sqrt(2 / π) .* (x .+ 0.044715 .* x .^ 3)))

struct FeedForward
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
end

function FeedForward(D::Int, hidden::Int; rng=Random.default_rng())
    W1 = sqrt(2 / D) .* randn(rng, hidden, D)
    W2 = sqrt(2 / hidden) .* randn(rng, D, hidden)
    return FeedForward(W1, zeros(hidden), W2, zeros(D))
end

function (ff::FeedForward)(x::AbstractArray{<:Real,3})
    H, W, D = size(x)
    xr = reshape(permutedims(x, (3, 1, 2)), D, H * W)
    h = gelu(ff.W1 * xr .+ ff.b1)
    o = ff.W2 * h .+ ff.b2
    return permutedims(reshape(o, D, H, W), (2, 3, 1))
end

"""
GFNetBlock: x = x + GlobalFilterLayer(LN(x));  x = x + FFN(LN(x))
(논문 Figure 1의 기본 빌딩 블록, self-attention 대신 전역 필터 계층 사용)
"""
struct GFNetBlock
    filter::GlobalFilterLayer
    ffn::FeedForward
end

function GFNetBlock(H::Int, W::Int, D::Int, hidden::Int; rng=Random.default_rng())
    return GFNetBlock(GlobalFilterLayer(H, W, D; rng=rng), FeedForward(D, hidden; rng=rng))
end

function (blk::GFNetBlock)(x::AbstractArray{<:Real,3})
    x = x .+ blk.filter(layernorm(x))
    x = x .+ blk.ffn(layernorm(x))
    return x
end

# ---------------------------------------------------------------------------
# 4. 전체 GFNet: Patch Embedding -> N x GFNetBlock -> Global Average Pool -> Linear
# ---------------------------------------------------------------------------

struct GFNet
    patch::Int
    Wproj::Matrix{Float64}
    blocks::Vector{GFNetBlock}
    head::Matrix{Float64}
    headb::Vector{Float64}
end

function GFNet(; img_size::Int=64, patch::Int=8, D::Int=32, hidden::Int=64,
    depth::Int=4, num_classes::Int=10, rng=Random.default_rng())
    Hp = Wp = img_size ÷ patch
    Wproj = 0.02 .* randn(rng, D, patch * patch)
    blocks = [GFNetBlock(Hp, Wp, D, hidden; rng=rng) for _ in 1:depth]
    head = 0.02 .* randn(rng, num_classes, D)
    headb = zeros(num_classes)
    return GFNet(patch, Wproj, blocks, head, headb)
end

function (m::GFNet)(img::AbstractMatrix{<:Real})
    rows, cols = size(img)
    Hp, Wp = rows ÷ m.patch, cols ÷ m.patch
    D = size(m.Wproj, 1)
    tokens = zeros(Float64, Hp, Wp, D)
    for i in 1:Hp, j in 1:Wp
        blk = img[(i-1)*m.patch+1:i*m.patch, (j-1)*m.patch+1:j*m.patch]
        tokens[i, j, :] = m.Wproj * vec(blk)
    end
    x = tokens
    for blk in m.blocks
        x = blk(x)
    end
    pooled = vec(mean(x, dims=(1, 2)))   # global average pooling
    return m.head * pooled .+ m.headb
end

# ---------------------------------------------------------------------------
# 5. 계산 복잡도 벤치마크: Global Filter O(L log L) vs Spatial MLP O(L²) (Table 1)
#    side=512 (실제 샘플 이미지 해상도)까지 포함. Spatial MLP의 L×L 혼합 행렬은
#    512×512 격자에서 512GB 넘게 필요해 실제 할당이 불가능하므로, `max_mlp_bytes`를
#    넘는 크기는 실제 할당/연산을 건너뛰고 이론적 메모리량만 보고한다. 이 자체가
#    논문이 지적하는 "self-attention/spatial-MLP는 고해상도에서 메모리 부족으로
#    실행 불가"라는 핵심 동기를 그대로 재현한다.
# ---------------------------------------------------------------------------
function benchmark_complexity(sides=(8, 16, 32, 64, 512); D::Int=16, reps::Int=5, max_mlp_bytes::Float64=4e9)
    results = NamedTuple[]
    for side in sides
        x = randn(side, side, D)
        gfl = GlobalFilterLayer(side, side, D)
        gfl(x)  # warm-up (JIT)
        t_gf = @elapsed for _ in 1:reps
            gfl(x)
        end
        t_gf /= reps

        ntok = side * side
        mlp_bytes_needed = Float64(ntok)^2 * 8   # Wsp::Matrix{Float64}, ntok×ntok
        if mlp_bytes_needed <= max_mlp_bytes
            Wsp = 0.01 .* randn(ntok, ntok)
            xr = reshape(x, ntok, D)
            Wsp * xr  # warm-up
            t_mlp = @elapsed for _ in 1:reps
                Wsp * xr
            end
            t_mlp /= reps
            mlp_feasible = true
        else
            t_mlp = NaN
            mlp_feasible = false
        end

        push!(results, (side=side, tokens=ntok, gf_ms=t_gf * 1000,
            mlp_ms=t_mlp * 1000, mlp_feasible=mlp_feasible, mlp_bytes_needed=mlp_bytes_needed))
    end
    return results
end

# ---------------------------------------------------------------------------
# 6. 메모리 점유 비교: Global Filter 파라미터 O(HWD) vs Spatial MLP 파라미터 O(H²W²)
#    (Table 1의 "# Parameters" 열에 해당) + 순전파 1회당 실제 런타임 할당량(@allocated)
#    마찬가지로 side=512에서 Spatial MLP 가중치는 실제로 할당하지 않고 이론값만 계산.
# ---------------------------------------------------------------------------
function benchmark_memory(sides=(8, 16, 32, 64, 512); D::Int=16, max_mlp_bytes::Float64=4e9)
    results = NamedTuple[]
    for side in sides
        Ŵ = side ÷ 2 + 1
        gfl = GlobalFilterLayer(side, side, D)
        x = randn(side, side, D)
        gfl(x)  # warm-up (JIT) before measuring allocations

        gf_param_bytes = sizeof(gfl.K)                 # ComplexF64 저장 실제 바이트 수
        gf_param_count = length(gfl.K) * 2              # 실수부+허수부 (논문 HWD 근사)
        gf_alloc_bytes = @allocated gfl(x)               # 순전파 1회 런타임 할당량

        ntok = side * side
        mlp_param_count = ntok^2                         # 논문 H²W² (채널 간 공유)
        mlp_param_bytes = Float64(mlp_param_count) * 8

        if mlp_param_bytes <= max_mlp_bytes
            Wsp = 0.01 .* randn(ntok, ntok)
            xr = reshape(x, ntok, D)
            Wsp * xr  # warm-up
            mlp_alloc_bytes = @allocated Wsp * xr
            mlp_feasible = true
        else
            mlp_alloc_bytes = NaN
            mlp_feasible = false
        end

        push!(results, (side=side, Ŵ=Ŵ, tokens=ntok,
            gf_params=gf_param_count, gf_param_kb=gf_param_bytes / 1024, gf_alloc_kb=gf_alloc_bytes / 1024,
            mlp_params=mlp_param_count, mlp_param_kb=mlp_param_bytes / 1024,
            mlp_alloc_kb=mlp_feasible ? mlp_alloc_bytes / 1024 : NaN, mlp_feasible=mlp_feasible))
    end
    return results
end

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
normalize01(x) = begin
    lo, hi = extrema(x)
    (x .- lo) ./ (hi - lo + eps())
end

function humanize_kb(kb::Real)
    isnan(kb) && return "N/A"
    kb < 1024 && return "$(round(kb, digits=1))KB"
    kb < 1024^2 && return "$(round(kb / 1024, digits=2))MB"
    return "$(round(kb / 1024^2, digits=2))GB"
end

function main()
    println("="^72)
    println("Global Filter Network (GFNet) - Rao et al., NeurIPS 2021, arXiv:2107.00645")
    println("="^72)

    # --- [1] 항등 필터(K=1) 왕복 재구성 오차: rfft2/irfft2 파이프라인 정확성 검증 ---
    println("\n[1] 항등 필터(K=1) 왕복 재구성 오차")
    rng = Random.MersenneTwister(0)
    x1 = randn(rng, 8, 10, 4)
    id_layer = GlobalFilterLayer(ones(ComplexF64, 8, 10 ÷ 2 + 1, 4))
    y1 = id_layer(x1)
    println("  max|x - GlobalFilter(x; K=1)| = ", maximum(abs.(x1 .- y1)))

    # --- [2] 전역 필터 ↔ 깊이별 전역 원형 합성곱 동치성 (Appendix A) ---
    println("\n[2] 전역 필터(FFT 경로) ↔ 전역 원형 합성곱(공간 경로) 동치성 검증")
    H, W, D = 8, 8, 2
    kernel = 0.3 .* randn(rng, H, W, D)
    Kspec = rfft(kernel, (2, 1))
    freq_layer = GlobalFilterLayer(Kspec)
    xt = randn(rng, H, W, D)
    y_freq = freq_layer(xt)
    y_spatial = circular_conv2_depthwise(xt, kernel)
    println("  max|FFT 경로 - 원형합성곱 경로| = ", maximum(abs.(y_freq .- y_spatial)))

    # --- [3] 계산 복잡도 비교: O(L log L) vs O(L²), 512×512(실제 이미지 해상도) 포함 ---
    println("\n[3] 계산 복잡도 비교: Global Filter O(L log L)  vs  Spatial MLP O(L²)")
    results = benchmark_complexity()
    println("  ", rpad("H×W", 10), rpad("tokens L", 10), rpad("GlobalFilter(ms)", 20), "SpatialMLP(ms)")
    for r in results
        mlp_str = r.mlp_feasible ? string(round(r.mlp_ms, digits=3)) :
                  "OOM (혼합행렬 이론상 $(round(r.mlp_bytes_needed / 1e9, digits=1))GB 필요)"
        println("  ", rpad("$(r.side)×$(r.side)", 10), rpad(string(r.tokens), 10),
            rpad(string(round(r.gf_ms, digits=3)), 20), mlp_str)
    end

    # --- [4] 메모리 점유 비교: 파라미터 O(HWD) vs O(H²W²) + 순전파 런타임 할당량, 512×512 포함 ---
    println("\n[4] 메모리 점유 비교: Global Filter 파라미터 O(HWD)  vs  Spatial MLP 파라미터 O(H²W²)")
    mresults = benchmark_memory()
    println("  ", rpad("H×W", 8), rpad("GF params", 12), rpad("GF weight", 12), rpad("GF/call alloc", 15),
        rpad("MLP params", 16), rpad("MLP weight", 12), "MLP/call alloc")
    for r in mresults
        gf_w = humanize_kb(r.gf_param_kb)
        mlp_w = humanize_kb(r.mlp_param_kb) * (r.mlp_feasible ? "" : " (이론값, 미할당)")
        gf_a = humanize_kb(r.gf_alloc_kb)
        mlp_a = r.mlp_feasible ? humanize_kb(r.mlp_alloc_kb) : "N/A"
        println("  ", rpad("$(r.side)×$(r.side)", 8), rpad(string(r.gf_params), 12), rpad(gf_w, 12), rpad(gf_a, 15),
            rpad(string(r.mlp_params), 16), mlp_w, "  ", mlp_a)
    end
    last64 = mresults[findlast(r -> r.side == 64, mresults)]
    println("  -> 64×64(D=16) 기준 Spatial MLP 가중치가 Global Filter보다 ",
        round(last64.mlp_param_kb / last64.gf_param_kb, digits=1), "배 더 많은 메모리 사용")
    last512 = mresults[findlast(r -> r.side == 512, mresults)]
    if !last512.mlp_feasible
        println("  -> 512×512(D=16, 실제 샘플 이미지 해상도) 기준 Spatial MLP 혼합행렬은 ",
            round(last512.mlp_param_kb / 1024^2, digits=1),
            "GB가 필요해 이 시스템(16GB RAM)에서 실제 할당 자체가 불가능함",
            " (반면 Global Filter는 ", humanize_kb(last512.gf_param_kb), "면 충분)")
    end

    # --- [5] 실제 영상에 대한 전역 필터(저역/고역) 시각화 ---
    println("\n[5] 실제 영상에 대한 전역 필터(저역/고역) 시각화")
    img = Float64.(Gray.(testimage("cameraman")))
    rows, cols = size(img)
    x_img = reshape(img, rows, cols, 1)

    Ŵ = cols ÷ 2 + 1
    urow = [min(i - 1, rows - i + 1) / rows for i in 1:rows]
    vcol = [(j - 1) / cols for j in 1:Ŵ]
    radius = [hypot(urow[i], vcol[j]) for i in 1:rows, j in 1:Ŵ]
    cutoff = 0.08
    Klow = 1.0 ./ (1.0 .+ (radius ./ cutoff) .^ 4)   # Butterworth형 저역통과
    Khigh = 1.0 .- Klow                              # 상보적 고역통과

    layer_low = GlobalFilterLayer(complex.(reshape(Klow, rows, Ŵ, 1)))
    layer_high = GlobalFilterLayer(complex.(reshape(Khigh, rows, Ŵ, 1)))

    layer_low(x_img)  # warm-up (JIT)
    t_512 = @elapsed layer_low(x_img)
    alloc_512 = @allocated layer_low(x_img)

    img_low = clamp01.(dropdims(layer_low(x_img), dims=3))
    img_high = normalize01(dropdims(layer_high(x_img), dims=3))

    save("gfnet_source.png", Gray.(img))
    save("gfnet_filter_lowpass.png", Gray.(img_low))
    save("gfnet_filter_highpass.png", Gray.(img_high))
    save("gfnet_filter_comparison.png", hcat(Gray.(img), Gray.(img_low), Gray.(img_high)))

    edge_orig = sum(abs2, diff(img, dims=1))
    edge_low = sum(abs2, diff(img_low, dims=1))
    edge_high = sum(abs2, diff(img_high, dims=1))
    println("  이미지 크기(H×W)      = ", (rows, cols), " (D=1 채널)")
    println("  1회 순전파 소요시간    = ", round(t_512 * 1000, digits=3), "ms",
        "  (실제 512×512 해상도, rfft2->K⊙->irfft2)")
    println("  1회 순전파 런타임 할당 = ", humanize_kb(alloc_512 / 1024))
    println("  edge energy(원본)     = ", round(edge_orig, digits=2))
    println("  edge energy(저역필터)  = ", round(edge_low, digits=2))
    println("  edge energy(고역필터)  = ", round(edge_high, digits=2))
    println("  저장됨: gfnet_source.png, gfnet_filter_lowpass.png,",
        " gfnet_filter_highpass.png, gfnet_filter_comparison.png")
    println("  참고: 동일한 512×512 해상도에서 self-attention/spatial-MLP류의",
        " O(L²) 토큰 혼합 행렬은 [4]에서 보듯 수백GB 규모라 이 이미지 크기",
        " 자체를 직접 처리할 수 없음 - GFNet의 핵심 동기(log-linear 복잡도)가",
        " 정확히 이 지점에서 발생.")

    # --- [6] 미니 GFNet 순전파: patch embedding -> N x GFNetBlock -> GAP -> Linear ---
    println("\n[6] 미니 GFNet 순전파 (patch embedding -> 4×GFNetBlock -> GAP -> Linear)")
    model = GFNet(img_size=64, patch=8, D=32, hidden=64, depth=4, num_classes=10, rng=rng)
    img64 = Float64.(Gray.(imresize(testimage("cameraman"), (64, 64))))
    logits = model(img64)
    println("  입력 이미지 크기: ", size(img64), "  패치 크기: 8×8 -> 토큰 격자 8×8, D=32, depth=4")
    println("  출력 로짓(10-way, 미학습 가중치): ", round.(logits, digits=4))

    println("\n완료.")
end

main()
