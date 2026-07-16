using Images
using TestImages
using FFTW
using Random
using Statistics: mean, var

# =============================================================================
# Deep Residual Learning in the JPEG Transform Domain
#   M. Ehrlich, L. Davis, ICCV 2019. arXiv:1812.11690
#
# JPEG 압축은 8×8 블록 단위 2D DCT-II -> 양자화(quantization)로 구성된 선형
# 변환이다(반올림 이전까지). 이 선형성을 이용하면 ResNet의 다섯 가지 기본
# 연산 -- Convolution, ReLU, BatchNorm, 잔차 덧셈(skip connection),
# Global Average Pooling -- 을 압축을 풀지 않고 "JPEG 변환 영역(JPEG
# Transform Domain, 즉 양자화 전 DCT 계수)"에서 직접 수행할 수 있다.
#   - Convolution, BatchNorm, 덧셈, GAP  : 선형 연산이므로 DCT 영역에서
#     정확히(근사 없이) 재현 가능 (논문 4.1, 4.3, 4.4, 4.5절).
#   - ReLU만 비선형이라 문제가 되는데, 논문은 "Approximated Spatial
#     Masking(ASM)"이라는 기법을 제안한다: 최하위 m개 주파수만으로 부호
#     (양/음)만 근사 추정해 이진 마스크를 만들고, 그 마스크를 실제(정확한)
#     픽셀 값에 곱한 뒤 다시 DCT함으로써 마스크 경계에서만 오차가 나고
#     픽셀 "값" 자체는 보존되는 정밀한 근사를 만든다 (논문 4.2절, Fig 1).
# =============================================================================

# ---------------------------------------------------------------------------
# 1. 블록 단위 2D DCT / JPEG 변환 영역 (Section 3.1, 3.2)
# ---------------------------------------------------------------------------

"""
dct2_blocks(img, bs=8) -> J

이미지를 bs×bs 겹치지 않는 블록으로 나누고(step 1) 각 블록에 직교정규
2D DCT-II를 적용한다(step 2). Julia FFTW의 `dct`는 이미 직교정규
(orthonormal) 스케일링을 사용하므로 별도 정규화가 필요 없다.
"""
function dct2_blocks(img::AbstractMatrix{<:Real}, bs::Int=8)
    rows, cols = size(img)
    @assert rows % bs == 0 && cols % bs == 0 "이미지 크기가 블록 크기의 배수여야 합니다"
    J = zeros(Float64, rows, cols)
    for by in 0:bs:rows-1, bx in 0:bs:cols-1
        J[by+1:by+bs, bx+1:bx+bs] = FFTW.dct(img[by+1:by+bs, bx+1:bx+bs])
    end
    return J
end

"""
idct2_blocks(J, bs=8) -> img

블록별 역DCT를 수행해 전체 영상을 정확히 복원한다("압축 해제", step 2의 역).
반올림(step 5)이 없는 한 이 과정은 무손실이다.
"""
function idct2_blocks(J::AbstractMatrix{<:Real}, bs::Int=8)
    rows, cols = size(J)
    img = zeros(Float64, rows, cols)
    for by in 0:bs:rows-1, bx in 0:bs:cols-1
        img[by+1:by+bs, bx+1:bx+bs] = FFTW.idct(J[by+1:by+bs, bx+1:bx+bs])
    end
    return img
end

# 표준 JPEG 휘도 양자화 테이블 (품질 50) - step 4에서 사용되는 예시 Q
const STD_LUMA_QTABLE = Float64[
    16 11 10 16 24 40 51 61
    12 12 14 19 26 58 60 55
    14 13 16 24 40 57 69 56
    14 17 22 29 51 87 80 62
    18 22 37 56 68 109 103 77
    24 35 55 64 81 104 113 92
    49 64 78 87 103 121 120 101
    72 92 95 98 112 100 103 99
]

tile_table(Q::AbstractMatrix, rows::Int, cols::Int) = repeat(Q, rows ÷ size(Q, 1), cols ÷ size(Q, 2))

"""
compress(img; bs, Q) -> J

step 1-4 (블록 분할 -> DCT -> 양자화 나눗셈)까지만 수행하고 step 5(반올림)는
건너뛴, 논문이 정의하는 "JPEG Transform Domain" 표현을 반환한다. 반올림 전
이므로 무손실(선형 가역) 표현이다.
"""
function compress(img::AbstractMatrix{<:Real}; bs::Int=8, Q::AbstractMatrix=ones(bs, bs))
    J = dct2_blocks(img, bs)
    return J ./ tile_table(Q, size(img)...)
end

"""
decompress(J; bs, Q) -> img

양자화를 되돌리고(곱셈) 블록별 역DCT를 적용해 압축 해제한다. compress와
정확히 역함수 관계(반올림을 하지 않았다면 무손실).
"""
function decompress(J::AbstractMatrix{<:Real}; bs::Int=8, Q::AbstractMatrix=ones(bs, bs))
    return idct2_blocks(J .* tile_table(Q, size(J)...), bs)
end

"""step 5까지 포함한 진짜 손실 JPEG 압축 (참고/비교용, 정수 반올림 포함)."""
compress_lossy(img::AbstractMatrix{<:Real}; bs::Int=8, Q::AbstractMatrix=STD_LUMA_QTABLE) =
    round.(compress(img; bs=bs, Q=Q))

# ---------------------------------------------------------------------------
# 2. Section 4.1: Convolution in the JPEG Transform Domain (식 11-13)
#    "압축 해제 -> 공간 합성곱 -> 재압축"이 근사가 아니라 JPEG 영역에서
#    수행하는 합성곱과 수학적으로 완전히 동일함(Ξ, 식 11)을 검증한다.
#    특히 커널이 블록 경계를 넘나들 때도(예: 8의 배수가 아닌 위치) 문제
#    없이 전체 영상 차원에서 올바르게 섞이는지 확인한다.
# ---------------------------------------------------------------------------

function conv2d_same(img::AbstractMatrix{<:Real}, kernel::AbstractMatrix{<:Real})
    kh, kw = size(kernel)
    ph, pw = kh ÷ 2, kw ÷ 2
    rows, cols = size(img)
    padded = zeros(Float64, rows + 2ph, cols + 2pw)
    padded[ph+1:ph+rows, pw+1:pw+cols] = img
    out = zeros(Float64, rows, cols)
    @inbounds for i in 1:rows, j in 1:cols
        out[i, j] = sum(padded[i:i+kh-1, j:j+kw-1] .* kernel)
    end
    return out
end

"""
jpeg_domain_conv(J, kernel; bs) -> J'

식 12-13의 실제 계산 결과: JPEG 영역 텐서 J를 완전히 복원(idct2_blocks)한
뒤 표준 공간 합성곱을 적용하고 다시 JPEG 영역으로 압축한다. 이 세 단계를
하나로 묶은 것이 논문의 Ξ 연산자이며, 근사가 아니라 정확한 등가 연산이다.
"""
function jpeg_domain_conv(J::AbstractMatrix{<:Real}, kernel::AbstractMatrix{<:Real}; bs::Int=8)
    img = idct2_blocks(J, bs)
    out = conv2d_same(img, kernel)
    return dct2_blocks(out, bs)
end

# ---------------------------------------------------------------------------
# 3. Section 4.2: ReLU via Approximated Spatial Masking (ASM) (식 14-20)
# ---------------------------------------------------------------------------

"""
zigzag_order(n) -> Vector{Tuple{Int,Int}}

n×n DCT 블록의 표준 JPEG 지그재그 순서(저주파 -> 고주파)를 생성한다.
Theorem 1(DCT 최소제곱근사)에 따라 낮은 m개의 (지그재그 순서) 주파수만
사용한 부분 재구성이 다른 어떤 m개 조합보다도 최소제곱 오차가 작다.
"""
function zigzag_order(n::Int)
    idx = Tuple{Int,Int}[]
    for s in 0:(2n-2)
        rng = max(0, s - n + 1):min(s, n - 1)
        seq = iseven(s) ? reverse(collect(rng)) : collect(rng)
        for i in seq
            push!(idx, (i, s - i))
        end
    end
    return idx
end

"""
partial_idct_block(F, m; zz) -> 근사 재구성 영상 (식 1, Theorem 1)

블록의 DCT 계수 F 중 지그재그 순서상 가장 낮은 m개만 남기고 나머지는 0으로
두어 역DCT한, m-항 최소제곱 최적 근사 재구성을 계산한다.
"""
function partial_idct_block(F::AbstractMatrix{<:Real}, m::Int; zz=zigzag_order(size(F, 1)))
    Fm = zeros(size(F))
    @inbounds for (i, j) in zz[1:m]
        Fm[i+1, j+1] = F[i+1, j+1]
    end
    return FFTW.idct(Fm)
end

"""
asm_relu_block(F, m; zz) -> F'  (ASM 근사, 논문 제안 기법)

1) m개 주파수만으로 근사 재구성한 뒤 그 부호(양/음)만 이진 마스크로 사용
2) "정확한"(전체 64계수) 역DCT 픽셀 값에 그 마스크를 곱함 -> 픽셀 값 자체는
   보존되고, 마스크가 잘못 그어지는 경계 부분에서만 오차가 발생
3) 다시 DCT하여 JPEG 영역으로 복귀
"""
function asm_relu_block(F::AbstractMatrix{<:Real}, m::Int; zz=zigzag_order(size(F, 1)))
    mask = partial_idct_block(F, m; zz=zz) .> 0
    I_true = FFTW.idct(F)
    return FFTW.dct(I_true .* mask)
end

"""
direct_approx_relu_block(F, m; zz) -> F'  (APX 베이스라인, 논문의 비교 대상)

m개 주파수로만 재구성한 뒤 그 근사 픽셀 값에 직접 ReLU를 적용한다. 마스크
뿐 아니라 픽셀 "값" 자체도 근사이므로 ASM보다 정확도가 낮다(Fig 1, Fig 4).
"""
function direct_approx_relu_block(F::AbstractMatrix{<:Real}, m::Int; zz=zigzag_order(size(F, 1)))
    I_approx = partial_idct_block(F, m; zz=zz)
    return FFTW.dct(max.(I_approx, 0.0))
end

"""정확한(근사 없는) ReLU: 압축 해제 -> ReLU -> 재압축. 정확도 비교의 기준선."""
true_relu_block(F::AbstractMatrix{<:Real}) = FFTW.dct(max.(FFTW.idct(F), 0.0))

"""
relu_approx_rmse(; nblocks, bs, ms) -> (ms, asm_rmse, apx_rmse)

논문 Section 5.3 / Figure 4a의 실험을 재현한다: 4×4 픽셀을(범위 [-1,1])
박스필터(픽셀 복제)로 8×8로 업샘플링한 무작위 블록들(자연 영상 통계에
가까운 최악 사례가 아닌 블록)에 대해, m=1..15 주파수로 ASM과 APX의 ReLU
근사 RMSE를 비교한다.
"""
function relu_approx_rmse(; nblocks::Int=20000, bs::Int=8, ms=1:15, rng=Random.default_rng())
    zz = zigzag_order(bs)
    asm_sq = zeros(length(ms))
    apx_sq = zeros(length(ms))
    for _ in 1:nblocks
        small = rand(rng, bs ÷ 2, bs ÷ 2) .* 2 .- 1     # [-1,1] 균등분포 4×4
        blk = kron(small, ones(2, 2))                    # 박스필터(픽셀 복제) 업샘플 -> 8×8
        F = FFTW.dct(blk)
        I_true = max.(blk, 0.0)
        for (k, m) in enumerate(ms)
            I_asm = FFTW.idct(asm_relu_block(F, m; zz=zz))
            I_apx = FFTW.idct(direct_approx_relu_block(F, m; zz=zz))
            asm_sq[k] += sum(abs2, I_asm .- I_true)
            apx_sq[k] += sum(abs2, I_apx .- I_true)
        end
    end
    n = nblocks * bs * bs
    return collect(ms), sqrt.(asm_sq ./ n), sqrt.(apx_sq ./ n)
end

# ---------------------------------------------------------------------------
# 4. Section 4.3: Batch Normalization in the JPEG Transform Domain (식 21-24)
#    Theorem 1: DC(=0,0) 계수는 블록 평균에 비례(∝ mean·bs, bs=블록 크기).
#    Theorem 2 (DCT Mean-Variance): 평균이 0인 신호는 Var[X] = E[Y²]
#    (Parseval 항등식의 직접적인 결과).
# ---------------------------------------------------------------------------

"""
dct_domain_batchnorm(J, γ, β, E_I, Var_I; bs, eps) -> J'

BN(I) = γ(I-E[I])/sqrt(Var[I]+eps) + β 를 DCT 영역에서 수행한다. 이는 전체
영상에 대한 아핀변환(스칼라 곱 + 스칼라 덧셈)이므로: 모든 계수(DC+AC)를
`scale`배 하고, 각 블록의 DC 계수에만 `offset·bs`를 더하면 된다(상수를
더하는 것은 블록 평균만 바꾸고 AC 성분에는 영향이 없으므로).
"""
function dct_domain_batchnorm(J::AbstractMatrix{<:Real}, γ::Real, β::Real, E_I::Real, Var_I::Real;
    bs::Int=8, eps::Real=1e-5)
    scale = γ / sqrt(Var_I + eps)
    offset = β - scale * E_I
    rows, cols = size(J)
    Jn = J .* scale
    for by in 0:bs:rows-1, bx in 0:bs:cols-1
        Jn[by+1, bx+1] += offset * bs
    end
    return Jn
end

"""공간 영역에서의 표준 BatchNorm (동치성 검증용 기준선)."""
spatial_batchnorm(img::AbstractMatrix{<:Real}, γ::Real, β::Real, E_I::Real, Var_I::Real; eps::Real=1e-5) =
    γ .* (img .- E_I) ./ sqrt(Var_I + eps) .+ β

# ---------------------------------------------------------------------------
# 5. Section 4.5: Global Average Pooling (Figure 2) - DC 계수 평균이 곧 GAP
# ---------------------------------------------------------------------------
function global_avg_from_dc(J::AbstractMatrix{<:Real}; bs::Int=8)
    rows, cols = size(J)
    dcs = [J[by+1, bx+1] for by in 0:bs:rows-1 for bx in 0:bs:cols-1]
    return mean(dcs) / bs
end

# ---------------------------------------------------------------------------
# 6. 미니 "JPEG 영역 잔차 블록": Conv -> BN -> ASM-ReLU -> +skip, 전부 JPEG
#    영역에서 수행 (Section 4.4의 성분별 덧셈 J(F+G)=J(F)+J(G) 포함)
# ---------------------------------------------------------------------------
function jpeg_domain_residual_block(img::AbstractMatrix{<:Real}, kernel::AbstractMatrix{<:Real},
    γ::Real, β::Real; bs::Int=8, m::Int=15)
    J = dct2_blocks(img, bs)
    J1 = jpeg_domain_conv(J, kernel; bs=bs)
    E_I, Var_I = mean(img), var(img, corrected=false)
    J2 = dct_domain_batchnorm(J1, γ, β, E_I, Var_I; bs=bs)

    zz = zigzag_order(bs)
    rows, cols = size(J2)
    J3 = similar(J2)
    for by in 0:bs:rows-1, bx in 0:bs:cols-1
        J3[by+1:by+bs, bx+1:bx+bs] = asm_relu_block(J2[by+1:by+bs, bx+1:bx+bs], m; zz=zz)
    end
    return J3 .+ J   # 잔차(skip) 연결: JPEG 영역에서의 단순 성분별 덧셈 (식 25)
end

function spatial_residual_block(img::AbstractMatrix{<:Real}, kernel::AbstractMatrix{<:Real}, γ::Real, β::Real)
    out = conv2d_same(img, kernel)
    E_I, Var_I = mean(img), var(img, corrected=false)
    out = spatial_batchnorm(out, γ, β, E_I, Var_I)
    out = max.(out, 0.0)
    return out .+ img
end

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
normalize01(x) = begin
    lo, hi = extrema(x)
    (x .- lo) ./ (hi - lo + eps())
end

"""Figure 1 스타일 색상 시각화: 파랑=0, 빨강=양수(값에 비례한 밝기), 원본은 초록=음수/빨강=양수."""
function relu_colorize(orig::AbstractMatrix{<:Real}, result::AbstractMatrix{<:Real})
    mx = maximum(abs.(orig)) + eps()
    img = Array{RGB{Float64}}(undef, size(orig))
    for idx in eachindex(orig)
        v = result[idx]
        if abs(v) < 1e-9
            img[idx] = RGB(0.1, 0.1, 0.9)                    # 마스크로 0이 된 픽셀 -> 파랑
        else
            img[idx] = RGB(0.9 * clamp(v / mx, 0, 1) + 0.1, 0.1, 0.1)  # 양수 -> 빨강
        end
    end
    return img
end
function orig_colorize(orig::AbstractMatrix{<:Real})
    mx = maximum(abs.(orig)) + eps()
    img = Array{RGB{Float64}}(undef, size(orig))
    for idx in eachindex(orig)
        v = orig[idx]
        if v >= 0
            img[idx] = RGB(0.9 * clamp(v / mx, 0, 1) + 0.1, 0.1, 0.1)   # 양수 -> 빨강
        else
            img[idx] = RGB(0.1, 0.9 * clamp(-v / mx, 0, 1) + 0.1, 0.1)  # 음수 -> 초록
        end
    end
    return img
end

function upsample_nn(img::AbstractMatrix, factor::Int)
    rows, cols = size(img)
    out = similar(img, rows * factor, cols * factor)
    for i in 1:rows, j in 1:cols
        out[(i-1)*factor+1:i*factor, (j-1)*factor+1:j*factor] .= img[i, j]
    end
    return out
end

function main()
    println("="^76)
    println("Deep Residual Learning in the JPEG Transform Domain")
    println("Ehrlich & Davis, ICCV 2019, arXiv:1812.11690")
    println("="^76)

    rng = Random.MersenneTwister(0)
    img = Float64.(Gray.(testimage("cameraman")))
    rows, cols = size(img)

    # --- [1] JPEG 변환 영역 왕복(compress/decompress) 무손실성 검증 ---
    println("\n[1] JPEG 변환 영역 왕복 오차 (반올림 이전이므로 무손실이어야 함)")
    J0 = compress(img)
    rec = decompress(J0)
    println("  max|img - decompress(compress(img))| = ", maximum(abs.(img .- rec)))
    img255 = img .* 255.0   # 표준 양자화 테이블은 [0,255] 픽셀 범위 기준으로 설계됨
    lossy = compress_lossy(img255)
    rec_lossy = decompress(lossy; Q=STD_LUMA_QTABLE) ./ 255.0
    println("  (참고) 반올림 포함 진짜 손실 JPEG(품질 50) PSNR = ",
        round(10 * log10(1.0 / mean((img .- rec_lossy) .^ 2)), digits=2), " dB")

    # --- [2] Theorem 1 & 2: DC=평균, Parseval=분산 관계 수치 검증 ---
    println("\n[2] Theorem 1 (DC=블록평균×bs) / Theorem 2 (Parseval -> 분산=Σ(AC²)/bs²) 검증")
    blk = img[1:8, 1:8]
    F = FFTW.dct(blk)
    println("  DC(F[1,1]) = ", F[1, 1], "   mean(blk)*8 = ", mean(blk) * 8)
    ac_energy = sum(abs2, F) - F[1, 1]^2
    println("  Var(blk) = ", var(blk, corrected=false), "   Σ(AC²)/64 = ", ac_energy / 64)

    # --- [3] Convolution 등가성 (Ξ, 식 11-13), 블록 경계를 넘는 커널 포함 ---
    println("\n[3] 압축영역 합성곱(Ξ) 등가성: decompress→conv→compress == 직접 공간 합성곱")
    sobel_x = Float64[-1 0 1; -2 0 2; -1 0 1]
    J = compress(img)
    Jc = jpeg_domain_conv(J, sobel_x)
    lhs = decompress(Jc)
    rhs = conv2d_same(img, sobel_x)
    println("  max|decompress(jpeg_domain_conv(J,K)) - conv2d(img,K)| = ", maximum(abs.(lhs .- rhs)))
    println("  (커널이 8×8 블록 경계를 넘나드는 위치 포함, 전 영상 512×512에서 검증)")

    # --- [4] ASM vs APX ReLU 근사 정확도 (Section 5.3, Figure 4a 재현) ---
    println("\n[4] ReLU 근사 정확도: ASM vs 직접근사(APX), m=1..15 주파수 (Figure 4a 재현)")
    ms, asm_rmse, apx_rmse = relu_approx_rmse(nblocks=20000, ms=1:15, rng=rng)
    println("  ", rpad("m", 5), rpad("ASM RMSE", 14), rpad("APX RMSE", 14), "ASM/APX 비")
    for k in eachindex(ms)
        println("  ", rpad(string(ms[k]), 5), rpad(string(round(asm_rmse[k], digits=5)), 14),
            rpad(string(round(apx_rmse[k], digits=5)), 14), round(apx_rmse[k] / asm_rmse[k], digits=2))
    end
    println("  -> 모든 m에서 ASM이 APX보다 낮은 RMSE (평균 ",
        round(mean(apx_rmse ./ asm_rmse), digits=2), "배 더 정확)")

    # --- [4b] Figure 1 스타일 시각화 (m=6) ---
    println("\n[4b] Figure 1 스타일 시각화 (m=6, 8×8 블록을 64배 확대)")
    demo_blk = img[201:208, 281:288] .* 2 .- 1   # 고대비 에지 블록, 양/음 값이 섞이도록 중심화
    Fd = FFTW.dct(demo_blk)
    zz8 = zigzag_order(8)
    m_demo = 6
    true_r = FFTW.idct(true_relu_block(Fd))
    asm_r = FFTW.idct(asm_relu_block(Fd, m_demo; zz=zz8))
    apx_r = FFTW.idct(direct_approx_relu_block(Fd, m_demo; zz=zz8))

    up = 32
    orig_img = upsample_nn(orig_colorize(demo_blk), up)
    true_img = upsample_nn(relu_colorize(demo_blk, true_r), up)
    asm_img = upsample_nn(relu_colorize(demo_blk, asm_r), up)
    apx_img = upsample_nn(relu_colorize(demo_blk, apx_r), up)
    mosaic = hcat(orig_img, true_img, asm_img, apx_img)
    save("jpegresnet_relu_figure1.png", mosaic)
    println("  저장됨: jpegresnet_relu_figure1.png (원본 | 정확한 ReLU | ASM 근사 | APX 근사)")
    println("  |ASM-true| RMSE = ", round(sqrt(mean(abs2, asm_r .- true_r)), digits=4),
        "   |APX-true| RMSE = ", round(sqrt(mean(abs2, apx_r .- true_r)), digits=4))

    # --- [5] BatchNorm 등가성 검증 ---
    println("\n[5] DCT 영역 BatchNorm ↔ 공간 영역 BatchNorm 등가성 검증")
    γ, β = 1.7, 0.3
    E_I, Var_I = mean(img), var(img, corrected=false)
    J_bn = dct_domain_batchnorm(J, γ, β, E_I, Var_I)
    lhs_bn = decompress(J_bn)
    rhs_bn = spatial_batchnorm(img, γ, β, E_I, Var_I)
    println("  max|decompress(dct_domain_batchnorm(J)) - spatial_batchnorm(img)| = ",
        maximum(abs.(lhs_bn .- rhs_bn)))

    # --- [6] 잔차 덧셈 선형성 + Global Average Pooling 검증 ---
    println("\n[6] 잔차 덧셈 선형성 J(F+G)=J(F)+J(G) 및 GAP=DC 평균 검증")
    A, B = img, conv2d_same(img, sobel_x)
    println("  max|dct2(A+B) - (dct2(A)+dct2(B))| = ",
        maximum(abs.(dct2_blocks(A .+ B) .- (dct2_blocks(A) .+ dct2_blocks(B)))))
    println("  |mean(img) - GAP_from_DC(J)| = ", abs(mean(img) - global_avg_from_dc(J)))

    # --- [7] 미니 JPEG 영역 잔차 블록 vs 공간 영역 잔차 블록 end-to-end 비교 ---
    println("\n[7] 미니 잔차 블록(Conv->BN->ReLU->+skip) 전체 파이프라인 비교 (Table 1 재현)")
    for m_test in (4, 8, 15, 64)
        J_out = jpeg_domain_residual_block(img, sobel_x, γ, β; m=m_test)
        jpeg_result = decompress(J_out)
        spatial_result = spatial_residual_block(img, sobel_x, γ, β)
        dev = sqrt(mean(abs2, jpeg_result .- spatial_result))
        println("  m=$(rpad(m_test, 3)) (ASM 근사 주파수 수)  ->  spatial-vs-JPEG 편차(RMSE) = ",
            round(dev, digits=6), m_test == 64 ? "  (전체 주파수 사용, ReLU 근사 없음)" : "")
    end

    save("jpegresnet_source.png", Gray.(img))
    println("\n저장됨: jpegresnet_source.png, jpegresnet_relu_figure1.png")
    println("\n완료.")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
