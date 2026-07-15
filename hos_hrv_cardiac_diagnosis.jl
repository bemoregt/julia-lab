using FFTW
using Statistics
using Random
using Images

# =============================================================
# 1. 합성 ECG 신호 생성 (Gaussian-sum P-QRS-T 모델)
# =============================================================

"""
ecg_beat_template(t, rpeak) -> Vector{Float64}

R-피크 시각 `rpeak`을 기준으로 P, Q, R, S, T 파형을 각각 가우시안 펄스의
합으로 근사한 단일 박동 파형을 생성한다. (진폭, R 기준 오프셋(초), 폭)은
일반적인 정상 ECG 파형의 대략적인 형태를 따른다.
"""
function ecg_beat_template(t::AbstractVector{Float64}, rpeak::Float64)
    waves = ((0.15, -0.20, 0.025),   # P
             (-0.10, -0.05, 0.008),  # Q
             (1.00, 0.00, 0.010),    # R
             (-0.25, 0.04, 0.010),   # S
             (0.30, 0.30, 0.040))    # T
    y = zeros(length(t))
    for (amp, off, wid) in waves
        y .+= amp .* exp.(-((t .- (rpeak + off)) .^ 2) ./ (2 * wid^2))
    end
    return y
end

"""
generate_ecg(rr_intervals, fs) -> (t, ecg, rpeak_times)

R-R 간격 시퀀스(초 단위)로부터 각 박동의 R-피크 위치를 계산하고, 그 위에
`ecg_beat_template`을 중첩해 전체 ECG 신호를 합성한다.
"""
function generate_ecg(rr_intervals::Vector{Float64}, fs::Float64)
    rpeak_times = cumsum(rr_intervals)
    total_time = rpeak_times[end] + 0.5
    n = round(Int, total_time * fs)
    t = collect((0:n-1) ./ fs)
    ecg = zeros(n)
    for rp in rpeak_times
        ecg .+= ecg_beat_template(t, rp)
    end
    ecg .+= 0.01 .* randn(Xoshiro(1), n)   # 약한 관측 잡음
    return t, ecg, rpeak_times
end

# =============================================================
# 2. HRV(R-R 간격) 시나리오 생성 - 정상 vs 부정맥 유사 패턴
# =============================================================

"""
make_rr_normal(n_beats; rng) -> Vector{Float64}

정상 동리듬(NSR)의 R-R 간격을 모사한다. 호흡성 동성부정맥(RSA, ~0.25 Hz)과
저주파 자율신경 조절(LF, ~0.1 Hz) 변동을 사인파로 더하고 약한 잡음을 추가한다.
"""
function make_rr_normal(n_beats::Int; mean_rr::Float64=0.8, rng::AbstractRNG=Random.default_rng())
    beat_t = (0:n_beats-1) .* mean_rr
    rr = mean_rr .+
         0.03 .* sin.(2π .* 0.25 .* beat_t) .+
         0.02 .* sin.(2π .* 0.10 .* beat_t .+ 0.7) .+
         0.005 .* randn(rng, n_beats)
    return rr
end

"""
make_rr_arrhythmic(n_beats; rng) -> Vector{Float64}

부정맥 유사 R-R 간격을 모사한다. 정상 변동에 더해, 조기수축(짧은 R-R) +
보상성 휴지기(긴 R-R) 쌍을 무작위 위치에 반복적으로 삽입해 HRV 신호에
강한 비선형적 불규칙성을 만든다.
"""
function make_rr_arrhythmic(n_beats::Int; mean_rr::Float64=0.8, rng::AbstractRNG=Random.default_rng())
    rr = mean_rr .+
         0.02 .* sin.(2π .* 0.25 .* (0:n_beats-1) .* mean_rr) .+
         0.01 .* sin.(2π .* 0.10 .* (0:n_beats-1) .* mean_rr) .+
         0.01 .* randn(rng, n_beats)
    n_events = round(Int, n_beats * 0.08)
    for _ in 1:n_events
        idx = rand(rng, 3:n_beats-3)
        rr[idx] *= 0.55
        rr[idx+1] *= 1.45
    end
    return rr
end

# =============================================================
# 3. HRV 타코그램 -> 균일 샘플링된 시계열로 재보간
# =============================================================

function linear_interp(xs::Vector{Float64}, ys::Vector{Float64}, xq::Vector{Float64})
    yq = similar(xq)
    for (i, x) in enumerate(xq)
        if x <= xs[1]
            yq[i] = ys[1]
        elseif x >= xs[end]
            yq[i] = ys[end]
        else
            j = searchsortedfirst(xs, x)
            x0, x1 = xs[j-1], xs[j]
            y0, y1 = ys[j-1], ys[j]
            yq[i] = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        end
    end
    return yq
end

"""
resample_hrv(rr; fs_interp) -> (t_uniform, hr_uniform)

불균일한 R-R 간격 타코그램을, 스펙트럼 분석이 가능하도록 균일한 표본율
(`fs_interp`, 기본 4 Hz)의 순간 심박수(bpm) 시계열로 선형 보간한다. HRV
스펙트럼/고차 스펙트럼 분석의 표준적인 전처리 절차이다.
"""
function resample_hrv(rr::Vector{Float64}; fs_interp::Float64=4.0)
    rpeak_times = cumsum(rr)
    hr = 60.0 ./ rr
    t_uniform = collect(0:(1 / fs_interp):rpeak_times[end])
    hr_uniform = linear_interp(rpeak_times, hr, t_uniform)
    return t_uniform, hr_uniform
end

# =============================================================
# 4. 비스펙트럼(Bispectrum) 추정 (직접법, 구간 평균)
# =============================================================

hann(n::Int) = [0.5 - 0.5 * cos(2π * (i - 1) / (n - 1)) for i in 1:n]

"""
bispectrum(x; nseg, overlap) -> Matrix{ComplexF64}

신호 x를 길이 `nseg`의 중첩 구간으로 나누어(직접법), 각 구간의 FFT로부터
    B(k1,k2) = E[ X(k1) X(k2) X*(k1+k2) ]
를 구간 평균으로 추정한다. 3차 큐뮬런트의 푸리에 변환으로, 주파수 성분
간 비선형 위상 결합(quadratic phase coupling)을 포착한다.
"""
function bispectrum(x::Vector{Float64}; nseg::Int=64, overlap::Float64=0.5)
    N = length(x)
    step = max(1, round(Int, nseg * (1 - overlap)))
    w = hann(nseg)
    B = zeros(ComplexF64, nseg, nseg)
    count = 0
    s = 1
    while s + nseg - 1 <= N
        seg = (x[s:s+nseg-1] .- mean(x[s:s+nseg-1])) .* w
        X = fft(seg)
        for k1 in 1:nseg, k2 in 1:nseg
            k3 = mod(k1 - 1 + k2 - 1, nseg) + 1
            B[k1, k2] += X[k1] * X[k2] * conj(X[k3])
        end
        count += 1
        s += step
    end
    count == 0 && error("신호가 너무 짧아 비스펙트럼을 계산할 구간이 없습니다")
    return B ./ count
end

"""
principal_domain(nseg) -> Vector{Tuple{Int,Int}}

실수 신호 비스펙트럼의 대칭성으로 인해 중복되지 않는 주 영역(principal
domain): 0 <= k2 <= k1, k1+k2 <= nseg/2 (0-based 주파수 인덱스) 만을
1-based 인덱스 쌍으로 반환한다.
"""
function principal_domain(nseg::Int)
    idx = Tuple{Int,Int}[]
    for k1 in 0:(nseg ÷ 2)
        for k2 in 0:k1
            if k1 + k2 <= nseg ÷ 2
                push!(idx, (k1 + 1, k2 + 1))
            end
        end
    end
    return idx
end

# =============================================================
# 5. 고차 스펙트럼(HOS) 기반 특징 7종
#    (Chua/Acharya 계열 HOS 논문들에서 공통적으로 쓰이는 표준적인
#     비스펙트럼 엔트로피/로그진폭 모멘트/가중중심 특징으로 재구성)
# =============================================================

"""
hos_features(B, dom, fs, nseg) -> NamedTuple

비스펙트럼 B의 주 영역(dom)으로부터 7개의 고차 스펙트럼 특징을 계산한다.

- P1: 정규화 비스펙트럼 엔트로피 (진폭 분포의 샤논 엔트로피)
- P2: 정규화 비스펙트럼 제곱 엔트로피 (에너지 분포의 샤논 엔트로피)
- H1: 로그 진폭의 합
- H2: 로그 진폭의 평균
- H3: 로그 진폭의 표준편차 (분포의 퍼짐 정도)
- f1m, f2m: 진폭으로 가중한 (f1, f2) 축의 무게중심 주파수(Hz)
"""
function hos_features(B::Matrix{ComplexF64}, dom::Vector{Tuple{Int,Int}}, fs::Float64, nseg::Int)
    eps_ = 1e-12
    mags = [abs(B[k1, k2]) for (k1, k2) in dom]
    total = sum(mags) + eps_

    p = mags ./ total
    P1 = -sum(p .* log.(p .+ eps_)) / log(length(mags))

    mags2 = mags .^ 2
    total2 = sum(mags2) + eps_
    q = mags2 ./ total2
    P2 = -sum(q .* log.(q .+ eps_)) / log(length(mags))

    logmags = log.(mags .+ eps_)
    H1 = sum(logmags)
    H2 = mean(logmags)
    H3 = std(logmags)

    freqs = [(k - 1) * fs / nseg for k in 1:nseg]
    f1s = [freqs[k1] for (k1, k2) in dom]
    f2s = [freqs[k2] for (k1, k2) in dom]
    f1m = sum(f1s .* mags) / total
    f2m = sum(f2s .* mags) / total

    return (P1=P1, P2=P2, H1=H1, H2=H2, H3=H3, f1m=f1m, f2m=f2m)
end

feature_names(f) = [:P1, :P2, :H1, :H2, :H3, :f1m, :f2m]
feature_vec(f) = [f.P1, f.P2, f.H1, f.H2, f.H3, f.f1m, f.f2m]

# =============================================================
# 6. 두 그룹 비교를 위한 간단한 Welch t-검정 (외부 의존성 없이 직접 구현)
# =============================================================

# Abramowitz-Stegun 근사 (표준정규분포 CDF)
function norm_cdf(z::Float64)
    return 0.5 * (1 + erf_approx(z / sqrt(2)))
end
function erf_approx(x::Float64)
    a1,a2,a3,a4,a5 = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
    p = 0.3275911
    s = sign(x); x = abs(x)
    t = 1/(1+p*x)
    y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*exp(-x^2)
    return s*y
end

function welch_t_pvalue(a::Vector{Float64}, b::Vector{Float64})
    ma, mb = mean(a), mean(b)
    va, vb = var(a), var(b)
    na, nb = length(a), length(b)
    se = sqrt(va/na + vb/nb)
    t = (ma - mb) / (se + 1e-12)
    # 자유도가 충분히 크면(>=20) 정규분포로 근사 (표시 목적의 근사치)
    p = 2 * (1 - norm_cdf(abs(t)))
    return t, p
end

# =============================================================
# 7. 간단한 라인 플롯 렌더링 (외부 플로팅 패키지 없이 직접 구현)
# =============================================================

function draw_line_seg!(canvas::Matrix{Float64}, x0::Int, y0::Int, x1::Int, y1::Int)
    h, w = size(canvas)
    dx = abs(x1-x0); sx = x0<x1 ? 1 : -1
    dy = -abs(y1-y0); sy = y0<y1 ? 1 : -1
    err = dx+dy
    x,y = x0,y0
    while true
        if 1<=y<=h && 1<=x<=w
            canvas[y,x] = 1.0
        end
        (x==x1 && y==y1) && break
        e2 = 2*err
        if e2>=dy; err+=dy; x+=sx; end
        if e2<=dx; err+=dx; y+=sy; end
    end
end

function line_plot(xs::Vector{Float64}, ys::Vector{Float64}, width::Int, height::Int)
    canvas = zeros(Float64, height, width)
    xlo, xhi = minimum(xs), maximum(xs)
    ylo, yhi = minimum(ys), maximum(ys)
    yrange = (yhi - ylo) + eps()
    px(x) = 1 + round(Int, (x - xlo) / (xhi - xlo + eps()) * (width - 1))
    py(y) = height - round(Int, (y - ylo) / yrange * (height - 1))
    for i in 1:length(xs)-1
        draw_line_seg!(canvas, px(xs[i]), py(ys[i]), px(xs[i+1]), py(ys[i+1]))
    end
    return canvas
end

function heatmap_image(M::Matrix{Float64})
    lo, hi = minimum(M), maximum(M)
    return (M .- lo) ./ (hi - lo + eps())
end

"""
jet_colormap(v) -> RGB{N0f8}

0~1 정규화된 값을 파랑(낮음)-청록-초록-노랑-빨강(높음)으로 이어지는
"jet" 스타일 슈도컬러로 매핑한다.
"""
function jet_colormap(v::Float64)
    v = clamp(v, 0.0, 1.0)
    r = clamp(1.5 - abs(4v - 3), 0.0, 1.0)
    g = clamp(1.5 - abs(4v - 2), 0.0, 1.0)
    b = clamp(1.5 - abs(4v - 1), 0.0, 1.0)
    return RGB{N0f8}(r, g, b)
end

"""
pseudocolor_image(M) -> Matrix{RGB{N0f8}}

행렬을 0~1로 정규화한 뒤 jet 슈도컬러 이미지로 변환한다.
"""
function pseudocolor_image(M::Matrix{Float64})
    return jet_colormap.(heatmap_image(M))
end

# =============================================================
# 8. 파이프라인 실행: 정상 vs 부정맥 유사 HRV 비교
# =============================================================

function analyze_one(rr::Vector{Float64}; fs_interp::Float64=4.0, nseg::Int=64)
    t_u, hr_u = resample_hrv(rr; fs_interp=fs_interp)
    B = bispectrum(hr_u; nseg=nseg, overlap=0.5)
    dom = principal_domain(nseg)
    feats = hos_features(B, dom, fs_interp, nseg)
    return feats, B, t_u, hr_u
end

function main()
    fs_ecg = 250.0
    n_beats = 300
    n_trials = 15

    rng_n = Xoshiro(10)
    rng_a = Xoshiro(20)

    # --- 대표 샘플 1개씩으로 파형/타코그램/비스펙트럼 시각화 ---
    rr_normal = make_rr_normal(n_beats; rng=rng_n)
    rr_arrhy = make_rr_arrhythmic(n_beats; rng=rng_a)

    t_n, ecg_n, rp_n = generate_ecg(rr_normal, fs_ecg)
    t_a, ecg_a, rp_a = generate_ecg(rr_arrhy, fs_ecg)

    feats_n, B_n, tu_n, hru_n = analyze_one(rr_normal)
    feats_a, B_a, tu_a, hru_a = analyze_one(rr_arrhy)

    # ECG 파형 (앞 8초 구간)
    mask_n = t_n .<= 8.0
    mask_a = t_a .<= 8.0
    save("hos_ecg_normal.png", Gray.(clamp01.(line_plot(t_n[mask_n], ecg_n[mask_n], 900, 220))))
    save("hos_ecg_arrhythmic.png", Gray.(clamp01.(line_plot(t_a[mask_a], ecg_a[mask_a], 900, 220))))

    # HRV 타코그램 (심박수, bpm)
    save("hos_hrv_normal.png", Gray.(clamp01.(line_plot(tu_n, hru_n, 900, 220))))
    save("hos_hrv_arrhythmic.png", Gray.(clamp01.(line_plot(tu_a, hru_a, 900, 220))))

    # 비스펙트럼 크기 히트맵: fftshift로 저주파(DC) 성분이 이미지 중앙에 오도록 하고,
    # jet 슈도컬러로 표현 (시각화를 위해 확대)
    Bn_centered = abs.(fftshift(B_n))
    Ba_centered = abs.(fftshift(B_a))
    save("hos_bispectrum_normal.png", imresize(pseudocolor_image(Bn_centered), ratio=8))
    save("hos_bispectrum_arrhythmic.png", imresize(pseudocolor_image(Ba_centered), ratio=8))

    println("=== 대표 샘플 1개 특징 비교 ===")
    println(rpad("특징", 6), rpad("Normal", 14), rpad("Arrhythmic", 14))
    for name in feature_names(feats_n)
        vn = getfield(feats_n, name)
        va = getfield(feats_a, name)
        println(rpad(string(name), 6), rpad(round(vn, digits=4), 14), rpad(round(va, digits=4), 14))
    end

    # --- 여러 시행(trial)으로 그룹 통계 비교 (논문의 ANOVA/p-value 검증에 대응) ---
    println()
    println("=== $(n_trials)회 반복 시행 그룹 비교 (평균±표준편차, Welch t-검정 근사 p-value) ===")
    names = feature_names(feats_n)
    normal_feats = zeros(n_trials, length(names))
    arrhy_feats = zeros(n_trials, length(names))

    for trial in 1:n_trials
        rrn = make_rr_normal(n_beats; rng=Xoshiro(1000 + trial))
        rra = make_rr_arrhythmic(n_beats; rng=Xoshiro(2000 + trial))
        fn, _, _, _ = analyze_one(rrn)
        fa, _, _, _ = analyze_one(rra)
        normal_feats[trial, :] .= feature_vec(fn)
        arrhy_feats[trial, :] .= feature_vec(fa)
    end

    println(rpad("특징", 6), rpad("Normal(mean±std)", 22), rpad("Arrhythmic(mean±std)", 22), rpad("t", 8), "p(approx)")
    for (i, name) in enumerate(names)
        a = normal_feats[:, i]; b = arrhy_feats[:, i]
        t, p = welch_t_pvalue(a, b)
        println(rpad(string(name), 6),
                rpad("$(round(mean(a),digits=3))±$(round(std(a),digits=3))", 22),
                rpad("$(round(mean(b),digits=3))±$(round(std(b),digits=3))", 22),
                rpad(round(t, digits=2), 8),
                round(p, digits=5))
    end

    println()
    println("완료: hos_ecg_{normal,arrhythmic}.png, hos_hrv_{normal,arrhythmic}.png,")
    println("       hos_bispectrum_{normal,arrhythmic}.png 저장됨")
end

main()
