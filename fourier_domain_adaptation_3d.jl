using Images
using TestImages
using FFTW
using ImageFiltering
using Random
using Statistics: mean

"""
fda_3d(vol_src, vol_tgt, beta) -> Array{Float64,3}

3D Fourier Domain Adaptation (Sahu et al., "3D Fourier Domain Adaptation for
Improving CBCT Tooth Segmentation Under Scanner Parameter Shift", ISBI 2023).
2D FDA(Yang & Soatto, CVPR 2020)의 3차원 볼륨 확장.

    x^{s->t} = IFFT( [ Mβ∘F^A(x^t) + (1-Mβ)∘F^A(x^s),  F^P(x^s) ] )

소스 볼륨의 위상(F^P, 해부학적 구조 정보)은 그대로 유지한 채, 3D 주파수
스펙트럼 중심(저주파, 스캐너 프로토콜에 따른 잡음/스무딩 특성을 담고 있는
영역)의 진폭(F^A)만 타겟 볼륨의 것으로 β 비율만큼 치환한다. 논문에서는
이 방식으로 한 스캐너 프로토콜(소스)의 라벨링된 학습 데이터를 다른
프로토콜(타겟)의 외형에 맞게 변환(mutate)하여 분할 모델의 도메인 이동
문제를 완화한다.
"""
function fda_3d(vol_src::Array{Float64,3}, vol_tgt::Array{Float64,3}, beta::Float64)
    @assert size(vol_src) == size(vol_tgt) "두 볼륨의 크기가 같아야 합니다"
    D, H, W = size(vol_src)

    Fsrc = fftshift(fft(vol_src))
    Ftgt = fftshift(fft(vol_tgt))

    amp_src = abs.(Fsrc)
    pha_src = angle.(Fsrc)
    amp_tgt = abs.(Ftgt)

    cz, cy, cx = D ÷ 2 + 1, H ÷ 2 + 1, W ÷ 2 + 1
    bz = max(1, round(Int, beta * D / 2))
    by = max(1, round(Int, beta * H / 2))
    bx = max(1, round(Int, beta * W / 2))

    z1, z2 = max(1, cz - bz), min(D, cz + bz)
    y1, y2 = max(1, cy - by), min(H, cy + by)
    x1, x2 = max(1, cx - bx), min(W, cx + bx)

    amp_new = copy(amp_src)
    amp_new[z1:z2, y1:y2, x1:x2] .= amp_tgt[z1:z2, y1:y2, x1:x2]

    Fnew = amp_new .* exp.(im .* pha_src)
    return real.(ifft(ifftshift(Fnew)))
end

function draw_histogram(vals::AbstractVector{<:Real}, width::Int, height::Int; nbins::Int=48)
    counts = zeros(Int, nbins)
    for v in vals
        b = clamp(floor(Int, v * nbins) + 1, 1, nbins)
        counts[b] += 1
    end
    maxc = max(1, maximum(counts))
    canvas = fill(Gray{N0f8}(0.0), height, width)
    bw = max(1, width ÷ nbins)
    for b in 1:nbins
        barheight = round(Int, (counts[b] / maxc) * (height - 1))
        barheight == 0 && continue
        col_range = ((b - 1) * bw + 1):min(width, b * bw)
        row_range = (height - barheight + 1):height
        canvas[row_range, col_range] .= Gray{N0f8}(1.0)
    end
    return canvas
end

function mid_slice(vol::Array{Float64,3})
    return vol[:, :, size(vol, 3) ÷ 2 + 1]
end

function main()
    base = Float64.(Gray.(testimage("mri-stack")))   # 실제 3D MRI 볼륨, (226,186,27)
    println("볼륨 크기: ", size(base))

    rng = MersenneTwister(42)

    # 타겟 도메인: 필터링 없는 "날카로운" 원본 볼륨 + 약간의 스캐너 잡음
    target = clamp.(base .+ 0.02 .* randn(rng, size(base)...), 0.0, 1.0)

    # 소스 도메인: 디노이징/스무딩 필터가 적용된 프로토콜을 시뮬레이션
    # (논문의 Planmeca AINO 스무딩 필터 + 다른 값 범위 정규화 상황과 유사)
    smoothed = imfilter(base, Kernel.gaussian((1.5, 1.5, 1.5)))
    source = clamp.(smoothed .* 0.75 .+ 0.10, 0.0, 1.0)

    betas = [0.05, 0.1, 0.2]
    slice_panels = Matrix{Gray{N0f8}}[]
    hist_panels = Matrix{Gray{N0f8}}[]

    push!(slice_panels, Gray{N0f8}.(mid_slice(source)))
    push!(hist_panels, draw_histogram(vec(source), 200, 120))

    adapted_vols = Dict{Float64,Array{Float64,3}}()
    for beta in betas
        adapted = fda_3d(source, target, beta)
        adapted = clamp.(adapted, 0.0, 1.0)
        adapted_vols[beta] = adapted
        push!(slice_panels, Gray{N0f8}.(mid_slice(adapted)))
        push!(hist_panels, draw_histogram(vec(adapted), 200, 120))
    end

    push!(slice_panels, Gray{N0f8}.(mid_slice(target)))
    push!(hist_panels, draw_histogram(vec(target), 200, 120))

    save("fda3d_source.png", Gray{N0f8}.(mid_slice(source)))
    save("fda3d_target.png", Gray{N0f8}.(mid_slice(target)))
    for beta in betas
        save("fda3d_adapted_beta_$(beta).png", Gray{N0f8}.(mid_slice(adapted_vols[beta])))
    end

    save("fda3d_slice_comparison.png", hcat(slice_panels...))
    save("fda3d_histogram_comparison.png", hcat(hist_panels...))

    println("완료: fda3d_source.png, fda3d_target.png, fda3d_adapted_beta_*.png,")
    println("       fda3d_slice_comparison.png, fda3d_histogram_comparison.png 저장됨")
    println("패널(왼쪽부터): source, beta=0.05, beta=0.1, beta=0.2, target")
    println()
    println("평균 밝기 - source: ", round(mean(source), digits=4),
            "  target: ", round(mean(target), digits=4))
    for beta in betas
        println("평균 밝기 - adapted(beta=$beta): ", round(mean(adapted_vols[beta]), digits=4))
    end
end

main()
