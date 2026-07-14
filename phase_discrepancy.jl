using Images
using TestImages
using FFTW
using Statistics: mean

"""
phase_only_image(img) -> Matrix{Float64}

영상의 진폭 스펙트럼을 모두 1로 대체하고 위상 스펙트럼만 남겨 역FFT로
복원한 "위상전용(phase-only) 영상"을 반환한다. Oppenheim-Lim(1981)의
고전적 결과에 따르면 자연 영상의 구조(에지·모양) 정보는 대부분 위상에
담겨 있으므로, 이 재구성 영상은 밝기·대비(진폭) 변화와 무관하게 원 영상의
구조만을 드러낸다.
"""
function phase_only_image(img::AbstractMatrix{<:Real})
    F = fft(Float64.(img))
    phase = angle.(F)
    P = ifft(exp.(im .* phase))
    return real.(P)
end

"""
phase_discrepancy(img1, img2) -> (pd, P1, P2)

두 영상을 각각 주파수 영역에서 위상전용 영상으로 재구성한 뒤, 그 결과를
공간(spatial) 영역에서 뺀 위상불일치(Phase Discrepancy) 차영상을 구한다.

    P1 = IFFT(exp(i*angle(FFT(img1))))
    P2 = IFFT(exp(i*angle(FFT(img2))))
    PD = P1 - P2

각 영상의 진폭(밝기·대비) 스펙트럼은 재구성 전에 이미 제거되므로, img1과
img2 사이에 전역적인 밝기/대비 차이가 있어도 PD는 거의 영향받지 않고
위상이 실제로 달라지는 구조적 변화(결함, 이물, 형태 변화) 영역에서만
큰 값을 갖는다. 이는 공간 영역에서 바로 픽셀 차(|img1-img2|)를 구하는
방식과 대비된다.
"""
function phase_discrepancy(img1::AbstractMatrix{<:Real}, img2::AbstractMatrix{<:Real})
    @assert size(img1) == size(img2) "두 영상의 크기가 같아야 합니다"
    P1 = phase_only_image(img1)
    P2 = phase_only_image(img2)
    return P1 .- P2, P1, P2
end

normalize01(x) = begin
    lo, hi = extrema(x)
    (x .- lo) ./ (hi - lo + eps())
end

function main()
    img = testimage("peppers_gray")
    ref = Float64.(Gray.(img))          # 기준(원본) 영상
    rows, cols = size(ref)

    # 시험 영상 생성: (1) 전역 밝기/대비 변화(포화 없이 순수 선형 변환) + (2) 국소 결함(이물) 패치 삽입
    test = ref .* 1.12 .+ 0.02   # 조명/대비 변화 시뮬레이션 (0~1 범위를 벗어나지 않도록 포화 방지)

    y0, y1 = rows ÷ 2 - 25, rows ÷ 2 + 25
    x0, x1 = cols ÷ 2 - 25, cols ÷ 2 + 25
    test[y0:y1, x0:x1] .= 0.9            # 국소 결함(밝은 이물) 삽입

    # (a) 공간 영역 단순 차영상 - 전역 밝기 변화에 취약
    naive_diff = abs.(ref .- test)

    # (b) 주파수 영역 위상불일치(PD) 차영상 - 밝기 변화에 강건, 구조 변화만 검출
    pd, P1, P2 = phase_discrepancy(ref, test)
    pd_abs = abs.(pd)

    save("pd_reference.png", Gray.(ref))
    save("pd_test.png", Gray.(clamp01.(test)))
    save("pd_naive_diff.png", Gray.(clamp01.(normalize01(naive_diff))))
    save("pd_phase_discrepancy.png", Gray.(clamp01.(normalize01(pd_abs))))

    println("완료: pd_reference.png, pd_test.png, pd_naive_diff.png, pd_phase_discrepancy.png 저장됨")
    println("입력 크기: ", size(ref))
    println("결함 위치(행,열): ", (y0:y1, x0:x1))
    println()
    println("[단순 공간 차영상]  평균=", round(mean(naive_diff), digits=4),
            "  결함영역 평균=", round(mean(naive_diff[y0:y1, x0:x1]), digits=4))
    println("[위상불일치(PD) 차영상] 평균=", round(mean(pd_abs), digits=4),
            "  결함영역 평균=", round(mean(pd_abs[y0:y1, x0:x1]), digits=4))
end

main()
