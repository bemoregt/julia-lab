using Images
using TestImages
using FFTW

"""
homomorphic_filter(img; d0, c, gamma_l, gamma_h)

Grayscale float image (values in [0,1]) -> homomorphic-filtered grayscale float image.

- 조명(illumination, 저주파) 성분은 억제하고 반사율(reflectance, 고주파) 성분은 강조하여
  그림자/불균일 조명을 보정하고 디테일(에지)을 살리는 필터.
- d0: 컷오프 거리(주파수 반지름), c: 전이 기울기(steepness),
  gamma_l < 1: 저주파(조명) 감쇠 정도, gamma_h > 1: 고주파(반사율) 증폭 정도.
"""
function homomorphic_filter(img::AbstractMatrix{<:Real};
                             d0::Float64=30.0,
                             c::Float64=1.0,
                             gamma_l::Float64=0.4,
                             gamma_h::Float64=2.5)
    rows, cols = size(img)

    # 1) 로그 변환: 조명(곱셈)과 반사율(곱셈) 성분을 덧셈으로 분리
    log_img = log.(Float64.(img) .+ 1.0)

    # 2) 주파수 영역으로 변환
    F = fftshift(fft(log_img))

    # 3) 고주파 강조(high-emphasis) 필터 H(u,v) 생성
    #    H = (gamma_h - gamma_l) * (1 - exp(-c * D^2 / D0^2)) + gamma_l
    H = Matrix{Float64}(undef, rows, cols)
    cy, cx = (rows ÷ 2) + 1, (cols ÷ 2) + 1
    for u in 1:rows, v in 1:cols
        d2 = (u - cy)^2 + (v - cx)^2
        H[u, v] = (gamma_h - gamma_l) * (1 - exp(-c * d2 / d0^2)) + gamma_l
    end

    # 4) 필터 적용 후 공간 영역으로 역변환
    G = H .* F
    g = real.(ifft(ifftshift(G)))

    # 5) 지수 변환으로 로그 복원
    result = exp.(g) .- 1.0

    # 0~1 범위로 정규화
    lo, hi = extrema(result)
    return (result .- lo) ./ (hi - lo + eps())
end

function main()
    img = testimage("mandril_gray")   # 512x512 그레이스케일 테스트 이미지
    gray = Float64.(Gray.(img))

    filtered = homomorphic_filter(gray; d0=30.0, c=1.0, gamma_l=0.3, gamma_h=2.0)

    save("original.png", Gray.(gray))
    save("homomorphic_filtered.png", Gray.(clamp01.(filtered)))

    println("완료: original.png, homomorphic_filtered.png 저장됨")
    println("입력 크기: ", size(gray))
    println("출력 범위: ", extrema(filtered))
end

main()
