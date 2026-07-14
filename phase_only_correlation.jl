using Images
using TestImages
using FFTW

"""
hanning2d(rows, cols) -> Matrix

2D Hanning(한닝) 윈도우. 이미지 경계에서의 불연속(에지 효과)이
FFT 위상에 잡음을 섞는 것을 줄이기 위해 상관관계 계산 전에 곱해준다.
"""
function hanning2d(rows::Int, cols::Int)
    wy = [0.5 - 0.5 * cos(2π * (i - 1) / (rows - 1)) for i in 1:rows]
    wx = [0.5 - 0.5 * cos(2π * (j - 1) / (cols - 1)) for j in 1:cols]
    return wy * wx'
end

"""
phase_only_correlation(img1, img2; window=true) -> Matrix

두 영상 간의 Phase-Only Correlation(POC, 위상한정상관) 맵을 계산한다.
진폭 성분을 모두 제거하고 위상 성분만 남긴 교차전력스펙트럼(cross-power
spectrum)을 역FFT하여, 두 영상 간 평행이동량에 해당하는 위치에
날카로운 상관 피크가 나타나도록 한다.

    R(u,v) = F1(u,v) * conj(F2(u,v)) / |F1(u,v) * conj(F2(u,v))|
    poc(x,y) = IFFT(R)(x,y)

반환된 맵은 `fftshift`로 중심에 0-시프트(피크 없음)가 오도록 정렬된다.
"""
function phase_only_correlation(img1::AbstractMatrix{<:Real},
                                 img2::AbstractMatrix{<:Real};
                                 window::Bool=true)
    @assert size(img1) == size(img2) "두 영상의 크기가 같아야 합니다"
    rows, cols = size(img1)

    w = window ? hanning2d(rows, cols) : ones(rows, cols)

    F1 = fft(Float64.(img1) .* w)
    F2 = fft(Float64.(img2) .* w)

    cross = F2 .* conj.(F1)
    R = cross ./ (abs.(cross) .+ eps())

    poc = real.(ifft(R))
    return fftshift(poc)
end

"""
find_peak(poc) -> (dy, dx, peak_value)

POC 맵에서 최댓값 위치를 찾아 중심(정합 시프트 0) 기준 상대 좌표
(dy, dx)로 변환한다.
"""
function find_peak(poc::AbstractMatrix{<:Real})
    rows, cols = size(poc)
    idx = argmax(poc)
    cy, cx = rows ÷ 2 + 1, cols ÷ 2 + 1
    dy = idx[1] - cy
    dx = idx[2] - cx
    return dy, dx, poc[idx]
end

function main()
    img = testimage("lighthouse")
    gray = Float64.(Gray.(img))
    rows, cols = size(gray)

    # 알려진 평행이동량으로 두 번째 영상 생성 (세로 dy, 가로 dx 픽셀 이동)
    dy_true, dx_true = 15, -10
    shifted = circshift(gray, (dy_true, dx_true))

    poc = phase_only_correlation(gray, shifted; window=true)
    dy_est, dx_est, peak_val = find_peak(poc)

    save("poc_img1.png", Gray.(gray))
    save("poc_img2_shifted.png", Gray.(shifted))

    # 시각화를 위해 0~1 정규화 (피크가 매우 뾰족하므로 대비를 위해 sqrt 적용)
    poc_pos = max.(poc, 0.0)
    lo, hi = extrema(poc_pos)
    poc_vis = sqrt.((poc_pos .- lo) ./ (hi - lo + eps()))
    save("poc_map.png", Gray.(clamp01.(poc_vis)))

    println("완료: poc_img1.png, poc_img2_shifted.png, poc_map.png 저장됨")
    println("입력 크기: ", size(gray))
    println("실제 이동량 (dy, dx)   = ($dy_true, $dx_true)")
    println("검출된 이동량 (dy, dx) = ($dy_est, $dx_est)")
    println("상관 피크값            = ", peak_val)
end

main()
