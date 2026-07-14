using Images
using TestImages
using FFTW
using Statistics

"""
filtergrids(rows, cols) -> (radius, theta)

주파수 영역 좌표 그리드 생성. 중심(0,0)이 배열의 (1,1)에 오도록
`ifftshift`를 적용하여 `fft`의 출력 레이아웃과 바로 맞물리게 한다.
"""
function filtergrids(rows::Int, cols::Int)
    xrange = isodd(cols) ? ((-(cols-1)÷2):((cols-1)÷2)) ./ cols : ((-cols÷2):(cols÷2-1)) ./ cols
    yrange = isodd(rows) ? ((-(rows-1)÷2):((rows-1)÷2)) ./ rows : ((-rows÷2):(rows÷2-1)) ./ rows

    x = repeat(reshape(collect(xrange), 1, cols), rows, 1)
    y = repeat(reshape(collect(yrange), rows, 1), 1, cols)

    radius = sqrt.(x .^ 2 .+ y .^ 2)
    theta = atan.(-y, x)

    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[1, 1] = 1.0   # DC 성분에서 log(0), 0-division 방지
    return radius, theta
end

"""
lowpassfilter(rows, cols, cutoff, n) -> Matrix

버터워스(Butterworth) 저역통과 필터. 로그-가버 필터가 나이퀴스트 주파수
부근에서 왜곡(ringing)되는 것을 억제하기 위해 곱해준다.
"""
function lowpassfilter(rows::Int, cols::Int, cutoff::Float64, n::Int)
    radius, _ = filtergrids(rows, cols)
    return 1.0 ./ (1.0 .+ (radius ./ cutoff) .^ (2n))
end

"""
phase_congruency(img; nscale, norient, minWaveLength, mult, sigmaOnf, k, cutOff, g)

Kovesi 방식의 다중 스케일·다중 방향 위상합동(Phase Congruency) 에지 맵을 계산한다.

- nscale: 스케일(주파수 대역) 개수
- norient: 방향(orientation) 개수
- minWaveLength: 최소 파장(가장 미세한 스케일)
- mult: 스케일 간 파장 배수
- sigmaOnf: 로그-가버 필터의 대역폭(작을수록 좁은 대역)
- k: 잡음 임계값 배수(표준편차의 몇 배를 잡음으로 볼지)
- cutOff, g: 주파수 확산에 따른 가중치(sigmoid) 파라미터

반환: PC (0~1 정규화된 위상합동 에지 맵)
"""
function phase_congruency(img::AbstractMatrix{<:Real};
                           nscale::Int=4,
                           norient::Int=6,
                           minWaveLength::Float64=3.0,
                           mult::Float64=2.1,
                           sigmaOnf::Float64=0.55,
                           k::Float64=2.0,
                           cutOff::Float64=0.5,
                           g::Float64=10.0)
    epsilon = 1e-4
    rows, cols = size(img)
    im = Float64.(img)
    IM = fft(im)

    radius, theta = filtergrids(rows, cols)
    sintheta = sin.(theta)
    costheta = cos.(theta)
    lp = lowpassfilter(rows, cols, 0.45, 15)

    PC_total = zeros(Float64, rows, cols)

    for o in 1:norient
        angl = (o - 1) * pi / norient

        ds = sintheta .* cos(angl) .- costheta .* sin(angl)
        dc = costheta .* cos(angl) .+ sintheta .* sin(angl)
        dtheta = abs.(atan.(ds, dc))
        dtheta = min.(dtheta .* norient ./ 2, pi)
        spread = (cos.(dtheta) .+ 1) ./ 2

        sumE = zeros(Float64, rows, cols)
        sumO = zeros(Float64, rows, cols)
        sumAn = zeros(Float64, rows, cols)
        maxAn = zeros(Float64, rows, cols)
        tau = 0.0

        for s in 1:nscale
            wavelength = minWaveLength * mult^(s - 1)
            fo = 1.0 / wavelength
            logGabor = exp.(-(log.(radius ./ fo)) .^ 2 ./ (2 * log(sigmaOnf)^2))
            logGabor .*= lp
            logGabor[1, 1] = 0.0

            filt = logGabor .* spread
            EO = ifft(IM .* filt)
            E = real.(EO)
            O = imag.(EO)

            sumE .+= E
            sumO .+= O
            An = sqrt.(E .^ 2 .+ O .^ 2)
            sumAn .+= An

            if s == 1
                tau = median(An) / sqrt(log(4))   # 레일리(Rayleigh) 잡음 추정
                maxAn = copy(An)
            else
                maxAn = max.(maxAn, An)
            end
        end

        # 스케일 간 잡음 에너지 기댓값/표준편차 (기하급수 합)
        totalTau = tau * (1 - (1 / mult)^nscale) / (1 - 1 / mult)
        estNoiseEnergyMean = totalTau * sqrt(pi / 2)
        estNoiseEnergySigma = totalTau * sqrt((4 - pi) / 2)
        T = estNoiseEnergyMean + k * estNoiseEnergySigma

        # 위상합동(벡터합 에너지 / 진폭합) = Morrone-Owens 정의
        XEnergy = sqrt.(sumE .^ 2 .+ sumO .^ 2)

        width = (sumAn ./ (maxAn .+ epsilon) .- 1) ./ (nscale - 1)
        weight = 1.0 ./ (1.0 .+ exp.(g .* (cutOff .- width)))

        PC_o = weight .* max.(XEnergy .- T, 0.0) ./ (sumAn .+ epsilon)
        PC_total .+= PC_o
    end

    lo, hi = extrema(PC_total)
    return (PC_total .- lo) ./ (hi - lo + epsilon)
end

function main()
    img = testimage("cameraman")
    gray = Float64.(Gray.(img))

    pc = phase_congruency(gray; nscale=4, norient=6)

    save("pc_original.png", Gray.(gray))
    save("pc_edges.png", Gray.(clamp01.(pc)))

    println("완료: pc_original.png, pc_edges.png 저장됨")
    println("입력 크기: ", size(gray))
    println("PC 범위: ", extrema(pc))
end

main()
