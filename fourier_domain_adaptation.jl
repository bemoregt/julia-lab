using Images
using TestImages
using FFTW

"""
fda_channel(src, tar; beta) -> Matrix{Float64}

단일 채널에 대한 FDA(Fourier Domain Adaptation) 연산. 소스 영상의 위상
(phase, 구조/내용 정보)은 그대로 유지한 채, 저주파(스펙트럼 중심 부근)
진폭(amplitude, 스타일/조명/색감 정보)만 타겟 영상의 것으로 바꿔치기한다.

    Fsrc = fftshift(fft(src)),  Ftar = fftshift(fft(tar))
    중심에서 beta 비율만큼의 저주파 영역에서 |Fsrc| -> |Ftar|로 교체
    출력 = ifft( |Fsrc'| * exp(i*angle(Fsrc)) )

`beta`가 클수록 더 넓은 저주파 대역을 교체하여 타겟의 스타일이 더 강하게
전이된다 (Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic
Segmentation", CVPR 2020).
"""
function fda_channel(src::Matrix{Float64}, tar::Matrix{Float64}, beta::Float64)
    H, W = size(src)
    Fsrc = fftshift(fft(src))
    Ftar = fftshift(fft(tar))

    amp_src = abs.(Fsrc)
    pha_src = angle.(Fsrc)
    amp_tar = abs.(Ftar)

    cy, cx = H ÷ 2 + 1, W ÷ 2 + 1
    bh = max(1, round(Int, beta * H / 2))
    bw = max(1, round(Int, beta * W / 2))
    r1, r2 = max(1, cy - bh), min(H, cy + bh)
    c1, c2 = max(1, cx - bw), min(W, cx + bw)

    amp_new = copy(amp_src)
    amp_new[r1:r2, c1:c2] .= amp_tar[r1:r2, c1:c2]

    Fnew = amp_new .* exp.(im .* pha_src)
    return real.(ifft(ifftshift(Fnew)))
end

"""
fda(src_rgb, tar_rgb; beta) -> Matrix{RGB{N0f8}}

RGB 컬러 영상에 대해 각 채널(R,G,B)별로 독립적으로 FDA를 적용한다.
두 영상의 크기가 같아야 한다.
"""
function fda(src_rgb::Matrix{<:Colorant}, tar_rgb::Matrix{<:Colorant}, beta::Float64)
    @assert size(src_rgb) == size(tar_rgb) "두 영상의 크기가 같아야 합니다"
    src = Float64.(channelview(RGB.(src_rgb)))   # 3 x H x W
    tar = Float64.(channelview(RGB.(tar_rgb)))

    out = similar(src)
    for ch in 1:3
        out[ch, :, :] = fda_channel(src[ch, :, :], tar[ch, :, :], beta)
    end
    out = clamp.(out, 0.0, 1.0)
    return colorview(RGB, out) |> x -> RGB{N0f8}.(x)
end

function main()
    src_img = testimage("lighthouse")      # 소스 도메인: 차가운 톤의 야외 대낮 풍경
    tar_img = testimage("monarch_color")    # 타겟 도메인: 따뜻한 톤의 자연광 사진
    println("소스 크기: ", size(src_img), "  타겟 크기: ", size(tar_img))

    save("fda_source.png", src_img)
    save("fda_target.png", tar_img)

    betas = [0.01, 0.05, 0.15]
    panels = Matrix{RGB{N0f8}}[RGB{N0f8}.(src_img)]
    for beta in betas
        out = fda(src_img, tar_img, beta)
        save("fda_output_beta_$(beta).png", out)
        push!(panels, out)
    end
    push!(panels, RGB{N0f8}.(tar_img))

    save("fda_comparison.png", hcat(panels...))

    println("완료: fda_source.png, fda_target.png, fda_output_beta_*.png, fda_comparison.png 저장됨")
    println("패널(왼쪽부터): source, beta=0.01, beta=0.05, beta=0.15, target")
end

main()
