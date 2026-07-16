using Images
using FFTW
using ImageFiltering
using Random
using Statistics: mean, median, var

# =============================================================================
# Simple and Efficient Text Localization for Compressed Image in Mobile Phone
#   J.H. Kim, A. Canedo-Rodriguez, J.H. Kim, J. Kelly
#   J. Signal and Information Processing, 2014. DOI:10.4236/jsip.2014.54022
#
# 핵심 아이디어: 모바일폰 사진은 대부분 JPEG로 저장되므로, 압축을 풀지 않고
# JPEG이 이미 계산해 둔 8×8 블록 DCT 계수만으로 간판(signboard) 텍스트 영역을
# 찾는다. 문자(character)는 배경과 달리 가로/세로 두 방향 모두에서 강한
# 고주파 변화를 만들어내므로, 각 블록에서 "순수 가로 방향" 및 "순수 세로
# 방향" DCT 계수의 에너지를 정의하고(Block Text Energy), 이를 행(row)과
# 열(column) 단위로 누적한 프로파일에서 텍스트 줄과 텍스트 폭을 순차적으로
# (행 먼저, 그다음 그 행 범위 안에서 열) 좁혀 들어가 최종 경계상자를 얻는다.
#
# 주의(재구성 명시): 원문 PDF에서 본문 텍스트를 추출하는 과정에서 실제 수식
# (2)~(14)의 기호/계수가 이미지로만 남아 텍스트로 복원되지 않았다(본문에
# "(2)", "(3)"처럼 번호만 남고 수식 본문이 유실됨). 아래 구현은 논문 3장의
# 서술(Section 3, "Proposed Algorithm")과 4~5장의 설명에 기반해 알고리즘의
# 설계 의도— 블록별 방향 에너지 정의, 행→열 2단계 탐색, 에너지×중앙근접도
# 기준의 후보 선택(Eq.8), 중앙값 필터 스무딩(Eq.9-10), 반복적 배경 레벨
# 추정(Eq.11-14, Figure 7)—를 최대한 충실히 재현한 것이며, 원 논문의 정확한
# 계수/상수까지 동일함을 보장하지는 않는다.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. 블록 DCT 방향 에너지 (Section 3, Eq.2-3)
#    Gonzalez-Woods 관례: F(u,v)에서 u=행(세로) 방향 주파수, v=열(가로)
#    방향 주파수. 문자 획은 배경과 달리 가로/세로 두 방향 모두에서 뚜렷한
#    에지를 만들므로, "순수 가로 방향" 성분(u=0, v=1..7)과 "순수 세로
#    방향" 성분(u=1..7, v=0)의 에너지를 블록마다 따로 정의한다.
#
#    dct2_blocks/idct2_blocks를 분리해 둔 이유: 실제 모바일 JPEG 파이프라인
#    에서는 파일을 열기만 해도(엔트로피 디코딩+역양자화) 블록 DCT 계수 J가
#    "공짜로" 주어진다. 반면 공간 도메인 방식은 그 J를 픽셀로 되돌리는
#    IDCT("압축 해제", idct2_blocks)를 먼저 거쳐야만 분석을 시작할 수
#    있다 — 이 IDCT 단계 자체가 논문이 말하는 "avoiding the computationally
#    expensive process of uncompressing the image"의 실체이며, 아래 8절의
#    벤치마크가 정확히 이 차이를 측정한다.
# ---------------------------------------------------------------------------

"""dct2_blocks(img, bs=8) -> J, 블록별 직교정규 2D DCT-II (JPEG이 저장하는 표현)."""
function dct2_blocks(img::AbstractMatrix{<:Real}, bs::Int=8)
    rows, cols = size(img)
    @assert rows % bs == 0 && cols % bs == 0 "이미지 크기가 블록 크기의 배수여야 합니다"
    J = zeros(Float64, rows, cols)
    @inbounds for by in 0:bs:rows-1, bx in 0:bs:cols-1
        J[by+1:by+bs, bx+1:bx+bs] = FFTW.dct(Float64.(img[by+1:by+bs, bx+1:bx+bs]))
    end
    return J
end

"""idct2_blocks(J, bs=8) -> img, 블록별 역DCT("압축 해제"). 공간 도메인 방식만 필요로 하는 단계."""
function idct2_blocks(J::AbstractMatrix{<:Real}, bs::Int=8)
    rows, cols = size(J)
    img = zeros(Float64, rows, cols)
    @inbounds for by in 0:bs:rows-1, bx in 0:bs:cols-1
        img[by+1:by+bs, bx+1:bx+bs] = FFTW.idct(J[by+1:by+bs, bx+1:bx+bs])
    end
    return img
end

"""
energy_from_dct(J, bs=8) -> (BHTE, BVTE)

이미 계산되어 있는 블록 DCT 계수 J로부터 직접 Block Horizontal/Vertical
Text Energy를 읽어낸다(Eq.2-3의 재구성). 재계산이 필요 없으므로 실제 JPEG
파이프라인에서는 이 단계의 한계비용이 거의 0이다.
"""
function energy_from_dct(J::AbstractMatrix{<:Real}, bs::Int=8)
    rows, cols = size(J)
    nr, nc = rows ÷ bs, cols ÷ bs
    BHTE = zeros(Float64, nr, nc)
    BVTE = zeros(Float64, nr, nc)
    @inbounds for bi in 1:nr, bj in 1:nc
        F = @view J[(bi-1)*bs+1:bi*bs, (bj-1)*bs+1:bj*bs]
        BHTE[bi, bj] = sum(abs, @view F[1, 2:bs])   # u=0, v=1..bs-1 (가로 방향)
        BVTE[bi, bj] = sum(abs, @view F[2:bs, 1])   # u=1..bs-1, v=0 (세로 방향)
    end
    return BHTE, BVTE
end

"""block_text_energy(img, bs=8) -> (BHTE, BVTE). 픽셀에서 바로 시작할 때의 편의 함수(=dct2_blocks+energy_from_dct)."""
block_text_energy(img::AbstractMatrix{<:Real}, bs::Int=8) = energy_from_dct(dct2_blocks(img, bs), bs)

# ---------------------------------------------------------------------------
# 1b. 공간 도메인 대조군 (Section 2에서 언급하는 "Texture-Based" 부류: 에지/
#     그라디언트 에너지를 압축 해제된 픽셀에서 직접 계산) — 8절 벤치마크에서
#     DCT 도메인 방식과 정확도는 동일한 탐색 로직으로, 속도만 비교하기 위한
#     대응 특징 추출기.
# ---------------------------------------------------------------------------

const SOBEL_X = Float64[-1 0 1; -2 0 2; -1 0 1]        # 가로(열) 방향 변화 검출
const SOBEL_Y = Float64[-1 -2 -1; 0 0 0; 1 2 1]        # 세로(행) 방향 변화 검출

function conv2d_same(img::AbstractMatrix{<:Real}, kernel::AbstractMatrix{<:Real})
    kh, kw = size(kernel)
    ph, pw = kh ÷ 2, kw ÷ 2
    rows, cols = size(img)
    padded = zeros(Float64, rows + 2ph, cols + 2pw)
    padded[ph+1:ph+rows, pw+1:pw+cols] = img
    out = zeros(Float64, rows, cols)
    @inbounds for i in 1:rows, j in 1:cols
        out[i, j] = sum(@view(padded[i:i+kh-1, j:j+kw-1]) .* kernel)
    end
    return out
end

"""
spatial_text_energy(img, bs=8) -> (BHTE, BVTE)

완전히 압축 해제된 전체 해상도 픽셀에 Sobel류 그라디언트 필터를 적용해
가로/세로 변화 에너지를 구한 뒤 블록 단위로 합산한다. block_text_energy와
같은 물리적 의미(가로/세로 방향 에너지)를 갖도록 설계했으므로, 이후의
행→열 탐색(localize_from_energy)에 그대로 대입해 정확도는 동일하게
유지하면서 "특징을 어디서/어떻게 얻었는가"에 따른 속도차만 분리해서 잴 수
있다.
"""
function spatial_text_energy(img::AbstractMatrix{<:Real}, bs::Int=8)
    gx = conv2d_same(img, SOBEL_X)
    gy = conv2d_same(img, SOBEL_Y)
    rows, cols = size(img)
    nr, nc = rows ÷ bs, cols ÷ bs
    BHTE = zeros(Float64, nr, nc)
    BVTE = zeros(Float64, nr, nc)
    @inbounds for bi in 1:nr, bj in 1:nc
        BHTE[bi, bj] = sum(abs, @view gx[(bi-1)*bs+1:bi*bs, (bj-1)*bs+1:bj*bs])
        BVTE[bi, bj] = sum(abs, @view gy[(bi-1)*bs+1:bi*bs, (bj-1)*bs+1:bj*bs])
    end
    return BHTE, BVTE
end

# ---------------------------------------------------------------------------
# 2. 공용 유틸: 1차원 중앙값 필터, 이진 마스크의 연속 구간(그룹) 탐색
# ---------------------------------------------------------------------------

"""medfilt3(v) -> window=3 중앙값 필터(양끝은 그대로 둠), Eq.9-10에서 사용."""
function medfilt3(v::AbstractVector{<:Real})
    n = length(v)
    out = copy(v)
    @inbounds for i in 2:n-1
        out[i] = median(@view v[i-1:i+1])
    end
    return out
end

"""find_groups(mask) -> [(start,stop), ...], mask에서 true가 연속되는 구간들."""
function find_groups(mask::AbstractVector{Bool})
    groups = Tuple{Int,Int}[]
    n = length(mask)
    i = 1
    while i <= n
        if mask[i]
            j = i
            while j + 1 <= n && mask[j+1]
                j += 1
            end
            push!(groups, (i, j))
            i = j + 1
        else
            i += 1
        end
    end
    return groups
end

overlap_len(a::Tuple{Int,Int}, b::Tuple{Int,Int}) = max(0, min(a[2], b[2]) - max(a[1], b[1]) + 1)

# ---------------------------------------------------------------------------
# 3. 텍스트 로컬라이제이션 결과
# ---------------------------------------------------------------------------

struct TextLocResult
    r1::Int; r2::Int   # 텍스트 줄의 블록-행 경계
    c1::Int; c2::Int   # 텍스트 폭의 블록-열 경계
    Trow::Vector{Float64}   # 행 방향 Text Energy 프로파일 (전체 이미지)
    scol::Vector{Float64}   # 선택된 행 범위 안에서의 정규화된 열 프로파일 s(j)
    thr_row::Float64
    thr_col::Float64
end

"""
localize_from_energy(BHTE, BVTE; var_eps, max_iter) -> TextLocResult

Section 3의 알고리즘 재구성 (특징 행렬 BHTE/BVTE의 출처와 무관한 공용 탐색):
  1) 행 프로파일 T_row(i) = ΣH_row(i)·ΣV_row(i) 을 평균 임계값으로 거칠게
     이진화해 후보 그룹들을 찾고, "에너지가 크면서 이미지 중앙에 가까운"
     그룹을 Eq.8 방식(에너지 / (1+중앙과의 거리))으로 선택한다.
  2) 선택된 그룹 주변의 지역 윈도우에서 임계값을 다시 계산해 행 경계를
     정밀화한다(Figure 5의 2단계 정밀화).
  3) 정밀화된 행 범위 안에서 열 프로파일을 만들고, 중앙값 필터를 두 번
     적용한 S(j)(Eq.9-10)를 정규화한 뒤, 배경 레벨을 반복적으로 추정하는
     Eq.11-14(Figure 7)의 알고리즘으로 전경(텍스트) 임계값을 구한다.
  4) 임계값을 넘는 열 그룹 중 중앙에 가장 가까운 것("middle peak")을
     최종 텍스트 폭 경계로 선택한다.

DCT 도메인(block_text_energy/energy_from_dct)과 공간 도메인
(spatial_text_energy) 어느 쪽에서 만든 BHTE/BVTE를 넣어도 동일하게
동작하므로, 8절 벤치마크에서 "탐색 알고리즘은 고정하고 특징 추출 단계의
비용만" 공정하게 비교할 수 있다.
"""
function localize_from_energy(BHTE::AbstractMatrix{<:Real}, BVTE::AbstractMatrix{<:Real};
                               var_eps::Float64=1e-5, max_iter::Int=50)
    nr, nc = size(BHTE)

    # --- (a) 행 프로파일 & 조대(coarse) 텍스트 줄 후보 선택 ---
    Hrow = vec(sum(BHTE, dims=2))
    Vrow = vec(sum(BVTE, dims=2))
    Trow = Hrow .* Vrow
    thr_row = mean(Trow)
    groups = find_groups(Trow .> thr_row)
    isempty(groups) && error("텍스트 줄 후보를 찾지 못했습니다")
    mid_r = (nr + 1) / 2
    score(g) = sum(@view Trow[g[1]:g[2]]) / (1 + abs((g[1] + g[2]) / 2 - mid_r))  # Eq.8
    r1, r2 = groups[argmax(score.(groups))]

    # --- (b) 지역 윈도우에서 임계값 재추정 -> 행 경계 정밀화 (Figure 5) ---
    h = r2 - r1 + 1
    lo = max(1, r1 - h ÷ 2)
    hi = min(nr, r2 + h ÷ 2)
    thr_row2 = mean(@view Trow[lo:hi])
    local_groups = find_groups(Trow[lo:hi] .> thr_row2)
    if !isempty(local_groups)
        target = (r1 - lo + 1, r2 - lo + 1)
        ov = [overlap_len(g, target) for g in local_groups]
        best = local_groups[argmax(ov)]
        r1, r2 = best[1] + lo - 1, best[2] + lo - 1
    end

    # --- (c) 선택된 행 범위 안에서 열 프로파일, 중앙값 필터 2회 (Eq.9-10) ---
    Hcol = vec(sum(@view(BHTE[r1:r2, :]), dims=1))
    Vcol = vec(sum(@view(BVTE[r1:r2, :]), dims=1))
    S = medfilt3(Hcol) .* medfilt3(Vcol)
    S = medfilt3(S)
    slo, shi = extrema(S)
    s = (S .- slo) ./ (shi - slo + eps())

    # --- (d) 반복적 배경 레벨 추정으로 전경(텍스트) 임계값 산출 (Eq.11-14) ---
    T = mean(s)
    for _ in 1:max_iter
        bg = s[s .< T]
        isempty(bg) && break
        sigma2 = var(bg; corrected=false)
        T = mean(bg)
        sigma2 < var_eps && break
    end

    fg_groups = find_groups(s .> T)
    isempty(fg_groups) && error("텍스트 폭 후보를 찾지 못했습니다")
    mid_c = (nc + 1) / 2
    c1, c2 = fg_groups[argmin([abs((g[1] + g[2]) / 2 - mid_c) for g in fg_groups])]

    return TextLocResult(r1, r2, c1, c2, Trow, s, thr_row, T)
end

"""localize_text_dct(img; bs=8) -> TextLocResult. DCT 도메인 경로: block_text_energy + localize_from_energy."""
function localize_text_dct(img::AbstractMatrix{<:Real}; bs::Int=8,
                            var_eps::Float64=1e-5, max_iter::Int=50)
    BHTE, BVTE = block_text_energy(img, bs)
    return localize_from_energy(BHTE, BVTE; var_eps=var_eps, max_iter=max_iter)
end

"""localize_text_spatial(img; bs=8) -> TextLocResult. 공간 도메인 경로(비교용): spatial_text_energy + localize_from_energy."""
function localize_text_spatial(img::AbstractMatrix{<:Real}; bs::Int=8,
                                var_eps::Float64=1e-5, max_iter::Int=50)
    BHTE, BVTE = spatial_text_energy(img, bs)
    return localize_from_energy(BHTE, BVTE; var_eps=var_eps, max_iter=max_iter)
end

"""block bounds -> 픽셀 좌표 (r1px,r2px,c1px,c2px), bs=블록 크기."""
pixel_bbox(res::TextLocResult, bs::Int=8) =
    ((res.r1 - 1) * bs + 1, res.r2 * bs, (res.c1 - 1) * bs + 1, res.c2 * bs)

# ---------------------------------------------------------------------------
# 4. 5×7 비트맵 폰트로 합성 간판(signboard) 텍스트 장면 생성 (실험용)
# ---------------------------------------------------------------------------

glyph(rows::Vector{String}) = BitMatrix(reduce(vcat, [reshape(collect(r) .== '#', 1, :) for r in rows]))

const FONT5x7 = Dict{Char,BitMatrix}(
    'C' => glyph([".###.", "#....", "#....", "#....", "#....", "#....", ".###."]),
    'A' => glyph([".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"]),
    'F' => glyph(["#####", "#....", "#....", "###..", "#....", "#....", "#...."]),
    'E' => glyph(["#####", "#....", "#....", "###..", "#....", "#....", "#####"]),
    'S' => glyph([".####", "#....", "#....", ".###.", "....#", "....#", "####."]),
    'T' => glyph(["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."]),
    'O' => glyph([".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."]),
    'P' => glyph(["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."]),
)

"""render_text(word; scale, gap) -> Float64 행렬(1=잉크,0=배경), 5×7 비트맵 폰트를 scale배 확대."""
function render_text(word::AbstractString; scale::Int=5, gap::Int=2)
    glyphs = [FONT5x7[c] for c in word]
    gh, gw = 7, 5
    n = length(glyphs)
    W = n * gw + (n - 1) * gap
    canvas = falses(gh, W)
    x = 1
    for g in glyphs
        canvas[:, x:x+gw-1] .= g
        x += gw + gap
    end
    return kron(Float64.(canvas), ones(scale, scale))
end

"""
make_scene(; rows, cols, word, seed) -> Float64 grayscale image in [0,1]

논문의 두 가지 가정(JPEG, 사용자가 목표 텍스트를 화면 중앙 부근에 위치시킴)에
맞춰, 불균일한 조명의 배경 위에 중앙 부근의 signboard 텍스트와, 알고리즘이
Eq.8의 "중앙 근접도" 기준으로 걸러내야 하는 좌상단 방해 텍스처(예: 나뭇잎/
벽돌 같은 고주파 배경)를 함께 배치한다.
"""
function make_scene(; rows::Int=256, cols::Int=320, word::AbstractString="CAFE",
                     seed::Int=42)
    rng = MersenneTwister(seed)
    bg = [0.75 + 0.15 * sin(2π * (x / cols * 1.3 + y / rows * 0.7)) for y in 1:rows, x in 1:cols]
    bg .+= 0.05 .* randn(rng, rows, cols)
    scene = clamp.(bg, 0.55, 0.95)

    txt = render_text(word)
    th, tw = size(txt)
    ty, tx = rows ÷ 2 - th ÷ 2, cols ÷ 2 - tw ÷ 2
    ink = 0.10 .+ 0.04 .* randn(rng, th, tw)
    region = @view scene[ty+1:ty+th, tx+1:tx+tw]
    region .= region .* (1 .- txt) .+ ink .* txt

    dh, dw, dy, dx = 40, 64, 14, 14   # 좌상단 방해 텍스처(고주파 잡음 패치)
    scene[dy+1:dy+dh, dx+1:dx+dw] .= clamp.(0.5 .+ 0.35 .* randn(rng, dh, dw), 0.0, 1.0)

    return clamp.(scene, 0.0, 1.0)
end

"""
uneven_lighting(img) -> 카메라 비네팅(vignetting) 스타일의 방사형 조명
불균일을 적용한다. 순수 선형 램프는 8×8 블록 내에서 거의 전부 k=1 AC
계수 하나로만 새어들어가(DCT 기저함수 1이 단조증가 형태이기 때문) 우리의
가로/세로 방향 에너지 정의를 직접 오염시키는 극단적 사례가 되므로, 실제
사진에 더 가까운 완만한 방사형 감쇠를 사용한다.
"""
function uneven_lighting(img::AbstractMatrix{<:Real})
    rows, cols = size(img)
    cy, cx = (rows + 1) / 2, (cols + 1) / 2
    rmax = sqrt(cy^2 + cx^2)
    vignette = [1.0 - 0.35 * (sqrt((y - cy)^2 + (x - cx)^2) / rmax)^2 for y in 1:rows, x in 1:cols]
    clamp.(img .* vignette, 0.0, 1.0)
end

blurred_image(img::AbstractMatrix{<:Real}) = imfilter(img, Kernel.gaussian(2.5))

dim_light(img::AbstractMatrix{<:Real}; rng=MersenneTwister(7)) =
    clamp.(img .* 0.35 .+ 0.02 .* randn(rng, size(img)...), 0.0, 1.0)

# ---------------------------------------------------------------------------
# 5. 시각화 유틸
# ---------------------------------------------------------------------------

function draw_bbox!(img::AbstractMatrix{RGB{Float64}}, r1::Int, r2::Int, c1::Int, c2::Int;
                     color::RGB{Float64}=RGB(1.0, 0.0, 0.0), t::Int=2)
    rows, cols = size(img)
    r1c, r2c = clamp(r1, 1, rows), clamp(r2, 1, rows)
    c1c, c2c = clamp(c1, 1, cols), clamp(c2, 1, cols)
    for k in 0:t-1
        img[clamp(r1c + k, 1, rows), c1c:c2c] .= color
        img[clamp(r2c - k, 1, rows), c1c:c2c] .= color
        img[r1c:r2c, clamp(c1c + k, 1, cols)] .= color
        img[r1c:r2c, clamp(c2c - k, 1, cols)] .= color
    end
end

function hcat_with_gap(imgs::Vector{<:AbstractMatrix{RGB{Float64}}}; gap::Int=8)
    h = size(imgs[1], 1)
    sep = fill(RGB(1.0, 1.0, 1.0), h, gap)
    out = imgs[1]
    for im in imgs[2:end]
        out = hcat(out, sep, im)
    end
    return out
end

# ---------------------------------------------------------------------------
# 6. DCT 도메인 vs 공간 도메인 속도 벤치마크
#    논문 Section 4-A의 핵심 주장 — "JPEG 도메인에서 작업하면 압축 해제라는
#    비싼 과정을 피할 수 있다" — 을 직접 측정한다. 두 경로 모두 동일한
#    블록 DCT 계수 J에서 출발한다: 실제 모바일 기기가 JPEG 파일을 여는
#    순간 J는 엔트로피 디코딩+역양자화만으로 이미 손에 들어오므로, J를
#    얻는 비용은 "공짜"로 간주해 벤치마크 시간에서 제외한다.
#      · DCT 도메인 경로   : energy_from_dct(J) -> localize_from_energy
#      · 공간 도메인 경로  : idct2_blocks(J)(=압축 해제) -> spatial_text_energy
#        (전체 해상도 Sobel 필터) -> localize_from_energy
#    탐색 알고리즘(localize_from_energy)은 두 경로가 완전히 동일하므로,
#    측정되는 시간차는 오직 "특징을 어디서 얻었는가"에서만 발생한다.
# ---------------------------------------------------------------------------

"""
benchmark_dct_vs_spatial(img; bs=8, reps=50) -> NamedTuple

동일한 블록 DCT 계수 J(JPEG 파일 파싱만으로 공짜로 얻는 데이터)에서 출발해
DCT 도메인 경로와 공간 도메인 경로의 처리시간을 reps회 반복 측정한
중앙값으로 비교한다. 공간 도메인 경로는 압축 해제(IDCT)와 특징 재계산
(Sobel) 두 단계로 나누어 각각 보고하므로, 속도차가 실제로 "압축 해제
과정"에서 비롯된다는 논문의 주장을 항목별로 검증할 수 있다.
"""
function benchmark_dct_vs_spatial(img::AbstractMatrix{<:Real}; bs::Int=8, reps::Int=50)
    J = dct2_blocks(img, bs)   # JPEG 파일을 파싱하면 공짜로 주어지는 데이터 (타이밍 제외)

    # 워밍업 (JIT 컴파일 비용을 측정에서 제외)
    localize_from_energy(energy_from_dct(J, bs)...)
    localize_from_energy(spatial_text_energy(idct2_blocks(J, bs), bs)...)

    t_dct = zeros(reps)
    t_decompress = zeros(reps)
    t_spatial_feat = zeros(reps)
    t_search_spatial = zeros(reps)

    local res_dct, res_spatial
    for k in 1:reps
        t_dct[k] = @elapsed begin
            BHTE, BVTE = energy_from_dct(J, bs)
            res_dct = localize_from_energy(BHTE, BVTE)
        end

        t_decompress[k] = @elapsed (decompressed = idct2_blocks(J, bs))
        t_spatial_feat[k] = @elapsed ((BHTE2, BVTE2) = spatial_text_energy(decompressed, bs))
        t_search_spatial[k] = @elapsed (res_spatial = localize_from_energy(BHTE2, BVTE2))
    end

    t_spatial_total = t_decompress .+ t_spatial_feat .+ t_search_spatial

    return (
        dct_ms=median(t_dct) * 1000,
        decompress_ms=median(t_decompress) * 1000,
        spatial_feature_ms=median(t_spatial_feat) * 1000,
        spatial_total_ms=median(t_spatial_total) * 1000,
        speedup=median(t_spatial_total) / median(t_dct),
        res_dct=res_dct,
        res_spatial=res_spatial,
    )
end

# ---------------------------------------------------------------------------
# 7. 데모
# ---------------------------------------------------------------------------

function run_and_report(name::AbstractString, img::AbstractMatrix{<:Real})
    t = @elapsed res = localize_text_dct(img)
    r1px, r2px, c1px, c2px = pixel_bbox(res)
    println("  [$(rpad(name, 13))] $(round(t * 1000, digits=2)) ms  ",
        "블록 rows[$(res.r1):$(res.r2)] cols[$(res.c1):$(res.c2)]  ",
        "픽셀 rows[$r1px:$r2px] cols[$c1px:$c2px]")
    rgb = RGB.(Gray.(clamp.(img, 0.0, 1.0)))
    draw_bbox!(rgb, r1px, r2px, c1px, c2px)
    save("dcttext_$(name).png", rgb)
    return rgb
end

function main()
    println("="^78)
    println("Simple and Efficient Text Localization for Compressed Image in Mobile Phone")
    println("Kim, Canedo-Rodriguez, Kim & Kelly (2014), DOI:10.4236/jsip.2014.54022")
    println("="^78)

    scene = make_scene()
    save("dcttext_source.png", Gray.(scene))
    npix = size(scene, 1) * size(scene, 2)
    println("\n[1] 합성 signboard 장면 (", size(scene, 1), "×", size(scene, 2),
        " = ", round(npix / 1e6, digits=3), " MP), 좌상단에 방해 텍스처 포함")

    println("\n[2] 기본 조건 및 Figure 9 스타일 강건성 조건들에 대한 로컬라이제이션 결과")
    println("      (논문 Section 5: 0.3MP 영상 기준 0.12초 처리시간과 비교)")
    panel_base = run_and_report("baseline", scene)
    panel_uneven = run_and_report("uneven_light", uneven_lighting(scene))
    panel_blur = run_and_report("blurred", blurred_image(scene))
    panel_dim = run_and_report("dim_light", dim_light(scene))

    comparison = hcat_with_gap([panel_base, panel_uneven, panel_blur, panel_dim])
    save("dcttext_comparison.png", comparison)
    println("\n완료: dcttext_source.png, dcttext_{baseline,uneven_light,blurred,dim_light}.png, ",
        "dcttext_comparison.png 저장됨")

    println("\n[3] DCT 도메인 vs 공간 도메인 속도 비교 (50회 반복 중앙값)")
    println("      두 경로 모두 동일한 블록 DCT 계수 J에서 출발 (J를 얻는 비용 자체는",
        " JPEG 파일을 여는 것만으로 공짜이므로 측정에서 제외)")
    bench = benchmark_dct_vs_spatial(scene)
    println("  DCT 도메인   전체(특징 읽기 + 탐색)            : ",
        round(bench.dct_ms, digits=4), " ms")
    println("  공간 도메인  압축 해제(IDCT)만                 : ",
        round(bench.decompress_ms, digits=4), " ms")
    println("  공간 도메인  Sobel 특징 재계산만               : ",
        round(bench.spatial_feature_ms, digits=4), " ms")
    println("  공간 도메인  전체(압축해제 + 특징 + 탐색)      : ",
        round(bench.spatial_total_ms, digits=4), " ms")
    println("  => 공간 도메인 방식이 DCT 도메인 방식보다 약 ",
        round(bench.speedup, digits=2), "배 느림")
    r1a, r2a, c1a, c2a = pixel_bbox(bench.res_dct)
    r1b, r2b, c1b, c2b = pixel_bbox(bench.res_spatial)
    println("  검출 결과 일치 확인 - DCT: 픽셀 rows[$r1a:$r2a] cols[$c1a:$c2a]  ",
        "공간: 픽셀 rows[$r1b:$r2b] cols[$c1b:$c2b]")
end

main()
