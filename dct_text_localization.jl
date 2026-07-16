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
# ---------------------------------------------------------------------------

"""
block_text_energy(img, bs=8) -> (BHTE, BVTE)

img를 겹치지 않는 bs×bs 블록으로 나누고 각 블록에 2D DCT-II를 적용해,
Block Horizontal Text Energy(BHTE, 순수 가로방향 AC 계수 절대값의 합)와
Block Vertical Text Energy(BVTE, 순수 세로방향 AC 계수 절대값의 합)를
블록 단위 행렬로 반환한다(Eq.2-3의 재구성).
"""
function block_text_energy(img::AbstractMatrix{<:Real}, bs::Int=8)
    rows, cols = size(img)
    @assert rows % bs == 0 && cols % bs == 0 "이미지 크기가 블록 크기의 배수여야 합니다"
    nr, nc = rows ÷ bs, cols ÷ bs
    BHTE = zeros(Float64, nr, nc)
    BVTE = zeros(Float64, nr, nc)
    @inbounds for bi in 1:nr, bj in 1:nc
        blk = @view img[(bi-1)*bs+1:bi*bs, (bj-1)*bs+1:bj*bs]
        F = FFTW.dct(Float64.(blk))
        BHTE[bi, bj] = sum(abs, @view F[1, 2:bs])   # u=0, v=1..bs-1 (가로 방향)
        BVTE[bi, bj] = sum(abs, @view F[2:bs, 1])   # u=1..bs-1, v=0 (세로 방향)
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
localize_text_dct(img; bs=8) -> TextLocResult

Section 3의 알고리즘 재구성:
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
"""
function localize_text_dct(img::AbstractMatrix{<:Real}; bs::Int=8,
                            var_eps::Float64=1e-5, max_iter::Int=50)
    rows, cols = size(img)
    nr, nc = rows ÷ bs, cols ÷ bs
    BHTE, BVTE = block_text_energy(img, bs)

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
# 6. 데모
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
end

main()
