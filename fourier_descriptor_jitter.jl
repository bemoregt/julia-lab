using Images
using TestImages
using FFTW
using Random
using Statistics: mean

"""
largest_component(mask::BitMatrix) -> BitMatrix

이진 마스크에서 8-연결(BFS) 연결요소들을 찾아 가장 큰 것만 남긴 마스크를
반환한다. (외부 패키지 없이 직접 구현한 간단한 연결요소 라벨링)
"""
function largest_component(mask::AbstractMatrix{Bool})
    rows, cols = size(mask)
    visited = falses(rows, cols)
    best_pixels = Vector{Tuple{Int,Int}}()
    neigh8 = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))

    for r in 1:rows, c in 1:cols
        if mask[r, c] && !visited[r, c]
            queue = [(r, c)]
            visited[r, c] = true
            comp = Vector{Tuple{Int,Int}}()
            head = 1
            while head <= length(queue)
                (cr, cc) = queue[head]
                head += 1
                push!(comp, (cr, cc))
                for (dr, dc) in neigh8
                    nr, nc = cr + dr, cc + dc
                    if 1 <= nr <= rows && 1 <= nc <= cols && mask[nr, nc] && !visited[nr, nc]
                        visited[nr, nc] = true
                        push!(queue, (nr, nc))
                    end
                end
            end
            if length(comp) > length(best_pixels)
                best_pixels = comp
            end
        end
    end

    out = falses(rows, cols)
    for (r, c) in best_pixels
        out[r, c] = true
    end
    return out
end

"""
moore_boundary_trace(mask) -> Vector{Tuple{Int,Int}}

무어 경계 추적(Moore-Neighbor tracing) 알고리즘으로 이진 마스크(단일
연결요소)의 외곽 경계를 시계방향으로 순서화된 (row, col) 좌표 목록으로
추출한다.
"""
function moore_boundary_trace(mask::AbstractMatrix{Bool})
    rows, cols = size(mask)
    is_fg(r, c) = (1 <= r <= rows && 1 <= c <= cols) && mask[r, c]

    b0 = nothing
    for r in 1:rows, c in 1:cols
        if mask[r, c]
            b0 = (r, c)
            break
        end
    end
    b0 === nothing && error("전경 픽셀이 없습니다")

    dirs = [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]

    boundary = [b0]
    b = b0
    bt = 0

    max_iter = rows * cols * 8
    for _ in 1:max_iter
        found = false
        next_b = b
        next_bt = bt
        for k in 0:7
            idx = mod(bt + k, 8)
            dr, dc = dirs[idx + 1]
            cand = (b[1] + dr, b[2] + dc)
            if is_fg(cand...)
                next_b = cand
                next_bt = mod(idx - 1, 8)
                found = true
                break
            end
        end
        if !found
            break
        end
        if next_b == b0
            break
        end
        push!(boundary, next_b)
        b, bt = next_b, next_bt
    end
    return boundary
end

"""
jitter_descriptors(Z, rng; base_amplitude, decay, keep_low) -> Vector{ComplexF64}

푸리에 서술자(Fourier Descriptor) Z의 각 고조파에 무작위 복소 잡음을 더해
윤곽선을 지터링(jitter)한다. 잡음의 공간적(픽셀 단위) 진폭은 저주파에서
`base_amplitude` 픽셀 정도이고 주파수가 높아질수록 `1/|f|^decay` 비율로
작아지도록 직접 보정한다 (단순히 각 고조파 "자신의" 크기에 비례시키면,
원처럼 고주파 성분이 원래 0에 가까운 매끈한 형태에서는 지터가 거의
보이지 않게 되므로, 절대적인 픽셀 진폭으로 지정한다). 이렇게 하면
저주파는 크게, 고주파는 작게 흔들려 잔거스러운 잡음이 아니라 매끄러운
형태 변형처럼 보인다. `keep_low`개의 최저 고조파(전체 위치·크기·회전을
결정)는 손대지 않고 그대로 유지해, 객체의 전체적인 위치/크기는
보존하면서 국소적인 윤곽 형태만 무작위로 변형한다 -- 분할(segmentation)
마스크 데이터 증강(augmentation)에 쓰기 위한 목적.
"""
function jitter_descriptors(Z::Vector{ComplexF64}, rng::AbstractRNG;
                             base_amplitude::Float64=6.0, decay::Float64=1.0,
                             keep_low::Int=2)
    N = length(Z)
    Zj = copy(Z)
    for i in 1:N
        f = i - 1
        if f > N ÷ 2
            f -= N
        end
        if abs(f) >= keep_low
            scale = base_amplitude * N / (abs(f)^decay)
            Zj[i] += scale * (randn(rng) + im * randn(rng))
        end
    end
    return Zj
end

function draw_line!(canvas::AbstractMatrix{Bool}, p0::ComplexF64, p1::ComplexF64)
    rows, cols = size(canvas)
    x0, y0 = round(Int, real(p0)), round(Int, imag(p0))
    x1, y1 = round(Int, real(p1)), round(Int, imag(p1))
    dx = abs(x1 - x0); sx = x0 < x1 ? 1 : -1
    dy = -abs(y1 - y0); sy = y0 < y1 ? 1 : -1
    err = dx + dy
    x, y = x0, y0
    while true
        if 1 <= y <= rows && 1 <= x <= cols
            canvas[y, x] = true
        end
        (x == x1 && y == y1) && break
        e2 = 2 * err
        if e2 >= dy
            err += dy; x += sx
        end
        if e2 <= dx
            err += dx; y += sy
        end
    end
end

"""
rasterize_contour(pts, rows, cols) -> BitMatrix

닫힌 윤곽선 점열(연속 좌표)을 브레젠험(Bresenham) 직선으로 이어 경계를
그린 뒤, 중심점에서 BFS로 내부를 채워 채워진 이진 마스크를 만든다.
(마스크 데이터 증강의 최종 산출물)
"""
function rasterize_contour(pts::Vector{ComplexF64}, rows::Int, cols::Int)
    canvas = falses(rows, cols)
    n = length(pts)
    for i in 1:n
        draw_line!(canvas, pts[i], pts[mod1(i + 1, n)])
    end

    seed = (clamp(round(Int, mean(imag.(pts))), 1, rows),
            clamp(round(Int, mean(real.(pts))), 1, cols))

    filled = copy(canvas)
    visited = falses(rows, cols)
    if !canvas[seed...]
        queue = [seed]
        visited[seed...] = true
        head = 1
        while head <= length(queue)
            (r, c) = queue[head]; head += 1
            filled[r, c] = true
            for (dr, dc) in ((-1,0),(1,0),(0,-1),(0,1))
                nr, nc = r + dr, c + dc
                if 1 <= nr <= rows && 1 <= nc <= cols && !visited[nr, nc] && !canvas[nr, nc]
                    visited[nr, nc] = true
                    push!(queue, (nr, nc))
                end
            end
        end
    end
    return filled
end

function main()
    # blobs 샘플(세균 군집 이미지)에서 가장 큰 개별 반점(땅콩 모양)의 윤곽선을 사용
    img = testimage("blobs")
    gray = Float64.(Gray.(img))
    rows, cols = size(gray)

    thresh = sum(gray) / length(gray)
    mask = gray .< thresh   # 어두운 반점을 전경으로 (밝은 배경과 분리)
    blob = largest_component(mask)
    boundary = moore_boundary_trace(blob)
    println("입력 크기: ", size(gray))
    println("추출된 윤곽선 경계 픽셀 수: ", length(boundary))

    z = ComplexF64[c + im * r for (r, c) in boundary]
    Z = fft(z)

    base_rgb = RGB{N0f8}.(Gray.(gray))
    panels_overlay = Matrix{RGB{N0f8}}[]
    panels_mask = Matrix{Gray{N0f8}}[]

    orig_mask = rasterize_contour(z, rows, cols)
    push!(panels_overlay, begin
        c = copy(base_rgb)
        for p in z
            r_, c_ = round(Int, imag(p)), round(Int, real(p))
            if 1 <= r_ <= rows && 1 <= c_ <= cols
                c[r_, c_] = RGB{N0f8}(1, 0, 0)
            end
        end
        c
    end)
    push!(panels_mask, Gray.(orig_mask))

    seeds = [101, 202, 303, 404]
    for s in seeds
        rng = MersenneTwister(s)
        Zj = jitter_descriptors(Z, rng; base_amplitude=1.0, decay=1.8, keep_low=2)
        zj = ifft(Zj)

        m = rasterize_contour(zj, rows, cols)
        canvas = copy(base_rgb)
        for p in zj
            r_, c_ = round(Int, imag(p)), round(Int, real(p))
            if 1 <= r_ <= rows && 1 <= c_ <= cols
                canvas[r_, c_] = RGB{N0f8}(1, 0, 0)
            end
        end
        push!(panels_overlay, canvas)
        push!(panels_mask, Gray.(m))
    end

    save("jitter_overlay_comparison.png", hcat(panels_overlay...))
    save("jitter_mask_comparison.png", hcat(panels_mask...))

    # 반점 주변을 확대한 비교본 (원본 영상 대비 물체가 작으므로 스무딩/지터링 효과 확인용)
    rs = [p[1] for p in boundary]; cs = [p[2] for p in boundary]
    margin = 15
    r0, r1 = max(1, minimum(rs) - margin), min(rows, maximum(rs) + margin)
    c0, c1 = max(1, minimum(cs) - margin), min(cols, maximum(cs) + margin)
    zoom_overlay = [imresize(p[r0:r1, c0:c1], ratio=4) for p in panels_overlay]
    zoom_mask = [imresize(p[r0:r1, c0:c1], ratio=4) for p in panels_mask]
    save("jitter_overlay_comparison_zoom.png", hcat(zoom_overlay...))
    save("jitter_mask_comparison_zoom.png", hcat(zoom_mask...))

    println("완료: jitter_overlay_comparison(_zoom).png, jitter_mask_comparison(_zoom).png 저장됨")
    println("패널(왼쪽부터): original, seed=101, seed=202, seed=303, seed=404")
end

main()
