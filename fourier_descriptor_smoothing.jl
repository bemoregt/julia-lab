using Images
using TestImages
using FFTW

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
            # BFS
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

    # 시작점: 래스터 스캔 순서로 처음 만나는 전경 픽셀
    b0 = nothing
    for r in 1:rows, c in 1:cols
        if mask[r, c]
            b0 = (r, c)
            break
        end
    end
    b0 === nothing && error("전경 픽셀이 없습니다")

    # 시계 방향 8-이웃: W, NW, N, NE, E, SE, S, SW (0-based index)
    dirs = [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]

    boundary = [b0]
    b = b0
    bt = 0   # b0의 서쪽은 항상 배경(래스터 스캔상 처음 만난 전경이므로)

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
            break   # 고립된 단일 픽셀
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
low_pass_fd(Z, K) -> Vector{ComplexF64}

푸리에 서술자(Fourier Descriptor) Z에서 저주파(|k| <= K) 성분만 남기고
나머지 고주파 성분을 0으로 만든다. K가 작을수록 더 강하게 스무딩된다.
"""
function low_pass_fd(Z::Vector{ComplexF64}, K::Int)
    N = length(Z)
    Zf = zeros(ComplexF64, N)
    for i in 1:N
        f = i - 1
        if f > N ÷ 2
            f -= N
        end
        if abs(f) <= K
            Zf[i] = Z[i]
        end
    end
    return Zf
end

"""
smooth_contour(boundary, K) -> Vector{ComplexF64}

경계 좌표를 복소수 z = x + iy 로 표현하고 FFT로 푸리에 서술자를 구한 뒤,
저주파 K개 성분만 남겨 역FFT로 재구성한 스무딩된 윤곽선을 반환한다.
"""
function smooth_contour(boundary::Vector{Tuple{Int,Int}}, K::Int)
    z = ComplexF64[c + im * r for (r, c) in boundary]   # x=col, y=row
    Z = fft(z)
    Zf = low_pass_fd(Z, K)
    return ifft(Zf)
end

function draw_contour!(canvas::Matrix{RGB{N0f8}}, pts::Vector{ComplexF64}, color::RGB{N0f8})
    rows, cols = size(canvas)
    for p in pts
        c = round(Int, real(p))
        r = round(Int, imag(p))
        if 1 <= r <= rows && 1 <= c <= cols
            canvas[r, c] = color
        end
    end
end

function main()
    img = testimage("blobs")
    gray = Float64.(Gray.(img))

    # blobs 샘플은 어두운 반점들이 밝은 배경 위에 흩어져 있음 -> 개별 반점(전경)을
    # 서로 분리된 연결요소로 분리하기 위해 평균 밝기보다 어두운 픽셀을 전경으로 삼음
    thresh = sum(gray) / length(gray)
    mask = gray .< thresh

    blob = largest_component(mask)
    boundary = moore_boundary_trace(blob)
    println("추출된 최대 연결요소 경계 픽셀 수: ", length(boundary))

    z_raw = ComplexF64[c + im * r for (r, c) in boundary]

    Ks = [length(boundary), 25, 10, 4, 1]   # 첫 값(전체 고조파) = 스무딩 없는 원본 재구성
    labels = ["raw (K=all)", "K=25", "K=10", "K=4", "K=1"]

    base_rgb = RGB{N0f8}.(Gray.(gray))
    panels = Matrix{RGB{N0f8}}[]

    for K in Ks
        canvas = copy(base_rgb)
        smoothed = smooth_contour(boundary, min(K, length(boundary) ÷ 2))
        draw_contour!(canvas, smoothed, RGB{N0f8}(1.0, 0.0, 0.0))
        push!(panels, canvas)
    end

    combined = hcat(panels...)
    save("fd_smoothing_comparison.png", combined)
    save("fd_blob_mask.png", Gray.(blob))

    # 반점 주변을 확대한 비교본도 저장 (스무딩 효과를 더 뚜렷하게 확인하기 위함)
    rs = [p[1] for p in boundary]; cs = [p[2] for p in boundary]
    margin = 15
    r0, r1 = max(1, minimum(rs) - margin), min(size(gray, 1), maximum(rs) + margin)
    c0, c1 = max(1, minimum(cs) - margin), min(size(gray, 2), maximum(cs) + margin)
    zoom_panels = [imresize(p[r0:r1, c0:c1], ratio=4) for p in panels]
    save("fd_smoothing_comparison_zoom.png", hcat(zoom_panels...))

    println("완료: fd_blob_mask.png, fd_smoothing_comparison.png 저장됨")
    println("입력 크기: ", size(gray))
    println("패널(왼쪽부터): ", join(labels, ", "))
end

main()
