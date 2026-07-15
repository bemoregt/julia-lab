using Images
using TestImages
using ImageFiltering
using LinearAlgebra
using Random
using Statistics: mean

# =============================================================
# 쿼터니언 <-> 회전행렬 변환
# =============================================================

function quat_to_rotmat(q::Vector{Float64})
    w, x, y, z = q
    return [1-2(y^2+z^2)   2(x*y-w*z)    2(x*z+w*y);
            2(x*y+w*z)     1-2(x^2+z^2)  2(y*z-w*x);
            2(x*z-w*y)     2(y*z+w*x)    1-2(x^2+y^2)]
end

function rotmat_to_quat(R::Matrix{Float64})
    tr = R[1,1] + R[2,2] + R[3,3]
    w = sqrt(max(0.0, 1+tr)) / 2
    x = sqrt(max(0.0, 1+R[1,1]-R[2,2]-R[3,3])) / 2 * sign(R[3,2]-R[2,3] + 1e-12)
    y = sqrt(max(0.0, 1-R[1,1]+R[2,2]-R[3,3])) / 2 * sign(R[1,3]-R[3,1] + 1e-12)
    z = sqrt(max(0.0, 1-R[1,1]-R[2,2]+R[3,3])) / 2 * sign(R[2,1]-R[1,2] + 1e-12)
    q = [w,x,y,z]
    return q ./ norm(q)
end

function axis_angle_to_rotmat(axis::Vector{Float64}, angle::Float64)
    a = axis ./ norm(axis)
    K = [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]
    return I(3) + sin(angle) .* K + (1-cos(angle)) .* (K*K)
end

quat_angle_diff_deg(q1::Vector{Float64}, q2::Vector{Float64}) =
    rad2deg(2*acos(clamp(abs(dot(q1,q2)), -1.0, 1.0)))

vec_angle_deg(a::Vector{Float64}, b::Vector{Float64}) =
    rad2deg(acos(clamp(dot(a,b)/(norm(a)*norm(b)), -1.0, 1.0)))

# =============================================================
# Harris 코너 검출 (매칭 특징점 후보 추출용, 외부 의존성 없이 직접 구현)
# =============================================================

function harris_response(img::Matrix{Float64}; sigma::Float64=1.5, k::Float64=0.04)
    Ix = zeros(size(img)); Iy = zeros(size(img))
    Ix[:, 2:end-1] .= (img[:, 3:end] .- img[:, 1:end-2]) ./ 2
    Iy[2:end-1, :] .= (img[3:end, :] .- img[1:end-2, :]) ./ 2
    Sxx = imfilter(Ix .^ 2, Kernel.gaussian(sigma))
    Syy = imfilter(Iy .^ 2, Kernel.gaussian(sigma))
    Sxy = imfilter(Ix .* Iy, Kernel.gaussian(sigma))
    detM = Sxx .* Syy .- Sxy .^ 2
    trM = Sxx .+ Syy
    return detM .- k .* trM .^ 2
end

function top_corners(R::Matrix{Float64}, nfeat::Int; min_dist::Float64=20.0, border::Int=20)
    rows, cols = size(R)
    cand = Tuple{Float64,Int,Int}[]
    for r in (1+border):(rows-border), c in (1+border):(cols-border)
        push!(cand, (R[r, c], r, c))
    end
    sort!(cand, by=x -> -x[1])
    selected = Tuple{Int,Int}[]
    for (val, r, c) in cand
        val <= 0 && break
        if all(p -> (p[1] - r)^2 + (p[2] - c)^2 > min_dist^2, selected)
            push!(selected, (r, c))
        end
        length(selected) >= nfeat && break
    end
    return selected
end

# =============================================================
# 카메라 모델 및 합성 대응점 생성
# =============================================================

struct Intrinsics
    fx::Float64; fy::Float64; cx::Float64; cy::Float64
end

pixel_to_normalized(pt::Tuple{Int,Int}, K::Intrinsics) =
    [(pt[2] - K.cx) / K.fx, (pt[1] - K.cy) / K.fy, 1.0]   # pt = (row, col)

"""
project_points(P3d, R, t, K) -> Vector{Vector{Float64}}

3차원 점들을 (R,t)로 정의된 카메라 좌표계로 옮긴 뒤, 정규화된 이미지
평면 좌표 [x/z, y/z, 1]로 투영한다. 카메라1 기준 표현이 P1이면
P2 = R*P1 + t (강체운동 제약 u*R*m + t = v*n 과 동일한 관례).
"""
function project_points(P3d::Vector{Vector{Float64}}, R::Matrix{Float64}, t::Vector{Float64})
    out = Vector{Vector{Float64}}()
    for P in P3d
        Pc = R * P .+ t
        push!(out, [Pc[1]/Pc[3], Pc[2]/Pc[3], 1.0])
    end
    return out
end

# =============================================================
# 정규화 8점 알고리즘으로 Essential Matrix 추정
# (QuEst 논문의 최소해 다항식-고유값 풀이 대신, 다수 대응점에 대해
#  수학적으로 동등한 목적 -- 매칭점으로부터 상대 회전/이동 복원 --
#  을 갖는 표준적이고 검증 가능한 방법을 사용함)
# =============================================================

function essential_matrix_8point(m::Vector{Vector{Float64}}, n::Vector{Vector{Float64}})
    N = length(m)
    A = zeros(N, 9)
    for i in 1:N
        x1, y1 = m[i][1], m[i][2]
        x2, y2 = n[i][1], n[i][2]
        A[i, :] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1.0]
    end
    F = svd(A)
    e = F.V[:, end]
    E = Matrix(reshape(e, 3, 3)')   # 행 우선(row-major) 순서로 채워지도록 전치
    Fe = svd(E)
    return Fe.U * Diagonal([1.0, 1.0, 0.0]) * Fe.Vt
end

"""
decompose_essential(E) -> Vector{Tuple{Matrix,Vector}}

Essential Matrix를 4개의 (R,t) 후보로 분해한다 (Hartley & Zisserman
표준 분해). t는 스케일까지만 복원 가능하므로 단위벡터로 반환한다.
"""
function decompose_essential(E::Matrix{Float64})
    F = svd(E)
    U, Vt = F.U, F.Vt
    if det(U) < 0; U = -U; end
    if det(Vt') < 0; Vt = -Vt; end
    W = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
    R1 = U * W * Vt
    R2 = U * W' * Vt
    t = U[:, 3] ./ norm(U[:, 3])
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]
end

function triangulate_depth(R::Matrix{Float64}, t::Vector{Float64}, m::Vector{Float64}, n::Vector{Float64})
    A = hcat(R * m, -n)
    uv = A \ (-t)
    return uv[1], uv[2]   # u (카메라1 깊이), v (카메라2 깊이)
end

"""
select_correct_pose(candidates, m, n) -> (R,t)

4개의 (R,t) 후보 중, 매칭점들이 두 카메라 모두에서 양의 깊이(카메라
앞쪽)를 갖는 후보를 다수결로 선택한다 (카이랄리티/cheirality 제약).
"""
function select_correct_pose(candidates, m::Vector{Vector{Float64}}, n::Vector{Vector{Float64}})
    best_score, best = -1, candidates[1]
    for (R, t) in candidates
        score = 0
        for i in 1:length(m)
            u, v = triangulate_depth(R, t, m[i], n[i])
            (u > 0 && v > 0) && (score += 1)
        end
        if score > best_score
            best_score, best = score, (R, t)
        end
    end
    return best, best_score
end

"""
estimate_pose(m, n) -> (R, t, quat)

매칭 대응점 (m,n)으로부터 상대 포즈를 추정하는 전체 파이프라인:
정규화 8점 알고리즘으로 E 추정 -> 4개 (R,t) 후보로 분해 -> 카이랄리티
제약으로 올바른 해 선택 -> 회전을 쿼터니언으로 변환.
"""
function estimate_pose(m::Vector{Vector{Float64}}, n::Vector{Vector{Float64}})
    E = essential_matrix_8point(m, n)
    candidates = decompose_essential(E)
    (R, t), score = select_correct_pose(candidates, m, n)
    q = rotmat_to_quat(R)
    return R, t, q, score
end

# =============================================================
# 1) 순수 합성 데이터로 파이프라인 정확성 검증 (sanity check)
# =============================================================

function sanity_check_synthetic()
    rng = Xoshiro(0)
    N = 60
    P3d = [ [randn(rng)*2, randn(rng)*2, 5.0 + rand(rng)*5.0] for _ in 1:N ]

    R_true = axis_angle_to_rotmat([0.2, 1.0, 0.3], deg2rad(12.0))
    t_true = [0.35, 0.05, 0.10]

    m = [ [P[1]/P[3], P[2]/P[3], 1.0] for P in P3d ]
    n = project_points(P3d, R_true, t_true)

    R_est, t_est, q_est, score = estimate_pose(m, n)
    q_true = rotmat_to_quat(R_true)

    rot_err = quat_angle_diff_deg(q_true, q_est)
    t_dir_err = min(vec_angle_deg(t_true, t_est), vec_angle_deg(t_true, -t_est))

    println("=== 1) 순수 합성 데이터 파이프라인 검증 ===")
    println("카이랄리티 통과 점 개수: $score / $N")
    println("회전 오차(쿼터니언 각도): ", round(rot_err, digits=4), " deg")
    println("이동 방향 오차: ", round(t_dir_err, digits=4), " deg  (스케일은 원리상 복원 불가)")
    return rot_err, t_dir_err
end

# =============================================================
# 2) 실제 이미지의 Harris 코너를 특징점으로 사용한 테스트
#    (2D 코너 위치는 실제 이미지에서, 3차원 깊이는 합성으로 부여해
#     정답(ground-truth) 포즈와 비교 가능하게 구성)
# =============================================================

function real_image_test()
    img = testimage("lighthouse")
    gray = Float64.(Gray.(img))
    rows, cols = size(gray)
    println()
    println("=== 2) 실제 이미지(Harris 코너) 기반 포즈 추정 테스트 ===")
    println("이미지 크기: ", size(gray))

    Rresp = harris_response(gray)
    corners = top_corners(Rresp, 40; min_dist=25.0, border=30)
    println("검출된 코너 개수: ", length(corners))

    K = Intrinsics(cols, cols, cols/2, rows/2)   # focal length ~ 이미지 너비, 주점은 중앙

    rng = Xoshiro(7)
    depths = [4.0 + rand(rng)*6.0 for _ in corners]   # 코너별 임의(합성) 깊이 부여
    P3d = [ depths[i] .* pixel_to_normalized(corners[i], K) for i in 1:length(corners) ]

    R_true = axis_angle_to_rotmat([0.1, 1.0, 0.15], deg2rad(9.0))
    t_true = [0.4, -0.05, 0.15]

    m = [ [P[1]/P[3], P[2]/P[3], 1.0] for P in P3d ]
    n = project_points(P3d, R_true, t_true)

    R_est, t_est, q_est, score = estimate_pose(m, n)
    q_true = rotmat_to_quat(R_true)

    rot_err = quat_angle_diff_deg(q_true, q_est)
    t_dir_err = min(vec_angle_deg(t_true, t_est), vec_angle_deg(t_true, -t_est))

    println("--- 잡음 없음 ---")
    println("카이랄리티 통과 점 개수: $score / $(length(corners))")
    println("정답 쿼터니언      : ", round.(q_true, digits=4))
    println("추정 쿼터니언      : ", round.(q_est, digits=4))
    println("회전 오차(각도)     : ", round(rot_err, digits=4), " deg")
    println("이동 방향 오차(각도): ", round(t_dir_err, digits=4), " deg")

    # 논문의 잡음 강건성 실험과 유사하게, 픽셀 단위 가우시안 잡음을 매칭점에
    # 추가해 성능 저하 양상을 확인 (0.5~2.0 픽셀 표준편차)
    println()
    println("--- 픽셀 노이즈에 따른 오차 변화 ---")
    rng2 = Xoshiro(99)
    for noise_px in (0.5, 1.0, 2.0)
        n_noisy = [ [nn[1] + randn(rng2)*noise_px/K.fx,
                     nn[2] + randn(rng2)*noise_px/K.fy, 1.0] for nn in n ]
        R_n, t_n, q_n, sc_n = estimate_pose(m, n_noisy)
        re = quat_angle_diff_deg(q_true, q_n)
        te = min(vec_angle_deg(t_true, t_n), vec_angle_deg(t_true, -t_n))
        println("  노이즈 σ=$(noise_px)px  ->  회전오차=$(round(re,digits=3))°, 이동방향오차=$(round(te,digits=3))°, 카이랄리티=$sc_n/$(length(corners))")
    end

    vis = RGB{N0f8}.(Gray.(gray))
    for (r, c) in corners
        for dr in -3:3, dc in -3:3
            (dr^2 + dc^2 <= 9) || continue
            rr, cc = r + dr, c + dc
            if 1 <= rr <= rows && 1 <= cc <= cols
                vis[rr, cc] = RGB{N0f8}(1, 0, 0)
            end
        end
    end
    save("quest_corners.png", vis)
    println("완료: quest_corners.png 저장됨 (빨간 점 = 사용된 특징점)")
end

# =============================================================
# 3) 진짜 "두 번째 이미지"를 실제로 생성 (호모그래피 워핑) +
#    양쪽 이미지에서 각각 독립적으로 특징 검출 및 패치 유사도 매칭 +
#    RANSAC 기반 강건 포즈 추정
# =============================================================

"""
to_pixel_homography(R, t, K, plane_depth) -> Matrix{Float64}

장면을 카메라1 기준 광축에 수직인 평면(깊이=plane_depth)으로 근사할 때,
(R,t)로 정의된 두 시점 사이의 픽셀 좌표 호모그래피를 계산한다.
    H3 = R + t*nᵀ/d   (정규화 좌표 간 매핑, n=[0,0,1], d=plane_depth)
    H_pixel = K * H3 * K⁻¹
"""
function to_pixel_homography(R::Matrix{Float64}, t::Vector{Float64}, K::Intrinsics, plane_depth::Float64)
    Kmat = [K.fx 0.0 K.cx; 0.0 K.fy K.cy; 0.0 0.0 1.0]
    n = [0.0, 0.0, 1.0]
    H3 = R .+ (t * n') ./ plane_depth
    return Kmat * H3 * inv(Kmat)
end

function bilinear_sample(img::Matrix{Float64}, y::Float64, x::Float64)
    rows, cols = size(img)
    (x < 1 || x > cols || y < 1 || y > rows) && return 0.0
    x0, y0 = floor(Int, x), floor(Int, y)
    x1, y1 = min(x0 + 1, cols), min(y0 + 1, rows)
    dx, dy = x - x0, y - y0
    v00, v01 = img[y0, x0], img[y0, x1]
    v10, v11 = img[y1, x0], img[y1, x1]
    return (1 - dy) * ((1 - dx) * v00 + dx * v01) + dy * ((1 - dx) * v10 + dx * v11)
end

"""
warp_image(img, Hpix) -> Matrix{Float64}

주어진 픽셀 호모그래피로 실제 이미지 전체를 역방향 매핑(inverse warping)
+ 쌍선형 보간하여 진짜 "두 번째 시점" 이미지를 렌더링한다.
"""
function warp_image(img::Matrix{Float64}, Hpix::Matrix{Float64})
    rows, cols = size(img)
    Hinv = inv(Hpix)
    out = zeros(rows, cols)
    for r in 1:rows, c in 1:cols
        p1 = Hinv * [Float64(c), Float64(r), 1.0]
        p1 ./= p1[3]
        out[r, c] = bilinear_sample(img, p1[2], p1[1])
    end
    return out
end

function extract_patch(img::Matrix{Float64}, r::Int, c::Int, half::Int)
    rows, cols = size(img)
    (r - half < 1 || r + half > rows || c - half < 1 || c + half > cols) && return nothing
    return img[r-half:r+half, c-half:c+half]
end

function ncc(a::Matrix{Float64}, b::Matrix{Float64})
    am = a .- mean(a); bm = b .- mean(b)
    return sum(am .* bm) / (sqrt(sum(am .^ 2) * sum(bm .^ 2)) + 1e-8)
end

"""
match_features(img1, corners1, img2, corners2; half, min_score) -> Vector{Tuple{Int,Int,Float64}}

두 이미지에서 각각 독립적으로 검출된 코너들을, 주변 패치의 정규화 상호
상관(NCC)으로 비교해 매칭한다. 상호 최선(mutual best match)인 쌍만
남겨 오매칭을 줄인다. 인덱스를 미리 알고 있는 것이 아니라, 순전히
이미지 외관(패치)만으로 대응점을 찾는다.
"""
function match_features(img1::Matrix{Float64}, corners1, img2::Matrix{Float64}, corners2;
                         half::Int=7, min_score::Float64=0.4)
    p1s = [extract_patch(img1, r, c, half) for (r, c) in corners1]
    p2s = [extract_patch(img2, r, c, half) for (r, c) in corners2]

    best_for_1 = fill((0, -Inf), length(corners1))
    for i in 1:length(corners1)
        p1s[i] === nothing && continue
        for j in 1:length(corners2)
            p2s[j] === nothing && continue
            s = ncc(p1s[i], p2s[j])
            if s > best_for_1[i][2]
                best_for_1[i] = (j, s)
            end
        end
    end

    best_for_2 = fill((0, -Inf), length(corners2))
    for j in 1:length(corners2)
        p2s[j] === nothing && continue
        for i in 1:length(corners1)
            p1s[i] === nothing && continue
            s = ncc(p1s[i], p2s[j])
            if s > best_for_2[j][2]
                best_for_2[j] = (i, s)
            end
        end
    end

    matches = Tuple{Int,Int,Float64}[]
    for i in 1:length(corners1)
        j, s = best_for_1[i]
        j == 0 && continue
        s < min_score && continue
        if best_for_2[j][1] == i
            push!(matches, (i, j, s))
        end
    end
    return matches
end

function epipolar_residual(E::Matrix{Float64}, m::Vector{Float64}, n::Vector{Float64})
    return abs(dot(n, E * m))
end

"""
estimate_pose_ransac(m, n; iters, thresh) -> (R, t, quat, inliers)

정규화 8점 알고리즘을 RANSAC으로 감싸, 특징점 매칭 과정에서 생긴
이상치(mismatch)에 강건하게 포즈를 추정한다.
"""
function estimate_pose_ransac(m::Vector{Vector{Float64}}, n::Vector{Vector{Float64}};
                               iters::Int=1000, thresh::Float64=0.01, rng::AbstractRNG=Random.default_rng())
    N = length(m)
    best_inliers = Int[]
    for _ in 1:iters
        idx = randperm(rng, N)[1:8]
        local E
        try
            E = essential_matrix_8point(m[idx], n[idx])
        catch
            continue
        end
        inliers = [i for i in 1:N if epipolar_residual(E, m[i], n[i]) < thresh]
        if length(inliers) > length(best_inliers)
            best_inliers = inliers
        end
    end
    E_final = essential_matrix_8point(m[best_inliers], n[best_inliers])
    candidates = decompose_essential(E_final)
    (R, t), score = select_correct_pose(candidates, m[best_inliers], n[best_inliers])
    q = rotmat_to_quat(R)
    return R, t, q, best_inliers
end

function draw_line_px!(img::Matrix{RGB{N0f8}}, r0::Int, c0::Int, r1::Int, c1::Int, color::RGB{N0f8})
    rows, cols = size(img)
    dx = abs(c1 - c0); sx = c0 < c1 ? 1 : -1
    dy = -abs(r1 - r0); sy = r0 < r1 ? 1 : -1
    err = dx + dy
    x, y = c0, r0
    while true
        if 1 <= y <= rows && 1 <= x <= cols
            img[y, x] = color
        end
        (x == c1 && y == r1) && break
        e2 = 2 * err
        if e2 >= dy; err += dy; x += sx; end
        if e2 <= dx; err += dx; y += sy; end
    end
end

function real_two_view_test()
    img = testimage("lighthouse")
    gray = Float64.(Gray.(img))
    rows, cols = size(gray)
    println()
    println("=== 3) 실제로 워핑해 만든 두 번째 이미지 + 독립 특징검출/매칭 + RANSAC 포즈 추정 ===")
    println("이미지 크기: ", size(gray))

    K = Intrinsics(cols, cols, cols / 2, rows / 2)
    R_true = axis_angle_to_rotmat([0.1, 1.0, 0.15], deg2rad(9.0))
    t_true = [0.4, -0.05, 0.15]
    plane_depth = 6.0

    Hpix = to_pixel_homography(R_true, t_true, K, plane_depth)
    gray2 = warp_image(gray, Hpix)
    save("quest_view1.png", Gray.(clamp01.(gray)))
    save("quest_view2_warped.png", Gray.(clamp01.(gray2)))

    resp1 = harris_response(gray)
    resp2 = harris_response(gray2)
    corners1 = top_corners(resp1, 60; min_dist=20.0, border=30)
    corners2 = top_corners(resp2, 60; min_dist=20.0, border=30)
    println("이미지1 코너: $(length(corners1))개   이미지2 코너: $(length(corners2))개 (각각 독립적으로 검출)")

    matches = match_features(gray, corners1, gray2, corners2; half=7, min_score=0.4)
    println("패치 유사도(NCC) 기반 매칭 쌍: $(length(matches))개")

    m = [pixel_to_normalized(corners1[i], K) for (i, j, s) in matches]
    n = [pixel_to_normalized(corners2[j], K) for (i, j, s) in matches]

    rng = Xoshiro(3)
    R_est, t_est, q_est, inliers = estimate_pose_ransac(m, n; iters=2000, thresh=0.01, rng=rng)
    q_true = rotmat_to_quat(R_true)

    rot_err = quat_angle_diff_deg(q_true, q_est)
    t_dir_err = min(vec_angle_deg(t_true, t_est), vec_angle_deg(t_true, -t_est))

    println("RANSAC 인라이어: $(length(inliers)) / $(length(matches))")
    println("정답 쿼터니언  : ", round.(q_true, digits=4))
    println("추정 쿼터니언  : ", round.(q_est, digits=4))
    println("회전 오차(각도): ", round(rot_err, digits=4))
    println("이동 방향 오차(각도): ", round(t_dir_err, digits=4))

    inlier_set = Set(inliers)
    vis = hcat(RGB{N0f8}.(Gray.(gray)), RGB{N0f8}.(Gray.(gray2)))
    for (idx, (i, j, s)) in enumerate(matches)
        r1, c1 = corners1[i]
        r2, c2 = corners2[j]
        color = idx in inlier_set ? RGB{N0f8}(0, 1, 0) : RGB{N0f8}(1, 0, 0)
        draw_line_px!(vis, r1, c1, r2, c2 + cols, color)
    end
    save("quest_matches.png", vis)
    println("완료: quest_view1.png, quest_view2_warped.png, quest_matches.png 저장됨")
    println("(quest_matches.png: 초록선=RANSAC 인라이어 매칭, 빨간선=이상치로 걸러진 매칭)")
end

# =============================================================
# 4) 사용자가 제공한 실제 사진 두 장으로 테스트 (정답 포즈를 모르는
#    진짜 실전 시나리오 -- 두 사진 모두에서 독립적으로 특징 검출 후
#    패치 유사도로 매칭, RANSAC으로 강건하게 상대 포즈 추정)
# =============================================================

function real_photo_pair_test(path1::String, path2::String)
    println()
    println("=== 4) 실제 사진 두 장으로 상대 포즈 추정 (정답 포즈 없음) ===")

    img1_raw = Float64.(Gray.(load(path1)))
    img2_raw = Float64.(Gray.(load(path2)))
    println("원본 크기 - 사진1: ", size(img1_raw), "   사진2: ", size(img2_raw))

    rows, cols = size(img1_raw)
    img2 = imresize(img2_raw, (rows, cols))   # 두 사진 크기를 통일

    K = Intrinsics(cols, cols, cols / 2, rows / 2)

    resp1 = harris_response(img1_raw)
    resp2 = harris_response(img2)
    corners1 = top_corners(resp1, 80; min_dist=18.0, border=25)
    corners2 = top_corners(resp2, 80; min_dist=18.0, border=25)
    println("사진1 코너: $(length(corners1))개   사진2 코너: $(length(corners2))개 (각각 독립 검출)")

    matches = match_features(img1_raw, corners1, img2, corners2; half=7, min_score=0.35)
    println("패치 유사도(NCC) 기반 매칭 쌍: $(length(matches))개")

    if length(matches) < 8
        println("매칭 쌍이 8개 미만이라 포즈 추정이 불가능합니다 (essential matrix 최소 요구치).")
        return
    end

    m = [pixel_to_normalized(corners1[i], K) for (i, j, s) in matches]
    n = [pixel_to_normalized(corners2[j], K) for (i, j, s) in matches]

    rng = Xoshiro(5)
    R_est, t_est, q_est, inliers = estimate_pose_ransac(m, n; iters=3000, thresh=0.01, rng=rng)

    axis, angle = quat_to_axis_angle(q_est)
    println("RANSAC 인라이어: $(length(inliers)) / $(length(matches))")
    println("추정 쿼터니언       : ", round.(q_est, digits=4))
    println("추정 회전축         : ", round.(axis, digits=3))
    println("추정 회전각         : ", round(rad2deg(angle), digits=3), " deg")
    println("추정 이동 방향(단위벡터): ", round.(t_est, digits=3))
    println("(주의: 두 사진의 실제 정답 포즈를 모르므로 오차 계산은 불가 -- 추정치만 보고)")

    inlier_set = Set(inliers)
    vis = hcat(RGB{N0f8}.(Gray.(img1_raw)), RGB{N0f8}.(Gray.(img2)))
    for (idx, (i, j, s)) in enumerate(matches)
        r1, c1 = corners1[i]
        r2, c2 = corners2[j]
        color = idx in inlier_set ? RGB{N0f8}(0, 1, 0) : RGB{N0f8}(1, 0, 0)
        draw_line_px!(vis, r1, c1, r2, c2 + cols, color)
    end
    save("quest_realphoto_matches.png", vis)
    println("완료: quest_realphoto_matches.png 저장됨")
end

function quat_to_axis_angle(q::Vector{Float64})
    w = clamp(q[1], -1.0, 1.0)
    angle = 2 * acos(w)
    s = sqrt(max(1 - w^2, 1e-12))
    axis = q[2:4] ./ s
    return axis, angle
end

sanity_check_synthetic()
real_image_test()
real_two_view_test()
real_photo_pair_test("/Users/air/Desktop/ScrShot 2.png", "/Users/air/Desktop/ScrShot 3.png")
