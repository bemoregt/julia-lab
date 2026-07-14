module FRFT

using LinearAlgebra

export frft, frft2d

# ---------------------------------------------------------------------------
# Discrete fractional Fourier transform via eigen-decomposition of the DFT
# matrix.
#
# The (normalized, unitary) DFT matrix F satisfies F^4 = I, so its
# eigenvalues are the 4th roots of unity exp(-i*pi*k/2), k = 0,1,2,3, each
# with high multiplicity (~N/4). Writing F = V * diag(lambda) * V' (V unitary
# since F is normal), a *naive* fractional power
#
#       F^a := V * diag(exp(-i*pi*a*k/2)) * V'        (BAD: only 4 distinct phases)
#
# is unitary and matches F exactly at a=0,1,2,3, but is a poor definition:
# every one of the ~N/4 eigenvectors sharing a class k is forced to rotate by
# the *exact same* phase for any fractional a, instead of smoothly fanning
# out. In practice this produces large-magnitude, spatially non-smooth
# artifacts (e.g. a spurious bright band near the image border) because
# entire (arbitrary, degenerate) eigenspaces move in lock-step rather than
# each mode picking up a phase proportional to its own "order".
#
# The standard fix (Pei & Yeh 1997; Candan, Kutay & Ozaktas 2000) assigns
# each of the N eigenvectors its own individual integer order
# n = 0, 1, ..., N-1 (n mod 4 == k, so integer-order behavior is unaffected)
# by ranking the eigenvectors *within* each class k by how "oscillatory"
# they are -- i.e. by how well they approximate the n-th continuous
# Hermite-Gauss function -- and uses
#
#       F^a := sum_n exp(-i*pi*a*n/2) * v_n v_n^H
#
# The ranking is obtained via the Dickinson & Steiglitz (1982) commuting
# matrix S below: S provably commutes with F (so it maps each F-eigenspace
# to itself) but -- unlike F -- has simple eigenvalues, and its eigenvalues
# decrease monotonically with increasing oscillation count. Diagonalizing S
# restricted to each F-eigenspace therefore recovers the smooth
# discrete-Hermite-Gauss ordering. This is verified in `test_frft.jl` (both
# via exact algebraic checks and via the border-artifact/energy-spread check
# in `main.jl`).
# ---------------------------------------------------------------------------

const _EIGENCACHE = Dict{Int,Tuple{Matrix{ComplexF64},Vector{Float64}}}()

# Dickinson & Steiglitz (1982) commuting matrix: a real symmetric,
# tridiagonal-with-corner (NOT circulant) matrix that provably commutes
# with the DFT matrix F, yet -- unlike F itself -- has (generically) simple
# eigenvalues. Restricting it to each of F's highly-degenerate eigenspaces
# and diagonalizing there picks out the smooth, Hermite-Gaussian-like
# "discrete Hermite" basis functions that the fractional Fourier transform
# literature (Pei & Yeh 1997; Candan, Kutay & Ozaktas 2000) uses, instead of
# an arbitrary (and often jagged/artifact-prone) orthonormal basis.
function _ds_matrix(N::Int)
    S = zeros(Float64, N, N)
    for n in 0:(N-1)
        S[n+1, n+1] = 2 * cos(2 * pi * n / N)
    end
    for n in 0:(N-2)
        S[n+1, n+2] = 1.0
        S[n+2, n+1] = 1.0
    end
    S[1, N] = 1.0
    S[N, 1] = 1.0
    return S
end

function _build_eigenbasis(N::Int)
    Fm = [exp(-2im * pi * k * n / N) for k in 0:(N-1), n in 0:(N-1)] ./ sqrt(N)
    ef = eigen(Fm)
    vals = ef.values
    vecs = ef.vectors

    # Assign each eigenvalue an exact order index k in {0,1,2,3} such that
    # lambda_k = exp(-i*pi*k/2); this is unambiguous since eigenvalues of F
    # can only be 4th roots of unity.
    ks = [mod(round(Int, -angle(v) * 2 / pi), 4) for v in vals]

    S = _ds_matrix(N)
    V = Matrix{ComplexF64}(undef, N, N)
    nsout = Vector{Float64}(undef, N)
    for k in 0:3
        idx = findall(==(k), ks)
        isempty(idx) && continue
        P = vecs[:, idx]
        Q0 = Matrix(qr(P).Q)                    # orthonormal basis of this F-eigenspace
        Srestr = Q0' * S * Q0                    # S restricted to the eigenspace (Hermitian)
        ew = eigen(Hermitian((Srestr + Srestr') / 2))   # ascending S-eigenvalues
        # Descending S-eigenvalue = increasing oscillation count = increasing
        # Hermite order l = 0, 1, ..., m-1 within this class; full order
        # n = k + 4l keeps n mod 4 == k (exact integer-order behavior) while
        # giving every eigenvector in the class its own distinct phase slope.
        m = length(idx)
        rev = m:-1:1
        Qg = Q0 * ew.vectors[:, rev]
        V[:, idx] = Qg
        nsout[idx] = [k + 4 * (l - 1) for l in 1:m]
    end

    return V, nsout
end

function _eigenbasis(N::Int)
    get!(_EIGENCACHE, N) do
        _build_eigenbasis(N)
    end
end

"""
    frft_matrix(N, a)

Return the `N x N` matrix implementing the discrete fractional Fourier
transform of order `a` (4-periodic).
"""
function frft_matrix(N::Int, a::Real)
    V, ns = _eigenbasis(N)
    a = mod(Float64(a), 4.0)
    D = exp.((-1im * pi / 2 * a) .* ns)
    return V * Diagonal(D) * V'
end

"""
    frft(f, a)

Discrete fractional Fourier transform (DFRFT) of order `a` (4-periodic:
`a=0` identity, `a=1` ordinary unitary DFT, `a=2` time reversal, `a=3`
inverse DFT). Unitary for every real `a`.
"""
function frft(f::AbstractVector{<:Number}, a::Real)
    N = length(f)
    Fa = frft_matrix(N, a)
    return Fa * ComplexF64.(f)
end

"""
    frft2d(img, a, b=a)

Separable 2D fractional Fourier transform: order `a` applied along each row,
order `b` applied along each column.
"""
function frft2d(img::AbstractMatrix{<:Number}, a::Real, b::Real=a)
    rows, cols = size(img)
    Fa = frft_matrix(cols, a)
    Fb = frft_matrix(rows, b)
    tmp = ComplexF64.(img) * transpose(Fa)
    return Fb * tmp
end

end # module
