include("src/FRFT.jl")
using .FRFT
using FFTW
using LinearAlgebra

N = 64
x = randn(ComplexF64, N)

println("== sanity checks for frft ==")

# a=0 must be identity
@assert norm(frft(x, 0.0) - x) < 1e-8
println("a=0 identity: OK")

# a=2 must be the parity operator x[n] -> x[(-n) mod N], i.e. F applied twice
# (NOT a plain reverse(); index 0 is fixed, the rest are mirrored)
parity(v) = [v[1]; reverse(v[2:end])]
@assert norm(frft(x, 2.0) - parity(x)) < 1e-8
println("a=2 parity (F^2): OK")

# a=1 must match ordinary (centered) DFT up to the algorithm's own convention
y1 = frft(x, 1.0)
println("a=1 energy ratio (‖F1 x‖ / ‖x‖): ", norm(y1) / norm(x))

# Energy (Parseval) should be preserved for any order since FRFT is unitary
for a in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.5, 3.2)
    y = frft(x, a)
    ratio = norm(y) / norm(x)
    println("a=$a energy ratio: ", ratio)
    @assert abs(ratio - 1.0) < 1e-8 "energy not preserved for a=$a (ratio=$ratio)"
end
println("Energy preservation across orders: OK (exact, unitary by construction)")

# Additivity / inverse property: F_{-a}(F_a(x)) ≈ x
for a in (0.4, 0.9, 1.5, 2.3)
    rec = frft(frft(x, a), -a)
    err = norm(rec - x) / norm(x)
    println("a=$a  inverse reconstruction relative error: ", err)
    @assert err < 1e-8 "inverse property failed for a=$a (err=$err)"
end
println("Inverse property F_{-a}(F_a(x)) ≈ x: OK (exact)")

# Additivity: F_a(F_b(x)) ≈ F_{a+b}(x)
for (a, b) in [(0.3, 0.4), (0.7, 0.9), (1.1, 0.6), (1.5, 1.5)]
    lhs = frft(frft(x, b), a)
    rhs = frft(x, a + b)
    err = norm(lhs - rhs) / norm(rhs)
    println("a=$a b=$b  additivity relative error: ", err)
    @assert err < 1e-8 "additivity failed for a=$a b=$b (err=$err)"
end
println("Additivity F_a(F_b(x)) ≈ F_{a+b}(x): OK (exact)")

println("\nAll 1D FRFT sanity checks passed.")
