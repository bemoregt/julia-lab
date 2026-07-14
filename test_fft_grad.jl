using FFTW
using Zygote
using LinearAlgebra
using Random

Random.seed!(0)

# 1) basic fft/ifft roundtrip gradient sanity check
x = randn(Float64, 16, 16)
f(x) = sum(abs2, real.(ifft(fft(x))))
g = Zygote.gradient(f, x)[1]
println("basic fft roundtrip grad ≈ 2x ? ", isapprox(g, 2 .* x; atol=1e-8))

# 2) non-mutating truncated spectral multiply (core FNO op), multi-channel
function spectral_conv(x::AbstractArray{<:Real,3}, W::AbstractArray{<:Complex,4}, modes1::Int, modes2::Int)
    Cin, H, W2 = size(x)
    Cout = size(W, 1)
    X = fft(x, (2, 3))
    Xlow = X[:, 1:modes1, 1:modes2]                                   # (Cin, m1, m2)
    Wr = reshape(W, Cout, Cin, modes1, modes2)
    Xr = reshape(Xlow, 1, Cin, modes1, modes2)
    Ylow = dropdims(sum(Wr .* Xr, dims=2), dims=2)                     # (Cout, m1, m2)
    top = cat(Ylow, zeros(ComplexF64, Cout, modes1, W2 - modes2), dims=3)
    full = cat(top, zeros(ComplexF64, Cout, H - modes1, W2), dims=2)
    y = real.(ifft(full, (2, 3)))
    return y
end

Cin, Cout, H, W2, m1, m2 = 3, 4, 16, 16, 5, 5
x2 = randn(Float64, Cin, H, W2)
W = randn(ComplexF64, Cout, Cin, m1, m2)

y = spectral_conv(x2, W, m1, m2)
println("spectral_conv output size: ", size(y), " (expect ($Cout,$H,$W2))")

loss(x2, W) = sum(abs2, spectral_conv(x2, W, m1, m2))
gx, gW = Zygote.gradient(loss, x2, W)
println("grad wrt x: size=", size(gx), " any NaN=", any(isnan, gx))
println("grad wrt W: size=", size(gW), " any NaN=", any(x -> isnan(real(x)) || isnan(imag(x)), gW))

# finite-difference check on real input entries (avoids Wirtinger-derivative
# sign/convention ambiguity that comes with differentiating wrt complex W)
eps = 1e-6
for (i, j, k) in [(1, 1, 1), (2, 5, 9), (3, 16, 16)]
    xp = copy(x2); xp[i, j, k] += eps
    xm = copy(x2); xm[i, j, k] -= eps
    fd = (loss(xp, W) - loss(xm, W)) / (2eps)
    println("finite-diff dLoss/dx[$i,$j,$k] = ", fd, "   autodiff = ", gx[i, j, k],
            "   match=", isapprox(fd, gx[i, j, k]; rtol=1e-3))
end
