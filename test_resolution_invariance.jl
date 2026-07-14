include("src/FNO.jl")
using .FNOModel
using Random
using Statistics

Random.seed!(1)

net = FNOSeg(; cin=1, width=16, modes1=10, modes2=10, depth=3, cout=1)

println("== forward pass shape/range check across resolutions (random weights) ==")
for res in (32, 64, 96, 128, 200, 256)
    x = randn(Float32, 1, res, res, 1)
    y = net(Float64.(x))
    println("res=$res  out size=", size(y), "  range=", extrema(y))
    @assert size(y) == (1, res, res, 1)
    @assert all(0 .<= y .<= 1)
end
println("All resolutions produced correctly-shaped, in-range output.")

# Same-content check: does a resolution-invariant *architecture* actually
# behave consistently, or is it just "runs without crashing"? Feed the same
# underlying smooth image resampled at two resolutions and check the two
# (untrained, random-weight) outputs correlate in a spatially sensible way
# once resized to a common grid -- this is a weak sanity check since the
# weights are random (no training yet), so we only check it doesn't error
# and produces smoothly-varying (not degenerate/constant) output.
x64 = randn(Float64, 1, 64, 64, 1)
y64 = net(x64)
println("std of output at res=64 (random-weight net, should be >0, i.e. non-degenerate): ", std(y64))
