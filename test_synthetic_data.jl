include("src/SyntheticData.jl")
using .SyntheticData
using Images
using FileIO
using Random

mkpath("output")
rng = MersenneTwister(7)

for i in 1:3
    shape = sample_shape(rng)
    for N in (64, 128)
        img, mask = rasterize(shape, N; noise=0.03, rng=rng)
        row = hcat(Gray.(img), Gray.(mask))
        save("output/synthetic_sample_$(i)_res$(N).png", row)
    end
end
println("saved sample previews to output/synthetic_sample_*.png (image | mask)")
