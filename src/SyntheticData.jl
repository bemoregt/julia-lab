module SyntheticData

using Random

export BlobShape, sample_shape, rasterize

"""
    BlobShape

A smooth, star-shaped 2D blob defined as a *continuous* function of (x,y) in
[0,1]^2 (radius as a truncated Fourier series in angle theta). Because the
shape is defined continuously rather than as a fixed-resolution bitmap, it
can be rasterized at *any* resolution with a consistent, well-defined
ground-truth boundary -- exactly what's needed to test whether a
segmentation model trained at one resolution generalizes to others, rather
than trivially testing on up/down-sampled copies of a single raster.
"""
struct BlobShape
    cx::Float64
    cy::Float64
    r0::Float64
    amps::Vector{Float64}
    phases::Vector{Float64}
    intensity_in::Float64
    intensity_out::Float64
end

function radius_at(shape::BlobShape, theta::Real)
    r = shape.r0
    for (k, (a, p)) in enumerate(zip(shape.amps, shape.phases))
        r += a * cos(k * theta + p)
    end
    return r
end

function sample_shape(rng::AbstractRNG; K::Int=4)
    cx = 0.35 + 0.30 * rand(rng)
    cy = 0.35 + 0.30 * rand(rng)
    r0 = 0.12 + 0.10 * rand(rng)
    amps = [r0 * 0.18 * rand(rng) / k for k in 1:K]     # decreasing amplitude -> smooth boundary
    phases = 2pi .* rand(rng, K)
    intensity_in = 0.60 + 0.25 * rand(rng)
    intensity_out = 0.20 + 0.15 * rand(rng)
    return BlobShape(cx, cy, r0, amps, phases, intensity_in, intensity_out)
end

"""
    rasterize(shape, N; noise=0.03, rng=Random.default_rng())

Render `shape` on an `N x N` grid (pixel centers at `(i-0.5)/N`). Returns
`(image, mask)`, both `N x N` `Float64` in `[0,1]`.
"""
function rasterize(shape::BlobShape, N::Int; noise::Real=0.03,
                    rng::AbstractRNG=Random.default_rng())
    img = Array{Float64}(undef, N, N)
    mask = Array{Float64}(undef, N, N)
    for iy in 1:N, ix in 1:N
        x = (ix - 0.5) / N
        y = (iy - 0.5) / N
        dx = x - shape.cx
        dy = y - shape.cy
        theta = atan(dy, dx)
        r = sqrt(dx^2 + dy^2)
        rb = radius_at(shape, theta)
        inside = r <= rb
        bg = 0.15 + 0.05 * sin(3 * x) * cos(3 * y)   # smooth background shading
        img[iy, ix] = inside ? shape.intensity_in : (shape.intensity_out + bg)
        mask[iy, ix] = inside ? 1.0 : 0.0
    end
    img = clamp.(img .+ noise .* randn(rng, N, N), 0.0, 1.0)
    return img, mask
end

end # module
