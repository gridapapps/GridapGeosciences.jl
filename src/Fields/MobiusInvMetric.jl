@inline function _mobius_inv_metric(R, W, offset, x)
  θ   = π*(x[1] + offset)/2
  ρ   = R + W*x[2]*cos(θ/2)
  g11 = (π/4)^2 * W^2 * x[2]^2 + (π/2)^2 * ρ^2
  SymTensorValue{2,Float64,3}(1/g11, 0.0, 1/W^2)
end

struct MobiusInvMetric <: Field
  radius :: Float64; half_width :: Float64; theta_offset :: Float64
end
Gridap.Arrays.evaluate!(_, m::MobiusInvMetric, x::Point) =
  _mobius_inv_metric(m.radius, m.half_width, m.theta_offset, x)
function Gridap.Arrays.return_cache(_::MobiusInvMetric, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, SymTensorValue{2,Float64,3}))
end
function Gridap.Arrays.evaluate!(cache, m::MobiusInvMetric, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  R, W, offset = m.radius, m.half_width, m.theta_offset
  @inbounds for i in eachindex(xs)
    cache.array[i] = _mobius_inv_metric(R, W, offset, xs[i])
  end
  cache.array
end

inverse_metric_field(f::MobiusMetric) =
  MobiusInvMetric(f.radius, f.half_width, f.theta_offset)
