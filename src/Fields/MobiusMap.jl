# ── MobiusMap ─────────────────────────────────────────────────────────────────

@inline function _mobius_map(R, W, offset, x)
  θ = π*(x[1] + offset)/2
  ρ = R + W*x[2]*cos(θ/2)
  Point(ρ*cos(θ), ρ*sin(θ), W*x[2]*sin(θ/2))
end
@inline function _mobius_jac(R, W, offset, x)
  θ    = π*(x[1] + offset)/2
  t    = x[2]
  ρ    = R + W*t*cos(θ/2)
  dρds = -W*t*sin(θ/2)*(π/4)
  dρdt =  W*cos(θ/2)
  TensorValue{2,3,Float64}(
    dρds*cos(θ) - ρ*sin(θ)*(π/2),  dρdt*cos(θ),
    dρds*sin(θ) + ρ*cos(θ)*(π/2),  dρdt*sin(θ),
    W*t*cos(θ/2)*(π/4),             W*sin(θ/2),
  )
end

struct MobiusMap <: Field
  radius :: Float64; half_width :: Float64; theta_offset :: Float64
end
Gridap.Arrays.evaluate!(_, m::MobiusMap, x::Point) =
  _mobius_map(m.radius, m.half_width, m.theta_offset, x)
function Gridap.Arrays.return_cache(_::MobiusMap, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, Point{3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::MobiusMap, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  R, W, offset = m.radius, m.half_width, m.theta_offset
  @inbounds for i in eachindex(xs)
    cache.array[i] = _mobius_map(R, W, offset, xs[i])
  end
  cache.array
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:MobiusMap}, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, TensorValue{2,3,Float64}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:MobiusMap}, x::Point) =
  _mobius_jac(f.object.radius, f.object.half_width, f.object.theta_offset, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:MobiusMap}, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  R, W, offset = f.object.radius, f.object.half_width, f.object.theta_offset
  @inbounds for i in eachindex(xs)
    cache.array[i] = _mobius_jac(R, W, offset, xs[i])
  end
  cache.array
end
