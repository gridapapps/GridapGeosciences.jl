# ── CylinderMap ──────────────────────────────────────────────────────────────
#
# Chart map (θ,z) → (r·cosθ, r·sinθ, z) and its explicit Jacobian.
# Gradient is exposed via FieldGradient{1,<:CylinderMap} (return_cache/evaluate!).
# Array evaluate! methods are required: lazy_map(∘, cell_ambient_maps, chart_maps)
# decomposes into lazy_map(evaluate, cell_ambient_maps, chart_coord_arrays) during
# FE assembly (ApplyOptimizations.jl), so the array path is in the hot path.

@inline _cyl_map(r, x) = Point(r*cos(x[1]), r*sin(x[1]), x[2])
@inline _cyl_jac(r, x) = TensorValue{2,3,Float64}(-r*sin(x[1]), 0.0, r*cos(x[1]), 0.0, 0.0, 1.0)

struct CylinderMap <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(_, m::CylinderMap, x::Point) = _cyl_map(m.radius, x)
function Gridap.Arrays.return_cache(_::CylinderMap, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, Point{3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CylinderMap, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  r = m.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _cyl_map(r, xs[i])
  end
  cache.array
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CylinderMap}, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, TensorValue{2,3,Float64}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CylinderMap}, x::Point) = _cyl_jac(f.object.radius, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CylinderMap}, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  r = f.object.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _cyl_jac(r, xs[i])
  end
  cache.array
end
