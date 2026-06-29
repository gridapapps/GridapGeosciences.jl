# ── CylinderMetric ────────────────────────────────────────────────────────────
#
# Pullback metric g = diag(r², 1) — constant in chart coordinates.
# Inverse g⁻¹ = diag(1/r², 1).

@inline _cyl_metric(r) = SymTensorValue{2,Float64,3}(r^2, 0.0, 1.0)

struct CylinderMetric <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(_, m::CylinderMetric, x::Point) = _cyl_metric(m.radius)
function Gridap.Arrays.return_cache(m::CylinderMetric, xs::AbstractArray{<:Point})
  CachedArray(fill(_cyl_metric(m.radius), size(xs)))
end
function Gridap.Arrays.evaluate!(cache, m::CylinderMetric, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  fill!(cache.array, _cyl_metric(m.radius))
  cache.array
end
