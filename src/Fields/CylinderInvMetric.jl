@inline _cyl_inv_metric(r) = SymTensorValue{2,Float64,3}(1/r^2, 0.0, 1.0)

struct CylinderInvMetric <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(_, m::CylinderInvMetric, x::Point) = _cyl_inv_metric(m.radius)
function Gridap.Arrays.return_cache(m::CylinderInvMetric, xs::AbstractArray{<:Point})
  CachedArray(fill(_cyl_inv_metric(m.radius), size(xs)))
end
function Gridap.Arrays.evaluate!(cache, m::CylinderInvMetric, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  fill!(cache.array, _cyl_inv_metric(m.radius))
  cache.array
end

inverse_metric_field(f::CylinderMetric) = CylinderInvMetric(f.radius)
