function _csphere_inv_metric(r::Float64, x::Point{2,T}) where T
  a, b  = tan(x[1]), tan(x[2])
  sa, sb = 1 + a^2, 1 + b^2
  ρ2    = 1 + a^2 + b^2
  c     = ρ2 / (r^2 * sa * sb)
  SymTensorValue{2,T,3}(c*sb, c*a*b, c*sa)
end

struct CubedSphereInvMetric <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CubedSphereInvMetric, x::Point{2}) =
  _csphere_inv_metric(m.radius, x)
function Gridap.Arrays.return_cache(m::CubedSphereInvMetric, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, SymTensorValue{2,Float64,3}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereInvMetric, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  r = m.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_inv_metric(r, xs[i])
  end
  cache.array
end

# ── FieldGradient of CubedSphereInvMetric ─────────────────────────────────────
#
# Returns ThirdOrderTensorValue{2,2,2} where T[k,i,j] = ∂h_{ij}/∂x_k,
# h = g⁻¹.  With a=tanα, b=tanβ, sa=1+a², sb=1+b², ρ²=1+a²+b², ci=1/(r²·sa·sb):
#   ∂h₁₁/∂α = -2ci·a·b²·sb
#   ∂h₁₁/∂β =  2ci·b·sb²
#   ∂h₁₂/∂α =  ci·b·(sa·ρ²−2a²b²)     [sa·ρ²−2a²b² = sa²+b²(1−a²)]
#   ∂h₁₂/∂β =  ci·a·(sb·ρ²−2a²b²)     [sb·ρ²−2a²b² = sb²+a²(1−b²)]
#   ∂h₂₂/∂α =  2ci·a·sa²
#   ∂h₂₂/∂β = -2ci·b·a²·sa

@inline function _csphere_inv_metric_grad(r::Float64, x::Point{2})
  a, b   = tan(x[1]), tan(x[2])
  sa, sb = 1 + a^2, 1 + b^2
  ρ2     = 1 + a^2 + b^2
  ci     = 1 / (r^2 * sa * sb)

  dh11_da = -2ci * a * b^2 * sb
  dh11_db =  2ci * b * sb^2
  dh12_da =  ci * b * (sa*ρ2 - 2*a^2*b^2)
  dh12_db =  ci * a * (sb*ρ2 - 2*a^2*b^2)
  dh22_da =  2ci * a * sa^2
  dh22_db = -2ci * b * a^2 * sa

  # Column-major storage with k (derivative) index fastest:
  # [1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2],[2,1,2],[1,2,2],[2,2,2]
  ThirdOrderTensorValue{2,2,2,Float64,8}(
    dh11_da, dh11_db, dh12_da, dh12_db,
    dh12_da, dh12_db, dh22_da, dh22_db,
  )
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CubedSphereInvMetric}, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, ThirdOrderTensorValue{2,2,2,Float64,8}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CubedSphereInvMetric}, x::Point{2}) =
  _csphere_inv_metric_grad(f.object.radius, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CubedSphereInvMetric}, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  r = f.object.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_inv_metric_grad(r, xs[i])
  end
  cache.array
end

inverse_metric_field(f::CubedSphereMetric) = CubedSphereInvMetric(f.radius)
