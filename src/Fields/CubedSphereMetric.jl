# ── CubedSphereMetric ─────────────────────────────────────────────────────────
#
# Pullback metric g = JᵀJ for the gnomonic projection.  All 6 panels share the
# same formula because permuting/negating components is an isometry: sign factors
# cancel in JᵀJ (s² = 1) and the permutation maps {c[j1],c[j2],c[j3]} to
# {c[1],c[2],c[3]} in a different order but the same set, giving identical g.
#
# With a = tanα, b = tanβ, sa = 1+a², sb = 1+b², ρ² = 1+a²+b²:
#   g₁₁ = r²·sa²·sb/ρ⁴
#   g₁₂ = −r²·a·b·sa·sb/ρ⁴
#   g₂₂ = r²·sa·sb²/ρ⁴
#
# Inverse (from 2×2 formula, det g = r⁴·sa²·sb²/ρ⁶):
#   g⁻¹ = (ρ²/(r²·sa·sb)) · [[sb, a·b], [a·b, sa]]

function _csphere_metric(r::Float64, x::Point{2,T}) where T
  a, b  = tan(x[1]), tan(x[2])
  sa, sb = 1 + a^2, 1 + b^2
  ρ4    = (1 + a^2 + b^2)^2
  c     = r^2 * sa * sb / ρ4
  SymTensorValue{2,T,3}(c*sa, -c*a*b, c*sb)
end

struct CubedSphereMetric <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CubedSphereMetric, x::Point{2}) =
  _csphere_metric(m.radius, x)
function Gridap.Arrays.return_cache(m::CubedSphereMetric, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, SymTensorValue{2,Float64,3}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereMetric, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  r = m.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_metric(r, xs[i])
  end
  cache.array
end

# ── FieldGradient of CubedSphereMetric ────────────────────────────────────────
#
# Returns ThirdOrderTensorValue{2,2,2} where T[k,i,j] = ∂g_{ij}/∂x_k.
# With a=tanα, b=tanβ, sa=1+a², sb=1+b², ρ²=1+a²+b², c6=r²/ρ⁶:
#   ∂g₁₁/∂α = 4c6·a·sa²·sb·b²
#   ∂g₁₁/∂β = 2c6·b·sa²·sb·(a²-sb)
#   ∂g₁₂/∂α = -c6·b·sa·sb·(ρ²+a²(3b²-sa))
#   ∂g₁₂/∂β = -c6·a·sa·sb·(ρ²+b²(3a²-sb))
#   ∂g₂₂/∂α = 2c6·a·sa·sb²·(b²-sa)
#   ∂g₂₂/∂β = 4c6·b·sa·sb²·a²

@inline function _csphere_metric_grad(r::Float64, x::Point{2})
  a, b   = tan(x[1]), tan(x[2])
  sa, sb = 1 + a^2, 1 + b^2
  ρ2     = 1 + a^2 + b^2
  c6     = r^2 / ρ2^3

  dg11_da = 4c6 * a * sa^2 * sb * b^2
  dg11_db = 2c6 * b * sa^2 * sb * (a^2 - sb)
  dg12_da = -c6 * b * sa * sb * (ρ2 + a^2*(3*b^2 - sa))
  dg12_db = -c6 * a * sa * sb * (ρ2 + b^2*(3*a^2 - sb))
  dg22_da = 2c6 * a * sa * sb^2 * (b^2 - sa)
  dg22_db = 4c6 * b * sa * sb^2 * a^2

  # Column-major storage with k (derivative) index fastest:
  # [1,1,1],[2,1,1],[1,2,1],[2,2,1],[1,1,2],[2,1,2],[1,2,2],[2,2,2]
  ThirdOrderTensorValue{2,2,2,Float64,8}(
    dg11_da, dg11_db, dg12_da, dg12_db,
    dg12_da, dg12_db, dg22_da, dg22_db,
  )
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CubedSphereMetric}, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, ThirdOrderTensorValue{2,2,2,Float64,8}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CubedSphereMetric}, x::Point{2}) =
  _csphere_metric_grad(f.object.radius, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CubedSphereMetric}, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  r = f.object.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_metric_grad(r, xs[i])
  end
  cache.array
end
