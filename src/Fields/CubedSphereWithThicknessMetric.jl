# ── CubedSphereWithThicknessMetric ────────────────────────────────────────────
#
# Pullback metric g = JᵀJ for CubedSphereWithThicknessMap.
#
# Because φ(γ,α,β) = scale · φ_surf(α,β) with scale = 1 + thickness·γ/radius,
# and φ_surf lies on a sphere so φ_surf ⊥ ∂φ_surf/∂α and φ_surf ⊥ ∂φ_surf/∂β,
# the metric is block-diagonal:
#
#   g₁₁ = thickness²                                (pure γγ term)
#   g₁₂ = g₁₃ = 0                                   (sphere orthogonality)
#   g₂₂ = scale²·r²·sa²·sb/ρ⁴  = scale²·g_surf₁₁
#   g₂₃ = −scale²·r²·a·b·sa·sb/ρ⁴ = scale²·g_surf₁₂
#   g₃₃ = scale²·r²·sa·sb²/ρ⁴  = scale²·g_surf₂₂
#
# with a=tanα, b=tanβ, sa=1+a², sb=1+b², ρ²=1+a²+b², scale=1+thickness·γ/radius.
#
# Inverse (block-diagonal):
#   h₁₁ = 1/thickness²
#   h₁₂ = h₁₃ = 0
#   h₂₂ = (1/scale²)·g_surf⁻¹₁₁ = ρ²·sb/(scale²·r²·sa·sb)
#   h₂₃ = (1/scale²)·g_surf⁻¹₁₂ = ρ²·a·b/(scale²·r²·sa·sb)
#   h₃₃ = (1/scale²)·g_surf⁻¹₂₂ = ρ²·sa/(scale²·r²·sa·sb)

function _csphere3d_metric(radius::Float64, thickness::Float64, x::Point{3,T}) where T
  γ, α, β = x[1], x[2], x[3]
  a, b   = tan(α), tan(β)
  sa, sb = 1 + a^2, 1 + b^2
  ρ4     = (1 + a^2 + b^2)^2
  s2     = (1 + thickness * γ / radius)^2
  c      = radius^2 * sa * sb / ρ4
  SymTensorValue{3,T,6}(
    T(thickness^2), zero(T), zero(T),
    s2*c*sa, -s2*c*a*b, s2*c*sb,
  )
end

struct CubedSphereWithThicknessMetric <: Field
  radius    :: Float64
  thickness :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CubedSphereWithThicknessMetric, x::Point{3}) =
  _csphere3d_metric(m.radius, m.thickness, x)
function Gridap.Arrays.return_cache(m::CubedSphereWithThicknessMetric, xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, SymTensorValue{3,Float64,6}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereWithThicknessMetric, xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  r, t = m.radius, m.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_metric(r, t, xs[i])
  end
  cache.array
end

# ── FieldGradient of CubedSphereWithThicknessMetric ───────────────────────────
#
# Returns ThirdOrderTensorValue{3,3,3} where T[k,i,j] = ∂g_{ij}/∂x_k,
# g = JᵀJ, x = (γ,α,β).  g₁₁=thickness² and g₁₂=g₁₃=0 are constant;
# only the (2,2)/(2,3)/(3,3) block has non-zero derivatives.
# With a=tanα, b=tanβ, sa=1+a², sb=1+b², ρ²=1+a²+b²,
# scale=1+thickness·γ/radius, s2=scale², cg=s2·radius²/ρ⁶,
# fγ=2·(thickness/radius)/scale:
#
#   ∂g₂₂/∂γ = fγ·g₂₂               ∂g₂₃/∂γ = fγ·g₂₃               ∂g₃₃/∂γ = fγ·g₃₃
#   ∂g₂₂/∂α = 4cg·a·sa²·sb·b²       ∂g₂₃/∂α = −cg·b·sa·sb·(ρ²+a²(3b²−sa))  ∂g₃₃/∂α = 2cg·a·sa·sb²·(b²−sa)
#   ∂g₂₂/∂β = 2cg·b·sa²·sb·(a²−sb)  ∂g₂₃/∂β = −cg·a·sa·sb·(ρ²+b²(3a²−sb)) ∂g₃₃/∂β = 4cg·b·sa·sb²·a²

@inline function _csphere3d_metric_grad(radius::Float64, thickness::Float64, x::Point{3})
  γ, α, β  = x[1], x[2], x[3]
  a, b     = tan(α), tan(β)
  sa, sb   = 1 + a^2, 1 + b^2
  ρ2       = 1 + a^2 + b^2
  t_over_r = thickness / radius
  scale    = 1 + t_over_r * γ
  s2       = scale^2
  c        = radius^2 * sa * sb / ρ2^2
  g22      = s2 * c * sa
  g23      = -s2 * c * a * b
  g33      = s2 * c * sb
  fγ       = 2 * t_over_r / scale
  cg       = s2 * radius^2 / ρ2^3   # = s2·c6 where c6=r²/ρ⁶

  dg22_dγ = fγ * g22
  dg23_dγ = fγ * g23
  dg33_dγ = fγ * g33

  dg22_dα =  4cg * a * sa^2 * sb * b^2
  dg23_dα = -cg * b * sa * sb * (ρ2 + a^2*(3*b^2 - sa))
  dg33_dα =  2cg * a * sa * sb^2 * (b^2 - sa)

  dg22_dβ =  2cg * b * sa^2 * sb * (a^2 - sb)
  dg23_dβ = -cg * a * sa * sb * (ρ2 + b^2*(3*a^2 - sb))
  dg33_dβ =  4cg * b * sa * sb^2 * a^2

  z = zero(Float64)
  # Column-major storage with k (derivative) index fastest: T[k,i,j] = ∂g_{ij}/∂x_k.
  ThirdOrderTensorValue{3,3,3,Float64,27}(
    z, z, z,  z,       z,       z,       z,       z,       z,        # j=1: all zero
    z, z, z,  dg22_dγ, dg22_dα, dg22_dβ, dg23_dγ, dg23_dα, dg23_dβ, # j=2
    z, z, z,  dg23_dγ, dg23_dα, dg23_dβ, dg33_dγ, dg33_dα, dg33_dβ, # j=3
  )
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CubedSphereWithThicknessMetric}, xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, ThirdOrderTensorValue{3,3,3,Float64,27}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CubedSphereWithThicknessMetric}, x::Point{3}) =
  _csphere3d_metric_grad(f.object.radius, f.object.thickness, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CubedSphereWithThicknessMetric}, xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  r, t = f.object.radius, f.object.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_metric_grad(r, t, xs[i])
  end
  cache.array
end
