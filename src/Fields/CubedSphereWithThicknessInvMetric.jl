function _csphere3d_inv_metric(radius::Float64, thickness::Float64, x::Point{3,T}) where T
  γ, α, β = x[1], x[2], x[3]
  a, b   = tan(α), tan(β)
  sa, sb = 1 + a^2, 1 + b^2
  ρ2     = 1 + a^2 + b^2
  s2     = (1 + thickness * γ / radius)^2
  ci     = ρ2 / (radius^2 * sa * sb)
  SymTensorValue{3,T,6}(
    T(1 / thickness^2), zero(T), zero(T),
    ci*sb/s2, ci*a*b/s2, ci*sa/s2,
  )
end

struct CubedSphereWithThicknessInvMetric <: Field
  radius    :: Float64
  thickness :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CubedSphereWithThicknessInvMetric, x::Point{3}) =
  _csphere3d_inv_metric(m.radius, m.thickness, x)
function Gridap.Arrays.return_cache(m::CubedSphereWithThicknessInvMetric, xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, SymTensorValue{3,Float64,6}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereWithThicknessInvMetric, xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  r, t = m.radius, m.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_inv_metric(r, t, xs[i])
  end
  cache.array
end

inverse_metric_field(f::CubedSphereWithThicknessMetric) =
  CubedSphereWithThicknessInvMetric(f.radius, f.thickness)

# ── FieldGradient of CubedSphereWithThicknessInvMetric ────────────────────────
#
# Returns ThirdOrderTensorValue{3,3,3} where T[k,i,j] = ∂h_{ij}/∂x_k,
# h = g⁻¹, x = (γ,α,β).  h₁₁=1/thickness² and h₁₂=h₁₃=0 are constant;
# only the (2,2)/(2,3)/(3,3) block has non-zero derivatives.
# With a=tanα, b=tanβ, sa=1+a², sb=1+b², ρ²=1+a²+b²,
# scale=1+thickness·γ/radius, s2=scale², ciq=1/(radius²·sa·sb·s2),
# fγ=−2·(thickness/radius)/scale:
#
#   ∂h₂₂/∂γ = fγ·h₂₂              ∂h₂₃/∂γ = fγ·h₂₃              ∂h₃₃/∂γ = fγ·h₃₃
#   ∂h₂₂/∂α = −2a·b²·ciq·sb       ∂h₂₃/∂α = ciq·b·(sa·ρ²−2a²b²)  ∂h₃₃/∂α = 2a·ciq·sa²
#   ∂h₂₂/∂β = 2b·ciq·sb²          ∂h₂₃/∂β = ciq·a·(sb·ρ²−2a²b²)  ∂h₃₃/∂β = −2b·a²·ciq·sa

@inline function _csphere3d_inv_metric_grad(radius::Float64, thickness::Float64, x::Point{3})
  γ, α, β  = x[1], x[2], x[3]
  a, b     = tan(α), tan(β)
  sa, sb   = 1 + a^2, 1 + b^2
  ρ2       = 1 + a^2 + b^2
  t_over_r = thickness / radius
  scale    = 1 + t_over_r * γ
  s2       = scale^2
  ci       = ρ2 / (radius^2 * sa * sb)
  h22      = ci * sb / s2
  h23      = ci * a * b / s2
  h33      = ci * sa / s2
  fγ       = -2 * t_over_r / scale
  ciq      = ci / (ρ2 * s2)   # = 1/(radius²·sa·sb·scale²)

  dh22_dγ = fγ * h22
  dh23_dγ = fγ * h23
  dh33_dγ = fγ * h33

  dh22_dα = -2a * b^2 * ciq * sb
  dh23_dα =  ciq * b * (sa*ρ2 - 2*a^2*b^2)
  dh33_dα =  2a * ciq * sa^2

  dh22_dβ =  2b * ciq * sb^2
  dh23_dβ =  ciq * a * (sb*ρ2 - 2*a^2*b^2)
  dh33_dβ = -2b * a^2 * ciq * sa

  z = zero(Float64)
  # Column-major storage with k (derivative) index fastest: T[k,i,j] = ∂h_{ij}/∂x_k.
  ThirdOrderTensorValue{3,3,3,Float64,27}(
    z, z, z,  z,       z,       z,       z,       z,       z,        # j=1: all zero
    z, z, z,  dh22_dγ, dh22_dα, dh22_dβ, dh23_dγ, dh23_dα, dh23_dβ, # j=2
    z, z, z,  dh23_dγ, dh23_dα, dh23_dβ, dh33_dγ, dh33_dα, dh33_dβ, # j=3
  )
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CubedSphereWithThicknessInvMetric}, xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, ThirdOrderTensorValue{3,3,3,Float64,27}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CubedSphereWithThicknessInvMetric}, x::Point{3}) =
  _csphere3d_inv_metric_grad(f.object.radius, f.object.thickness, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CubedSphereWithThicknessInvMetric}, xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  r, t = f.object.radius, f.object.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_inv_metric_grad(r, t, xs[i])
  end
  cache.array
end
