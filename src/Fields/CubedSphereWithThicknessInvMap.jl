# ── CubedSphereWithThicknessInvMap ────────────────────────────────────────────
#
# Inverse of CubedSphereWithThicknessMap: maps an ambient point (X,Y,Z) back to
# chart coordinates (γ,α,β) ∈ [0,1] × [−π/4, π/4]².
#
# The forward map is φ(γ,α,β) = (1 + thickness·γ/radius)·φ_surf(α,β),
# where φ_surf lies on the sphere of radius `radius`.  The inverse is:
#   ρ   = ‖xyz‖
#   γ   = (ρ − radius) / thickness
#   α,β = _csphere_inv_eval(panel, xyz)    (scale-invariant)
#
# Jacobian J[i,k] = ∂φ_inv_k/∂xyz_i  (TensorValue{3,3}):
#   col k=1 (γ): ∂γ/∂xyz_i = xyz[i] / (thickness·ρ)
#   col k=2 (α): ∂α/∂xyz_i = (_csphere_inv_jac(panel,xyz))[i,1]
#   col k=3 (β): ∂β/∂xyz_i = (_csphere_inv_jac(panel,xyz))[i,2]

@inline function _csphere3d_inv_eval(panel::Int, radius::Float64, thickness::Float64, x::Point{3})
  ρ = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
  αβ = _csphere_inv_eval(panel, x)
  Point((ρ - radius) / thickness, αβ[1], αβ[2])
end

@inline function _csphere3d_inv_jac(panel::Int, radius::Float64, thickness::Float64, x::Point{3,T}) where T
  ρ = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
  Jαβ = _csphere_inv_jac(panel, x)   # TensorValue{3,2}: J[i,k] = ∂(α,β)_k/∂xyz_i
  inv_tρ = inv(thickness * ρ)
  TensorValue{3,3,T}(
    x[1]*inv_tρ, x[2]*inv_tρ, x[3]*inv_tρ,  # col k=1: ∂γ/∂xyz
    Jαβ[1,1], Jαβ[2,1], Jαβ[3,1],           # col k=2: ∂α/∂xyz
    Jαβ[1,2], Jαβ[2,2], Jαβ[3,2],           # col k=3: ∂β/∂xyz
  )
end

struct CubedSphereWithThicknessInvMap <: Field
  panel     :: Int
  radius    :: Float64
  thickness :: Float64
end

Gridap.Arrays.evaluate!(_, m::CubedSphereWithThicknessInvMap, x::Point{3}) =
  _csphere3d_inv_eval(m.panel, m.radius, m.thickness, x)
function Gridap.Arrays.return_cache(_::CubedSphereWithThicknessInvMap, xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, Point{3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereWithThicknessInvMap, xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  p, r, t = m.panel, m.radius, m.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_inv_eval(p, r, t, xs[i])
  end
  cache.array
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CubedSphereWithThicknessInvMap},
                                    xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, TensorValue{3,3,Float64}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CubedSphereWithThicknessInvMap}, x::Point{3}) =
  _csphere3d_inv_jac(f.object.panel, f.object.radius, f.object.thickness, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CubedSphereWithThicknessInvMap},
                                 xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  p, r, t = f.object.panel, f.object.radius, f.object.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_inv_jac(p, r, t, xs[i])
  end
  cache.array
end
