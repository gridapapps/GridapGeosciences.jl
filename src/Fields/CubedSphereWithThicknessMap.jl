# ── CubedSphereWithThicknessMap ───────────────────────────────────────────────
#
# 3D extension of CubedSphereMap for one panel of a spherical shell.
#
# Input:  (γ, α, β) ∈ [0,1] × [−π/4, π/4]²
#   γ — normalised radial coordinate: γ=0 → inner sphere, γ=1 → outer sphere
#   α, β — gnomonic chart angles (same as CubedSphereMap)
# Output: (X, Y, Z) ∈ ℝ³
#
# Map:
#   XYZ_surf = _csphere_eval(panel, radius, Point(α,β))
#   φ(γ,α,β) = (1 + thickness·γ/radius) · XYZ_surf
#
# This follows the same extrusion as ForwardMap3D
#   (φ_surf + thickness·γ·n,  n = XYZ_surf/radius)
# but uses the explicit _csphere_eval / _csphere_jac kernels of CubedSphereMap
# instead of forward-AD.
#
# Jacobian J[i,k] = ∂φ_k/∂x_i  (Gridap convention, stored in TensorValue{3,3}):
#   J[1,k] = ∂φ_k/∂γ = (thickness/radius) · XYZ_surf[k]
#   J[2,k] = ∂φ_k/∂α = (1 + thickness·γ/radius) · Jsurf[1,k]
#   J[3,k] = ∂φ_k/∂β = (1 + thickness·γ/radius) · Jsurf[2,k]
# where Jsurf = _csphere_jac(panel, radius, Point(α,β)).

@inline function _csphere3d_eval(panel::Int, radius::Float64, thickness::Float64, x::Point{3})
  γ, α, β  = x[1], x[2], x[3]
  XYZ_surf = _csphere_eval(panel, radius, Point(α, β))
  (1 + thickness * γ / radius) * XYZ_surf
end

@inline function _csphere3d_jac(panel::Int, radius::Float64, thickness::Float64, x::Point{3,T}) where T
  γ, α, β  = x[1], x[2], x[3]
  αβ       = Point(α, β)
  XYZ_surf = _csphere_eval(panel, radius, αβ)
  Jsurf    = _csphere_jac(panel, radius, αβ)   # TensorValue{2,3,Float64}
  t_over_r = thickness / radius
  scale    = 1 + t_over_r * γ
  # Column-major storage: column k = (J[1,k], J[2,k], J[3,k])
  TensorValue{3,3,T}(
    t_over_r * XYZ_surf[1], scale * Jsurf[1,1], scale * Jsurf[2,1],
    t_over_r * XYZ_surf[2], scale * Jsurf[1,2], scale * Jsurf[2,2],
    t_over_r * XYZ_surf[3], scale * Jsurf[1,3], scale * Jsurf[2,3],
  )
end

struct CubedSphereWithThicknessMap <: Field
  panel     :: Int
  radius    :: Float64
  thickness :: Float64
end

Gridap.Arrays.evaluate!(cache, m::CubedSphereWithThicknessMap, x::Point{3}) =
  _csphere3d_eval(m.panel, m.radius, m.thickness, x)
function Gridap.Arrays.return_cache(m::CubedSphereWithThicknessMap, xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, Point{3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereWithThicknessMap, xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  p, r, t = m.panel, m.radius, m.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_eval(p, r, t, xs[i])
  end
  cache.array
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CubedSphereWithThicknessMap},
                                    xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, TensorValue{3,3,Float64}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CubedSphereWithThicknessMap}, x::Point{3}) =
  _csphere3d_jac(f.object.panel, f.object.radius, f.object.thickness, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CubedSphereWithThicknessMap},
                                 xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  p, r, t = f.object.panel, f.object.radius, f.object.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_jac(p, r, t, xs[i])
  end
  cache.array
end

# ── FieldGradient{2} of CubedSphereWithThicknessMap (Hessian) ────────────────
#
# Returns ThirdOrderTensorValue{3,3,3} where T[l,i,k] = ∂J[i,k]/∂x_l = ∂²φ_k/(∂x_i∂x_l),
# J being the Jacobian from FieldGradient{1}.  x = (γ,α,β), scale = 1+thickness·γ/radius.
#
# From J[1,k] = t_over_r·φ_surf[k],  J[2,k] = scale·Jsurf[1,k],  J[3,k] = scale·Jsurf[2,k]:
#
#   T[1,1,k] = 0                              (γγ: φ_surf independent of γ)
#   T[2,1,k] = T[1,2,k] = t_over_r·Jsurf[1,k]  (γα cross term)
#   T[3,1,k] = T[1,3,k] = t_over_r·Jsurf[2,k]  (γβ cross term)
#   T[2,2,k]             = scale·Hsurf[1,1,k]   (αα surface Hessian)
#   T[3,2,k] = T[2,3,k] = scale·Hsurf[1,2,k]   (αβ cross term; Hsurf[1,2,k]=Hsurf[2,1,k])
#   T[3,3,k]             = scale·Hsurf[2,2,k]   (ββ surface Hessian)

@inline function _csphere3d_hess(panel::Int, radius::Float64, thickness::Float64, x::Point{3})
  γ, α, β  = x[1], x[2], x[3]
  αβ       = Point(α, β)
  Jsurf    = _csphere_jac(panel, radius, αβ)    # TensorValue{2,3}
  Hsurf    = _csphere_hess(panel, radius, αβ)   # ThirdOrderTensorValue{2,2,3}
  t_over_r = thickness / radius
  scale    = 1 + t_over_r * γ
  z        = zero(Float64)
  # Column-major storage, l fastest, then i, then k (output component).
  ThirdOrderTensorValue{3,3,3,Float64,27}(
    # k=1:
    z,                    t_over_r*Jsurf[1,1], t_over_r*Jsurf[2,1],  # i=1
    t_over_r*Jsurf[1,1], scale*Hsurf[1,1,1],  scale*Hsurf[2,1,1],   # i=2
    t_over_r*Jsurf[2,1], scale*Hsurf[1,2,1],  scale*Hsurf[2,2,1],   # i=3
    # k=2:
    z,                    t_over_r*Jsurf[1,2], t_over_r*Jsurf[2,2],
    t_over_r*Jsurf[1,2], scale*Hsurf[1,1,2],  scale*Hsurf[2,1,2],
    t_over_r*Jsurf[2,2], scale*Hsurf[1,2,2],  scale*Hsurf[2,2,2],
    # k=3:
    z,                    t_over_r*Jsurf[1,3], t_over_r*Jsurf[2,3],
    t_over_r*Jsurf[1,3], scale*Hsurf[1,1,3],  scale*Hsurf[2,1,3],
    t_over_r*Jsurf[2,3], scale*Hsurf[1,2,3],  scale*Hsurf[2,2,3],
  )
end

function Gridap.Arrays.return_cache(_::FieldGradient{2,<:CubedSphereWithThicknessMap},
                                    xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, ThirdOrderTensorValue{3,3,3,Float64,27}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{2,<:CubedSphereWithThicknessMap}, x::Point{3}) =
  _csphere3d_hess(f.object.panel, f.object.radius, f.object.thickness, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{2,<:CubedSphereWithThicknessMap},
                                 xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  p, r, t = f.object.panel, f.object.radius, f.object.thickness
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere3d_hess(p, r, t, xs[i])
  end
  cache.array
end
