# ── CubedSphereMap ────────────────────────────────────────────────────────────
#
# Gnomonic projection for one panel of the cubed sphere.
# All 6 panels share the base map h(α,β) = (1/ρ, tanα/ρ, tanβ/ρ) with ρ = √(1+tan²α+tan²β);
# each panel permutes/negates the three components of h.
#
# _CSPHERE_PERM[p] = (j1,s1, j2,s2, j3,s3):
#   output component k = s_k · h[j_k],   k = 1,2,3
# This same table drives both the forward map and the explicit Jacobian.
#
# Jacobian derivation (Gridap convention: J[i,k] = ∂φ_k/∂x_i):
#   Base Jh columns (i=1 → ∂/∂α, i=2 → ∂/∂β):
#     col 1: (−a·sα/ρ³,  −b·sβ/ρ³)
#     col 2: ( sα·sβ/ρ³, −a·b·sβ/ρ³)
#     col 3: (−a·b·sα/ρ³, sα·sβ/ρ³)
#   where a = tanα, b = tanβ, sα = 1+a², sβ = 1+b², ρ³ = (1+a²+b²)^(3/2).
#   Panel p Jacobian: J_p[i,k] = r · s_k · Jh[i, j_k]

const _CSPHERE_PERM = (
    (1, 1, 2, 1, 3, 1),   # panel 1: φ = r*(  h₁,  h₂,  h₃)
    (3,-1, 2, 1, 1, 1),   # panel 2: φ = r*( -h₃,  h₂,  h₁)
    (2,-1, 1, 1, 3, 1),   # panel 3: φ = r*( -h₂,  h₁,  h₃)
    (1,-1, 3, 1, 2, 1),   # panel 4: φ = r*( -h₁,  h₃,  h₂)
    (2,-1, 3, 1, 1,-1),   # panel 5: φ = r*( -h₂,  h₃, -h₁)
    (3,-1, 1,-1, 2, 1),   # panel 6: φ = r*( -h₃, -h₁,  h₂)
)

function _csphere_eval(panel::Int, r::Float64, x::Point{2})
  a, b = tan(x[1]), tan(x[2])
  ρ    = sqrt(1 + a^2 + b^2)
  h    = (1/ρ, a/ρ, b/ρ)
  j1,s1, j2,s2, j3,s3 = _CSPHERE_PERM[panel]
  r * Point(s1*h[j1], s2*h[j2], s3*h[j3])
end

function _csphere_jac(panel::Int, r::Float64, x::Point{2,T}) where T
  a, b  = tan(x[1]), tan(x[2])
  ρ2    = 1 + a^2 + b^2
  ρ     = sqrt(ρ2); ρ3 = ρ2 * ρ
  sa, sb = 1+a^2, 1+b^2
  # Base Jh columns: c[k] = (Jh[1,k], Jh[2,k]) scaled by ρ³
  c = ((-a*sa, -b*sb), (sa*sb, -a*b*sb), (-a*b*sa, sa*sb))
  j1,s1, j2,s2, j3,s3 = _CSPHERE_PERM[panel]
  (r/ρ3) * TensorValue{2,3,T}(
    s1*c[j1][1], s1*c[j1][2],
    s2*c[j2][1], s2*c[j2][2],
    s3*c[j3][1], s3*c[j3][2],
  )
end

struct CubedSphereMap <: Field
  panel  :: Int
  radius :: Float64
end

Gridap.Arrays.evaluate!(cache, m::CubedSphereMap, x::Point{2}) =
  _csphere_eval(m.panel, m.radius, x)
function Gridap.Arrays.return_cache(m::CubedSphereMap, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, Point{3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereMap, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  p, r = m.panel, m.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_eval(p, r, xs[i])
  end
  cache.array
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CubedSphereMap}, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, TensorValue{2,3,Float64}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CubedSphereMap}, x::Point{2}) =
  _csphere_jac(f.object.panel, f.object.radius, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CubedSphereMap}, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  p, r = f.object.panel, f.object.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_jac(p, r, xs[i])
  end
  cache.array
end

# ── FieldGradient{2} of CubedSphereMap (Hessian) ─────────────────────────────
#
# Returns ThirdOrderTensorValue{2,2,3} where T[l,i,k] = ∂J[i,k]/∂x_l = ∂²φ_k/(∂x_i∂x_l),
# J being the Jacobian returned by FieldGradient{1,<:CubedSphereMap}.
#
# Derivation: J = (r/ρ³)·P with ∂(r/ρ³)/∂x_l = -3a_l·s_l·(r/ρ⁵), so by the product rule
# ∂J[i,k]/∂x_l = (r/ρ⁵)·s_k·N_{j_k}[l,i], where the N matrices absorb all a/b-dependence:
#   N_j[l,i] = ρ²·∂p_j[i]/∂x_l − 3a_l·s_l·p_j[i]
# With D=ρ²+3a²b², E=3ab·sa·sb, F=a·sa·sb·(2b²−sa), G=b·sa·sb·(2a²−sb):
#   N₁ = (−sa·D, E, E, −sb·D)        entries: (N[α,1], N[β,1], N[α,2], N[β,2])
#   N₂ = (F, G, G, −a·sb·D)
#   N₃ = (−b·sa·D, F, F, G)

@inline function _csphere_hess(panel::Int, r::Float64, x::Point{2})
  a, b   = tan(x[1]), tan(x[2])
  sa, sb = 1 + a^2, 1 + b^2
  ρ2     = 1 + a^2 + b^2
  ρ5     = ρ2^2 * sqrt(ρ2)

  D = ρ2 + 3*a^2*b^2
  E = 3*a*b*sa*sb
  F = a*sa*sb*(2*b^2 - sa)
  G = b*sa*sb*(2*a^2 - sb)

  n = (
    (-sa*D,   E,  E,  -sb*D  ),   # N₁: base column p₁ = (−a·sa, −b·sb)
    ( F,      G,  G,  -a*sb*D),   # N₂: base column p₂ = (sa·sb, −a·b·sb)
    (-b*sa*D, F,  F,   G     ),   # N₃: base column p₃ = (−a·b·sa, sa·sb)
  )
  j1,s1, j2,s2, j3,s3 = _CSPHERE_PERM[panel]
  nj1, nj2, nj3 = n[j1], n[j2], n[j3]
  fac = r / ρ5

  # Column-major layout: [l,i,k] with l fastest, then i, then k (output component)
  ThirdOrderTensorValue{2,2,3,Float64,12}(
    fac*s1*nj1[1], fac*s1*nj1[2], fac*s1*nj1[3], fac*s1*nj1[4],
    fac*s2*nj2[1], fac*s2*nj2[2], fac*s2*nj2[3], fac*s2*nj2[4],
    fac*s3*nj3[1], fac*s3*nj3[2], fac*s3*nj3[3], fac*s3*nj3[4],
  )
end

function Gridap.Arrays.return_cache(_::FieldGradient{2,<:CubedSphereMap}, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, ThirdOrderTensorValue{2,2,3,Float64,12}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{2,<:CubedSphereMap}, x::Point{2}) =
  _csphere_hess(f.object.panel, f.object.radius, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{2,<:CubedSphereMap}, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  p, r = f.object.panel, f.object.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_hess(p, r, xs[i])
  end
  cache.array
end
