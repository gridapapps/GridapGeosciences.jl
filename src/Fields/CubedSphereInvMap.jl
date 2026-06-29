# ── CubedSphereInvMap ─────────────────────────────────────────────────────────
#
# Inverse of CubedSphereMap: gnomonic de-projection from a sphere point
# back to chart (α,β) ∈ [−π/4, π/4]².
#
# _CSPHERE_INV_PERM[p] = (i1,sg1, i2,sg2, i3,sg3) encodes:
#   h₁ = sg1·xyz[i1],  h₂ = sg2·xyz[i2],  h₃ = sg3·xyz[i3]
#   α  = atan(h₂, h₁),  β  = atan(h₃, h₁)
# Derived from _CSPHERE_PERM by inversion: since output_k = s_k·h[j_k] and s_k²=1,
# h[j_k] = s_k·output_k, so the output component at position k recovers h[j_k].

const _CSPHERE_INV_PERM = (
    (1, 1, 2, 1, 3, 1),   # panel 1: h₁=X,   h₂=Y,   h₃=Z
    (3, 1, 2, 1, 1,-1),   # panel 2: h₁=Z,   h₂=Y,   h₃=-X
    (2, 1, 1,-1, 3, 1),   # panel 3: h₁=Y,   h₂=-X,  h₃=Z
    (1,-1, 3, 1, 2, 1),   # panel 4: h₁=-X,  h₂=Z,   h₃=Y
    (3,-1, 1,-1, 2, 1),   # panel 5: h₁=-Z,  h₂=-X,  h₃=Y
    (2,-1, 3, 1, 1,-1),   # panel 6: h₁=-Y,  h₂=Z,   h₃=-X
)

@inline function _csphere_inv_eval(panel::Int, xyz::Point{3})
  i1,sg1, i2,sg2, i3,sg3 = _CSPHERE_INV_PERM[panel]
  h1 = sg1 * xyz[i1]
  h2 = sg2 * xyz[i2]
  h3 = sg3 * xyz[i3]
  Point(atan(h2, h1), atan(h3, h1))
end

# Jacobian of the inverse map: J[j,k] = ∂φ_k/∂xyz_j (TensorValue{3,2}).
# α = atan(h₂,h₁), β = atan(h₃,h₁); each hᵢ depends on exactly one xyz component,
# so each column of J has exactly 2 non-zero entries.
@inline function _csphere_inv_jac(panel::Int, xyz::Point{3})
  i1,sg1, i2,sg2, i3,sg3 = _CSPHERE_INV_PERM[panel]
  h1 = sg1 * xyz[i1]
  h2 = sg2 * xyz[i2]
  h3 = sg3 * xyz[i3]
  r12 = h1^2 + h2^2
  r13 = h1^2 + h3^2
  da_i1 = -h2 * sg1 / r12;  da_i2 =  h1 * sg2 / r12
  db_i1 = -h3 * sg1 / r13;  db_i3 =  h1 * sg3 / r13
  da = ntuple(j -> ifelse(j==i1, da_i1, ifelse(j==i2, da_i2, 0.0)), Val(3))
  db = ntuple(j -> ifelse(j==i1, db_i1, ifelse(j==i3, db_i3, 0.0)), Val(3))
  TensorValue{3,2,Float64}(da[1], da[2], da[3], db[1], db[2], db[3])
end

struct CubedSphereInvMap <: Field
  panel  :: Int
  radius :: Float64
end

Gridap.Arrays.evaluate!(cache, m::CubedSphereInvMap, x::Point{3}) =
  _csphere_inv_eval(m.panel, x)
function Gridap.Arrays.return_cache(m::CubedSphereInvMap, xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, Point{2,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereInvMap, xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  p = m.panel
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_inv_eval(p, xs[i])
  end
  cache.array
end

function Gridap.Arrays.return_cache(_::FieldGradient{1,<:CubedSphereInvMap}, xs::AbstractArray{<:Point{3}})
  CachedArray(similar(xs, TensorValue{3,2,Float64}))
end
Gridap.Arrays.evaluate!(_, f::FieldGradient{1,<:CubedSphereInvMap}, x::Point{3}) =
  _csphere_inv_jac(f.object.panel, x)
function Gridap.Arrays.evaluate!(cache, f::FieldGradient{1,<:CubedSphereInvMap}, xs::AbstractArray{<:Point{3}})
  setsize!(cache, size(xs))
  p = f.object.panel
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_inv_jac(p, xs[i])
  end
  cache.array
end
