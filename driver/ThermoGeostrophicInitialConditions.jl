# Initial conditions for thermogeostrophic balance for the thermal
# shallow water equations. Adapted for the sphere from the planar
# configuration in section 5.2 of:
#     Eldred, Dubos and Kritsikis, JCP 379, 2018

const H₀ = 5960.0
const u₀ = 20.0
const c₀ = 0.05

function uθ(θϕr)
  θ,ϕ,r = θϕr
  u = u₀*cos(ϕ)
  u
end

# Initial velocity
function uᵢ(xyz)
  θϕr = xyz2θϕr(xyz)
  u   = uθ(θϕr)
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(u,0,0)
end

# Initial fluid depth
function hᵢ(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  h     = H₀
  ni    = 1000
  ϕₚ    = 0.0
  dϕ    = abs(ϕ/ni)
  sgn   = 1.0
  if ϕ < 0.0
    sgn = -1.0
  end
  for i in 1:ni
    ϕₚ   = ϕₚ + sgn*dϕ
    _θϕr = VectorValue(θ,ϕₚ,r)
    u    = uθ(_θϕr)
    _f   = 2.0*Ωₑ*sin(ϕₚ)
    h    = h - rₑ*u*(_f + tan(ϕₚ)*u/rₑ)*dϕ/g
  end
  h
end

function sᵢ(xyz)
  h = hᵢ(xyz)
  s = g*(1.0 + c₀*c₀*H₀*H₀/h/h)
  s
end

function Sᵢ(xyz)
  h = hᵢ(xyz)
  s = sᵢ(xyz)
  S = s*h
  S
end

# Topography
function topography(xyz)
  0.0
end
