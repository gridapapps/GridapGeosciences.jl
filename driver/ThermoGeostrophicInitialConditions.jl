# Initial conditions for thermogeostrophic balance for the thermal
# shallow water equations. Adapted for the sphere from the planar
# configuration in section 5.2 of:
#     Eldred, Dubos and Kritsikis, JCP 379, 2018

const H₀ = 5960.0
const U₀ = 20.0
const C₀ = 0.1

function uθ(θϕr)
  θ,ϕ,r = θϕr
  u = U₀*cos(ϕ)
  u
end

# Initial velocity
function u₀(xyz)
  θϕr = xyz2θϕr(xyz)
  u   = uθ(θϕr)
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(u,0,0)
end

# Initial fluid depth
function h₀(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  #h     = H₀
  #ni    = 1000
  #ϕₚ    = 0.0
  #dϕ    = abs(ϕ/ni)
  #sgn   = 1.0
  #if ϕ < 0.0
  #  sgn = -1.0
  #end
  #for i in 1:ni
  #  ϕₚ   = ϕₚ + sgn*dϕ
  #  _θϕr = VectorValue(θ,ϕₚ,r)
  #  u    = uθ(_θϕr)
  #  _f   = 2.0*Ωₑ*sin(ϕₚ)
  #  h    = h - rₑ*u*(_f + tan(ϕₚ)*u/rₑ)*dϕ/g
  #end
  h = -1.0*U₀*Ωₑ*xyz[3]*xyz[3]/rₑ/g + H₀
  #Rc = π/9.0
  Rc = π/8.0
  θc = -π/2.0 # mountain top longitude
  ϕc = +π/6.0 # mountain top latitude
  rc = sqrt((θ - θc)*(θ - θc) + (ϕ - ϕc)*(ϕ - ϕc))
  if rc < Rc
    h = h + 120.0*(1.0 - rc/Rc)
  end
  h
end

function s₀(xyz)
  h = h₀(xyz)
  s = g*(1.0 + C₀*H₀*H₀/h/h)
  s
end

function S₀(xyz)
  h = h₀(xyz)
  s = s₀(xyz)
  S = s*h
  S
end

# Topography
function topography(xyz)
  0.0
end
