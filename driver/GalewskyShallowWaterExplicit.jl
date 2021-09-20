module GalewskyShallowWaterExplicit

using Gridap
using GridapGeosciences

# Solves the Galewsky test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a shear flow instability of a zonal 
# jet triggered by an initial gravity wave.
# reference:
#   Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440

function uθ(θϕr)
  θ,ϕ,r = θϕr
  ϵ     = 1.0e-8
  umax  = 80.0
  ϕ₁    = π/7
  ϕ₂    = π/2 - ϕ₁
  en    = exp(-4.0/((ϕ₂ - ϕ₁)*(ϕ₂ - ϕ₁)))
  u     = 0.0
  if ϕ > ϕ₁ + ϵ && ϕ < ϕ₂ - ϵ
    u = (umax/en)*exp(1.0/((ϕ - ϕ₁)*(ϕ - ϕ₂)))
  end
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
  x,y,z = xyz
  θ,ϕ,r = θϕr
  h     = 10000.0
  hh    = 120.0
  α     = 1.0/3.0
  β     = 1.0/15.0
  ϕ₂    = π/4
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
  h = h + hh*cos(ϕ)*exp(-1.0*(θ/α)*(θ/α))*exp(-1.0*((ϕ₂ - ϕ)/β)*((ϕ₂ - ϕ)/β))
  h
end

order  = 1 
degree = 4

n      = 48
nstep  = 20*24*60*2 # 20 days
dt     = 60.0

model = CubedSphereDiscreteModel(n; radius=rₑ)

hf, uf = shallow_water_time_stepper(model, order, degree,
                                    h₀, u₀, f, g,
                                    dt, 0.5*dt, nstep;
                                    write_solution=true,
                                    write_solution_freq=240,
                                    write_diagnostics=true,
                                    write_diagnostics_freq=1,
                                    dump_diagnostics_on_screen=true)

end
