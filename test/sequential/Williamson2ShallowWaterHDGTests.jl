module Williamson2ShallowWaterHDGTests

using FillArrays
using Test
using WriteVTK
using Gridap
using GridapGeosciences
using GridapHybrid

include("Williamson2InitialConditions.jl")

function Gridap.Geometry.push_normal(invJt,n)
  v = invJt⋅n
  m = sqrt(inner(v,v))
  if m < eps()
    return zero(v)
  else
    return v/m
  end
end

function f3(xyz)
  f   = f₀(xyz)
  θϕr = xyz2θϕr(xyz)
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(0,0,f)
end

# Solves the steady state Williamson2 test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a modified coriolis term that exactly balances
# the potential gradient term to achieve a steady state
# reference:
# D. L. Williamson, J. B. Drake, J. J.HackRüdiger Jakob, P. N.Swarztrauber, (1992)
# J Comp. Phys. 102 211-224

#l2_err_u = [0.009872849324844975, 0.002843859325502451,  0.0007415233680147055]
#l2_err_h = [0.00561048961466849,  0.0014553891676895917, 0.0003681302039168149]

order  = 1
degree = 4

for i in 1:1
  n      = 4*2^i
  nstep  = 5*n
  Uc     = sqrt(g*H₀)
  dx     = 2.0*π*rₑ/(4*n)
  dt     = 0.05*dx/Uc
  println("timestep: ", dt)   # gravity wave time step

  model = CubedSphereDiscreteModel(n,2; radius=rₑ)
  shallow_water_hdg(model, order, degree,
                             ϕ₀, u₀, f3, g,
                             dt, nstep;
                             write_solution=true,
                             write_solution_freq=1,
                             write_diagnostics=true,
                             write_diagnostics_freq=1,
                             dump_diagnostics_on_screen=true)

  #Ω     = Triangulation(model)
  #dΩ    = Measure(Ω, degree)
  #hc    = CellField(h₀, Ω)
  #e     = h₀-hf
  #err_h = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(hc⋅hc)*dΩ))
  #uc    = CellField(u₀, Ω)
  #e     = u₀-uf
  #err_u = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(uc⋅uc)*dΩ))
  #println("n=", n, ",\terr_u: ", err_u, ",\terr_h: ", err_h)

  #@test abs(err_u - l2_err_u[i]) < 10.0^-12
  #@test abs(err_h - l2_err_h[i]) < 10.0^-12
end

end
