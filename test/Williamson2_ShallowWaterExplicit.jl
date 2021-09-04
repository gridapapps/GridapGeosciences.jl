module Williamson2_ShallowWaterExplicit

using FillArrays
using Test
using WriteVTK
using Gridap
using GridapGeosciences

# Solves the steady state Williamson2 test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a modified coriolis term that exactly balances
# the potential gradient term to achieve a steady state
# reference:
# D. L. Williamson, J. B. Drake, J. J.HackRüdiger Jakob, P. N.Swarztrauber, (1992)
# J Comp. Phys. 102 211-224

# Constants of the Williamson2 test case
const α  = π/4.0              # deviation of the coriolis term from zonal forcing
const U₀ = 38.61068276698372  # velocity scale
const H₀ = 2998.1154702758267 # mean fluid depth

# Modified coriolis term
function f₀(xyz)
   θϕr   = xyz2θϕr(xyz)
   θ,ϕ,r = θϕr
   2.0*Ωₑ*( -cos(θ)*cos(ϕ)*sin(α) + sin(ϕ)*cos(α) )
end

# Initial velocity
function u₀(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  u     = U₀*(cos(ϕ)*cos(α) + cos(θ)*sin(ϕ)*sin(α))
  v     = -U₀*sin(θ)*sin(α)
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(u,v,0)
end

# Initial fluid depth
function h₀(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  h  = -cos(θ)*cos(ϕ)*sin(α) + sin(ϕ)*cos(α)
  H₀ - (rₑ*Ωₑ*U₀ + 0.5*U₀*U₀)*h*h/g
end

l2_err_u = [0.011504453807392859, 0.003188984305055811,  0.0008192298898147198]
l2_err_h = [0.005636335001937436, 0.0014571807037802682, 0.0003681933640549439]

order  = 1 
degree = 4

for i in 1:3
  n      = 2*2^i
  nstep  = 5*n
  Uc     = sqrt(g*H₀)
  dx     = 2.0*π*rₑ/(4*n)
  dt     = 0.05*dx/Uc
  println("timestep: ", dt)   # gravity wave time step

  model = CubedSphereDiscreteModel(n; radius=rₑ)

  hf, uf = shallow_water_time_stepper(model, order, degree, h₀, u₀, f₀, g, nstep, 1, 20, dt, 0.0*dt)

  Ω     = Triangulation(model)
  dΩ    = Measure(Ω, degree)
  hc    = CellField(h₀, Ω)
  e     = h₀-hf
  err_h = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(hc⋅hc)*dΩ))
  uc    = CellField(u₀, Ω)
  e     = u₀-uf
  err_u = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(uc⋅uc)*dΩ))
  println("n=", n, ",\terr_u: ", err_u, ",\terr_h: ", err_h)

  @test abs(err_u - l2_err_u[i]) < 10.0^-12
  @test abs(err_h - l2_err_h[i]) < 10.0^-12
end

end
