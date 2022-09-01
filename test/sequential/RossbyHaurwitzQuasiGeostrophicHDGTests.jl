module RossbyHaurwitzQuasiGeostrophicHDGTests

using FillArrays
using Test
using WriteVTK
using Gridap
using GridapGeosciences
using GridapHybrid

function Gridap.Geometry.push_normal(invJt,n)
  v = invJt⋅n
  m = sqrt(inner(v,v))
  if m < eps()
    return zero(v)
  else
    return v/m
  end
end

function f₀(xyz)
  # magic numbers
  Ω     = 7.292e-5 # rotational frequency of the earth
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  2.0*Ω*sin(ϕ)
end

function ω₀(xyz)
  # magic numbers
  Ω     = 7.292e-5           # rotational frequency of the earth
  rₑ    = 6371220.0          # radius of the earth
  RHΩ   = 7.848e-6
  RHK   = 7.848e-6
  RHR   = 4
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  2*RHΩ*sin(ϕ) - RHK*sin(ϕ)*cos(ϕ)^RHR*(RHR*RHR + 3*RHR + 2)*cos(RHR*θ)
end

function u₀(xyz)
  # magic numbers
  Ω     = 7.292e-5           # rotational frequency of the earth
  rₑ    = 6371220.0          # radius of the earth
  RHΩ   = 7.848e-6
  RHK   = 7.848e-6
  RHR   = 4
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  u     = rₑ*RHΩ*cos(ϕ) + rₑ*RHK*cos(ϕ)^(RHR-1)*(RHR*sin(ϕ)*sin(ϕ)* - cos(ϕ)*cos(ϕ))*cos(RHR*θ)
  v     = -rₑ*RHK*RHR*cos(ϕ)^(RHR-1)*sin(ϕ)*sin(RHR*θ)
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(u,v,0)
end

# Solves the Williamson6 test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a modified coriolis term that exactly balances
# the potential gradient term to achieve a steady state
# reference:
# D. L. Williamson, J. B. Drake, J. J.HackRüdiger Jakob, P. N.Swarztrauber, (1992)
# J Comp. Phys. 102 211-224

#l2_err_u = 
#l2_err_h = 

order  = 1
degree = 4

for i in 1:1
  n      = 4*2^i
  nstep  = 5*n
  g      = 9.80616            # gravitational acceleration
  H₀     = 8.0e+3             # mean fluid depth
  Uc     = sqrt(g*H₀)
  dx     = 2.0*π*rₑ/(4*n)
  dt     = 0.05*dx/Uc
  println("timestep: ", dt)   # gravity wave time step

  model = CubedSphereDiscreteModel(n,2; radius=rₑ)
  quasi_geostrophic_hdg(model, order, degree,
                        ω₀, u₀, f₀,
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
