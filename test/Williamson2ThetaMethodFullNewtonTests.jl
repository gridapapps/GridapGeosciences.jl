module Williamsom2ThetaMethodFullNewtonTests

using Test
using Gridap
using GridapGeosciences

# Solves the steady state Williamson2 test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a modified coriolis term that exactly balances
# the potential gradient term to achieve a steady state
# reference:
# D. L. Williamson, J. B. Drake, J. J.HackRüdiger Jakob, P. N.Swarztrauber, (1992)
# J Comp. Phys. 102 211-224

include("Williamson2InitialConditions.jl")

l2_err_u = [0.011370921987771046 , 0.0029356344096698353 ]
l2_err_h = [0.005606685579166809, 0.001451999866505571 ]

order=1
degree=4
θ=0.0
for i in 1:2
  n      = 2*2^i
  nstep  = 5*n
  Uc     = sqrt(g*H₀)
  dx     = 2.0*π*rₑ/(4*n)
  dt     = 0.25*dx/Uc
  println("timestep: ", dt)   # gravity wave time step
  T      = dt*nstep
  model = CubedSphereDiscreteModel(n; radius=rₑ)
  hf, uf = shallow_water_theta_method_full_newton_time_stepper(model, order, degree,
                                                               h₀, u₀, f₀, g, θ, T, nstep;
                                                               write_solution=false,
                                                               write_solution_freq=5,
                                                               write_diagnostics=true,
                                                               write_diagnostics_freq=1,
                                                               dump_diagnostics_on_screen=true)

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



# RT = ReferenceFE(raviart_thomas,Float64,order)
# DG = ReferenceFE(lagrangian,Float64,order)
# CG = ReferenceFE(lagrangian,Float64,order+1)
# V  = FESpace(model,RT; conformity=:Hdiv) # Velocity and mass flux FE space
# Q  = FESpace(model,DG; conformity=:L2)   # Fluid depth FE space
# S  = FESpace(model,CG; conformity=:H1)   # Potential vorticity FE space
# U  = TrialFESpace(V)
# P  = TrialFESpace(Q)
# R  = TrialFESpace(S)

# Y = MultiFieldFESpace([V,Q,S,V])         # Monolithic FE space
# X = MultiFieldFESpace([U,P,R,U])

# Ω  = Triangulation(model)
# n  = get_normal_vector(model)
# dΩ = Measure(Ω,degree)
# dω = Measure(Ω,degree,ReferenceDomain())

# dt  = T/N
# τ   = dt/2


# #un = interpolate_everywhere(u₀_rognes,U)
# #hn = interpolate_everywhere(h₀_rognes,P)
# #b  = interpolate_everywhere(topography,P)

# a(u,v)=∫(v⋅u)dΩ
# l(v)=∫(v⋅u₀)dΩ
# un=solve(AffineFEOperator(a,l,U,V))

# a(u,v)=∫(v*u)dΩ
# l(v)=∫(v*h₀)dΩ
# hn=solve(AffineFEOperator(a,l,P,Q))

# a(u,v)=∫(v*u)dΩ
# l(v)=∫(v*f₀)dΩ
# fn=solve(AffineFEOperator(a,l,R,S))

# # Williamsom2

# function residual((Δu,Δh,qvort,F),(v,q,s,v2))
#   uiΔu  = un#+Δu #Operation(ui)(Δu,un)
#   hiΔh  = hn#+Δh #Operation(hi)(Δh,hn)
#   hbiΔh = hn#+Δh #Operation(hbi)(Δh,hn,b)
#   #∫((1.0/dt)*v⋅Δu
#   ∫(-(∇⋅(v))*(g*hbiΔh + 0.5*uiΔu⋅uiΔu)+
#        (qvort-τ*(uiΔu⋅∇(qvort)))*(v⋅⟂(F,n)))dΩ +   # eq1
#     #(1.0/dt)*q*Δh)dΩ
#      ∫(q*(divergence(F)))dΩ +  # eq2
#   ∫(s*qvort*hiΔh + ⟂(∇(s),n)⋅uiΔu - s*fn +   # eq3
#     v2⋅(F-hiΔh*uiΔu))dΩ                      # eq4
# end

# ΔuΔhqF=uhqF₀(q₀(un,hn,fn,R,S,n,dΩ),F₀(un,hn,U,V,dΩ),X,Y,dΩ)
# #Δu,Δh,q,F = ΔuΔhqF
# dY = get_fe_basis(Y)
# residualΔuΔhqF=residual(ΔuΔhqF,dY)
# @time r=assemble_vector(residualΔuΔhqF,Y)
# println(norm(r))

# rh=FEFunction(X,r)
# ruh,rhh,rqh,rFh=rh

# writevtk(Triangulation(model),"kk",cellfields=["un"=>un,"un_"=>u₀,
#                                                "hn"=>hn,"hn_"=>h₀,
#                                                "fn"=>fn,"fn_"=>f₀])
# writevtk(Triangulation(model),"rr",cellfields=["ruh"=>ruh,"rhh"=>rhh,"rqh"=>rqh,"rFh"=>rFh])


end # module
