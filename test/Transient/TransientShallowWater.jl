"""
solve the non-linear shallow water equations
∂ₜu + q F^† + ∇ᵧ(Φ) = 0
∂ₜφ + ∇ᵧ⋅F = 0
F = φu
Φ = 0.5(u⋅u) + gᵣφ
q = 1/φ( ∇ᵧ^†⋅u  + f )
"""

module TransientShallowWaterTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapDistributed
using GridapGeosciences
using GridapP4est
using Test


include("../Geophysical/Williamson_functions.jl")

function topography(xyz)
  0.0
end

a_e = 6.37e6 # m
g = 9.8 # m/2
ω = 7.29e-5 #s^-1
T = 12*24*3600 #s
H_0 = 2.94e4/g #m
u_0 = 2*π*a_e/T #m/s
b_0 = 0.0 #m
TF = 1*(3600*24)

L = a_e
_τ = 1/ω #sqrt(a_e/g)

_a = a_e/L
_g = g*_τ^2/L
_ω = ω*_τ
_H_0 = H_0/L
_T = T/_τ
_u0 = u_0/L*_τ #2*π*_a/_T
_b0 = b_0/L
_tF = TF/_τ



function transient_shallow_water_solver(atlas_model::Union{<:DiscreteModel{2,2},<:GridapDistributed.DistributedDiscreteModel{2,2}},
  p_fe::Int,dir::String,h::Function,vX::Function,f::Function,b::Function,
  CFL=0.1,lss=(LUSolver(),LUSolver());_i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)

  _i_am_main && println("nref = $lvl; p_fe = $p_fe; Dc = $Dc")

  ls_ode, ls_diag = lss

  ## finite element solver
  degree = 4*(p_fe+1)
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  R = TestFESpace(Ω_atlas, ReferenceFE(lagrangian,Float64,p_fe+1); conformity=:H1)
  H = TransientTrialFESpace(R)

  Q = TestFESpace(Ω_atlas, ReferenceFE(lagrangian,Float64,p_fe); conformity=:L2)
  P = TransientTrialFESpace(Q)

  V = TestFESpace(Ω_atlas, ReferenceFE(raviart_thomas,Float64,p_fe); conformity=:HDiv)
  U = TransientTrialFESpace(V)

  X_prog = TransientMultiFieldFESpace([U,P]) # u, p
  Y_prog = MultiFieldFESpace([V,Q]) # u, p

  X_diag = TransientMultiFieldFESpace([H,U,P]) # q, F, Φ
  Y_diag = MultiFieldFESpace([R,V,Q]) # q, F, Φ

  # metric information
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)

  ## initial conditions
  u_cf = meas_cf*((pinvJ∘transpose∘∇(ambient_map_cf))⋅(vX∘ambient_map_cf))
  u_int = interpolate(u_cf,U)

  h_cf = h∘ambient_map_cf
  b_cf = b∘ambient_map_cf
  h_int = interpolate(h_cf-b_cf,P)

  xh0 = interpolate_everywhere([u_int,h_int],X_prog(0.0))
  t0 = 0.0
  tF = _tF

  ## transient weak form
  cor_cf = f∘ambient_map_cf
  gravity = _g
  b_cf = b∘ambient_map_cf

  #### DIAGNOSTIC VARIABLES
  # vorticity
  Aperp = [0 -1
           1 0]
  Rperp = TensorValue(Aperp)
  Rperp_cf = CellField(Rperp,Ω_atlas)


  resq(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( q*p*w*meas_cf  )dΩ - ∫( cor_cf*w*meas_cf  )dΩ - ∫( (( (Rperp_cf⋅u)⋅inv_metric_cf)⋅∇(w))*meas_cf  )dΩ

  # mass flux
  resF(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( (F⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ - ∫( p*(u⋅(metric_cf⋅v))*(1.0/meas_cf)   )dΩ

  # Bernoulli potential
  resΦ(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( Φ*ψ*meas_cf  )dΩ - ∫( gravity*(p+b_cf)*ψ*meas_cf  )dΩ - ∫( 0.5*( u ⋅(metric_cf⋅u) )ψ*(1.0/meas_cf)  )dΩ

  res_y(t,((u,p),(q,F,Φ)),(w,v,ψ)) = resq(((u,p),(q,F,Φ)),(w,v,ψ)) + resF(((u,p),(q,F,Φ)),(w,v,ψ)) + resΦ(((u,p),(q,F,Φ)),(w,v,ψ))
  jac_y(t,((u,p),(q,F,Φ)),(dq,dF,dΦ),(w,v,ψ)) = ∫( dq*p*w*meas_cf  )dΩ + ∫( (dF⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ + ∫( dΦ*ψ*meas_cf  )dΩ


  #### PROGNOSTIC VARIABLES

  # equation for depth and velocity:
  mass(t,(dut,dpt),(v,r)) = ∫( (dut⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ + ∫( (dpt*r)*meas_cf )dΩ

  res_p(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = ∫( r*(∇⋅F) )dΩ
  res_u(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = (  ∫( ( q*( (Rperp_cf⋅ F)⋅v))  )dΩ
                                + ∫( -τ*( (q-q0)/dt )*( (Rperp_cf⋅ F)⋅v)   )dΩ
                                + ∫( -τ*(u⋅∇(q))*( (Rperp_cf⋅ F)⋅v)*(1.0/meas_cf)   )dΩ
                                - ∫( Φ*(∇⋅v) )dΩ
                    )

  res_x(t,((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = res_u(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) + res_p(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0))
  jac_x(t,((u,p),(q,F,Φ)),(du,dp),(v,r),(q0,F0,Φ0)) = ∫( -τ*(du⋅∇(q))*( (Rperp_cf⋅ F)⋅v)*(1.0/meas_cf)   )dΩ
  jac_xt(t,((u,p),(q,F,Φ)),(dut,dpt),(v,r),(q0,F0,Φ0)) =  ∫( (dut⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ + ∫( (dpt*r)*meas_cf )dΩ


  opT = TransientSemilinearFEOperator(mass,res_x,(jac_x,jac_xt),X_prog,Y_prog)
  opFE = FEOperator(res_y,jac_y,X_diag,Y_diag)
  opDAE = DAEFEOperator(opT,opFE,ls_diag)

  # transient parameters
  _dt = dx(atlas_model)*CFL/(p_fe*sqrt(gravity*_H_0))
  nsteps = tF/ _dt
  dt = tF/floor(nsteps)
  τ = dt/2

  # solve with SSP RK 3
  ode_solver = RungeKutta(ls_ode,ls_ode,dt,:EXRK_SSP_3_3)
  solT  = solve(ode_solver,opDAE,t0,_tF,xh0)

  ## iterate solution
  it = iterate(solT)
  xhF = xh0

  counter = 1
  while !isnothing(it)
    data, state = it
    t, xh = data

    xhF = xh
    _i_am_main && println("t = ", t)

    counter = counter + 1
    it = iterate(solT, state)
  end

  uh0,ph0 = xh0
  uhF,phF = xhF

  _e = uh0 - uhF
  e_u =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*(1.0/meas_cf) )dΩ_error))

  _e = ph0 - phF
  e_p = sqrt(sum(∫( (_e*_e)*meas_cf )dΩ_error))

  _i_am_main && println("eu = $e_u, ep = $e_p")
  return e_u,e_p,false


end


################################################################################
#### Auto convergence test
################################################################################
function main(model;_i_am_main=true)

  lss = (LUSolver(),LUSolver())
  dir = @__DIR__
  p = 1
  CFL = 0.1
  h = h₀(0.0)
  vX = tangent_vec(u₀(0.0))
  f = f₀(0.0)
  b = topography

  ## the error test here is for 3 levels of refinement only
  lvl = nref(model)
  @check lvl == 3

  eu, ep, = transient_shallow_water_solver(model,p,dir,h,vX,f,b,CFL,lss;_i_am_main=_i_am_main)

  @test isapprox(eu,0.0035085422284689785,;rtol=1e-2,atol=1e-3)
  @test isapprox(ep,3.5206285360509003e-6;rtol=1e-2,atol=1e-3)


end




end # module
