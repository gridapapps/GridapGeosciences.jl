"""
solve the non-linear shallow water equations
∂ₜu + q F^† + ∇ᵧ(Φ) = 0
∂ₜφ + ∇ᵧ⋅F = 0
F = φu
Φ = 0.5(u⋅u) + gᵣφ
q = 1/φ( ∇ᵧ^†⋅u  + f )
"""

module AmbientTransientShallowWaterTests

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


function transient_shallow_water_solver(
  extrinsic_atlas_model,
  p_fe::Int,dir::String,h::Function,vX::Function,f::Function,b::Function,
  CFL=0.1,lss=(LUSolver(),LUSolver());_i_am_main=true)

  Dc = num_cell_dims(extrinsic_atlas_model)
  lvl = nref(extrinsic_atlas_model)

  _i_am_main && println("nref = $lvl; p_fe = $p_fe; Dc = $Dc")

  ls_ode, ls_diag = lss

  ## finite element solver
  degree = 4*(p_fe+1)
  Ω_ambient = Triangulation(extrinsic_atlas_model)
  dΩ = Measure(Ω_ambient,degree)
  dΩ_error = Measure(Ω_ambient,2*degree)

  R = TestFESpace(Ω_ambient, ReferenceFE(lagrangian,Float64,p_fe+1); conformity=:H1)
  H = TransientTrialFESpace(R)

  Q = TestFESpace(Ω_ambient, ReferenceFE(lagrangian,Float64,p_fe); conformity=:L2)
  P = TransientTrialFESpace(Q)

  V = TestFESpace(Ω_ambient, ReferenceFE(raviart_thomas,Float64,p_fe); conformity=:HDiv)
  U = TransientTrialFESpace(V)

  X_prog = TransientMultiFieldFESpace([U,P]) # u, p
  Y_prog = MultiFieldFESpace([V,Q]) # u, p

  X_diag = TransientMultiFieldFESpace([H,U,P]) # q, F, Φ
  Y_diag = MultiFieldFESpace([R,V,Q]) # q, F, Φ

  ## initial conditions
  u_int = interpolate(vX,U)
  h_int = interpolate(x->h(x)-b(x),P)

  xh0 = interpolate_everywhere([u_int,h_int],X_prog(0.0))
  t0 = 0.0
  tF = _tF

  ## transient weak form
  cor_cf = CellField(f,Ω_ambient)
  gravity = _g
  b = CellField(b,Ω_ambient)

  ## Construct the coriolis term on the surface: ∫( ̃f ( ̃k × ̃u  )  )dΩ
  n_surf = get_sphere_surface_normal(Ω_ambient)

  #### DIAGNOSTIC VARIABLES
  # vorticity
  ### WARNING! The skew term is a different sign to the parametric version
  resq(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( q*p*w  )dΩ - ∫( cor_cf*w  )dΩ + ∫(  u⋅( n_surf × ∇(w) )  )dΩ

  # mass flux
  resF(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( F⋅v  )dΩ - ∫( p*(u⋅v)   )dΩ

  # Bernoulli potential
  resΦ(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( Φ*ψ  )dΩ - ∫( gravity*(p+b)*ψ  )dΩ - ∫( 0.5*( u⋅u )*ψ  )dΩ

  res_y(t,((u,p),(q,F,Φ)),(w,v,ψ)) = resq(((u,p),(q,F,Φ)),(w,v,ψ)) + resF(((u,p),(q,F,Φ)),(w,v,ψ)) + resΦ(((u,p),(q,F,Φ)),(w,v,ψ))
  jac_y(t,((u,p),(q,F,Φ)),(dq,dF,dΦ),(w,v,ψ)) = ∫( dq*p*w  )dΩ + ∫( dF⋅v )dΩ + ∫( dΦ*ψ  )dΩ


  #### PROGNOSTIC VARIABLES

  # equation for depth and velocity:
  mass(t,(dut,dpt),(v,r)) = ∫( dut⋅v  )dΩ + ∫( dpt*r )dΩ

  res_p(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = ∫( r*(∇⋅F) )dΩ
  res_u(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = (
                                  ∫( q*( ( n_surf × F)⋅v)  )dΩ
                                + ∫( -τ*( (q-q0)/dt )*( ( n_surf × F)⋅v)   )dΩ
                                + ∫( -τ*(u⋅∇(q))*( ( n_surf × F)⋅v)   )dΩ
                                - ∫( Φ*(∇⋅v) )dΩ
                    )

  res_x(t,((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = res_u(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) + res_p(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0))
  jac_x(t,((u,p),(q,F,Φ)),(du,dp),(v,r),(q0,F0,Φ0)) = ∫( -τ*(du⋅∇(q))*( ( n_surf × F)⋅v)   )dΩ
  jac_xt(t,((u,p),(q,F,Φ)),(dut,dpt),(v,r),(q0,F0,Φ0)) =  ∫( dut⋅v )dΩ + ∫( dpt*r )dΩ


  opT = TransientSemilinearFEOperator(mass,res_x,(jac_x,jac_xt),X_prog,Y_prog)
  opFE = FEOperator(res_y,jac_y,X_diag,Y_diag)
  opDAE = DAEFEOperator(opT,opFE,ls_diag)

  # transient parameters
  _dt = dx(extrinsic_atlas_model)*CFL/(p_fe*sqrt(gravity*_H_0))
  nsteps = tF/ _dt
  dt = tF/floor(nsteps)
  τ = dt/2

  # solve with SSP RK 3
  ode_solver = RungeKutta(ls_ode,ls_ode,dt,:EXRK_SSP_3_3)
  solT  = solve(ode_solver,opDAE,t0,_tF,xh0)

  ## iterate solution
  it = iterate(solT)
  xhF = xh0
  time_F = t0

  counter = 1
  while !isnothing(it)
    data, state = it
    t, xh = data

    xhF = xh
    time_F = t
    _i_am_main && println("t = ", t)

    counter = counter + 1
    it = iterate(solT, state)
  end

  uh0,ph0 = xh0
  uhF,phF = xhF

  time_F - tF

  _e = uh0 - uhF
  e_u =  sqrt(sum(∫( _e⋅_e )dΩ_error))

  _e = ph0 - phF
  e_p = sqrt(sum(∫( _e*_e )dΩ_error))

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
  vX = sphere_tangent_vec_component(u₀(0.0))
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
