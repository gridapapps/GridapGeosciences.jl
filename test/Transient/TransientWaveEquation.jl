"""
solve the linearised wave equation in
∂ₜu + ∇ᵧ(φ) = 0
∂ₜφ + ∇ᵧ⋅u = 0
"""

module TransientWaveEquationTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapDistributed
using GridapGeosciences
using GridapP4est
using Test


## initial conditions
function vecX(x)
  zero(x)
end

function depth(XYZ)
  1.0 + 0.01*exp(-5*((1-XYZ[1])^2+(0-XYZ[2])^2+(0-XYZ[3])^2))
end

function transient_wave_solver(atlas_model::Union{<:DiscreteModel{2,2},<:GridapDistributed.DistributedDiscreteModel{2,2}},
  p_fe::Int,dir::String,h::Function,vX::Function,CFL=0.1,ls=LUSolver(),tF=2*π;_i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)

  _i_am_main && println("nref = $lvl; p_fe = $p_fe; Dc = $Dc")

  ## finite element solver
  degree = 5*(p_fe+1)
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  Q = TestFESpace(Ω_atlas, ReferenceFE(lagrangian,Float64,p_fe); conformity=:L2)
  P = TransientTrialFESpace(Q)

  V = TestFESpace(Ω_atlas, ReferenceFE(raviart_thomas,Float64,p_fe); conformity=:HDiv)
  U = TransientTrialFESpace(V)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  ## initial conditions
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  meas_cf = sqrt∘det∘metric_cf
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  h_cf = h∘ambient_map_cf
  u_cf = meas_cf*((pinvJ∘covariant_basis_cf)⋅(vX∘ambient_map_cf))
  xh0 = interpolate([u_cf,h_cf],X)
  t0 = 0.0


  ## transient weak form
  mass(t, (dtu,dtp), (v,q)) = ∫( (v⋅ (metric_cf⋅dtu))*(1.0/meas_cf) )dΩ  + ∫( (q*dtp)*meas_cf )dΩ
  res(t,(u,p),(v,q)) =  ∫( q*(∇⋅u) )dΩ - ∫( p*(∇⋅v) )dΩ
  jac(t,(u,p),(du,dp),(v,q)) = res(t,(du,dp),(v,q))
  jac_t(t,(u,p),(dut,dpt),(v,q)) =  ∫( (dut⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ + ∫( (dpt*q)*meas_cf )dΩ

  opT = TransientSemilinearFEOperator(mass, res,(jac,jac_t), X, Y, constant_mass=true)

  # transient parameters
  _dt = dx(atlas_model)*CFL/p_fe
  nsteps = tF/ _dt
  dt = tF/floor(nsteps)

  # solve with SSP RK 3
  solver = RungeKutta(ls, ls, dt, :EXRK_SSP_3_3)
  solT = solve(solver, opT, t0, tF, xh0)

  ## iterate solution
  it = iterate(solT)
  xhF = xh0

  counter = 1

  while !isnothing(it)
    data, state = it
    t, xh = data
    xhF = xh

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

  ls = LUSolver()
  dir = @__DIR__
  p = 1
  CFL = 0.1
  tF = 2*π

  ## the error test here is for 3 levels of refinement only
  lvl = nref(model)
  @check lvl == 3

  eu, ep, = transient_wave_solver(model,
  p,dir,depth,vecX,CFL,ls,tF;_i_am_main=_i_am_main)

  @test isapprox(eu,0.001602838116424487;rtol=1e-2,atol=1e-3)
  @test isapprox(ep,0.010030006030668944;rtol=1e-2,atol=1e-3)
end



end # module
