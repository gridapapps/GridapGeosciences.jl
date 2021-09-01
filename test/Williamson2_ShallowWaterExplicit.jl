module Williamson2_ShallowWaterExplicit

using Gridap
using GridapGeosciences
using FillArrays
using WriteVTK

# Solves the steady state Williamson2 test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a modified coriolis term that exactly balances
# the potential gradient term to achieve a steady state
# reference:
# D. L. Williamson, J. B. Drake, J. J.HackRüdiger Jakob, P. N.Swarztrauber, (1992)
# J Comp. Phys. 102 211-224

# Constants of the Williamson2 test case
const rₑ = 6371220.0          # earth's radius
const g  = 9.80616            # gravitational acceleration
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

function grad_perp(α)
  grad_α = ∇(α)
  A = TensorValue{2,2}(0, -1, 1, 0)
  A⋅grad_α
end

function grad_perp_ref_domain(model, order, Ω, R, S, U, V, u, qₖ, wₖ)
  # ∫∇⟂α⋅udΩ
  # α: Test function,  ∈ H₁(Ω)
  # u: velocity,       ∈ H(div,Ω)
  #
  # arguments:
  # model: geometry of the domain
  # order: polynomial degree of the test functions
  # Ω:     the domain
  # U:     trial functions
  # V:     test functions
  # qₖ:    quadrature points
  # wₖ:    quadrature weights    # Evaluate the Jacobian at the quadrature points
  ξₖ  = get_cell_map(model)
  Jt  = lazy_map(Broadcasting(∇), ξₖ)
  Jq  = lazy_map(evaluate, Jt, qₖ)

  # H₁ test functions
  reffe_lgn    = ReferenceFE(lagrangian, Float64, order+1)
  basis, reffe_args, reffe_kwargs = reffe_lgn
  lgn          = ReferenceFE(model, basis, reffe_args...; reffe_kwargs...)
  α            = Gridap.ReferenceFEs.get_shapefuns(lgn[1])
  αₖ           = Fill(α, num_cells(model))
  # map the H₁ test functions into H(div) via the ∇⟂ operator in the reference element
  grad_perp_αₖ = lazy_map(Broadcasting(grad_perp), αₖ)

  # get the index of the panel for each element
  fl       = get_face_labeling(model)
  panel_id = fl.d_to_dface_to_entity[3]
  # grad perp turns out to have reverse orientation in half the panels
  panel_flip = ones(Bool, (length(grad_perp_αₖ), length(grad_perp_αₖ[1])))
  for i in 1:length(grad_perp_αₖ)
    panel_flip[i,:] .= true
    if panel_id[i] == 25 || panel_id[i] == 21 || panel_id[i] == 24
      panel_flip[i,:] .= false
    end
  end
  fpanel_flip   = lazy_map(Broadcasting(Gridap.Fields.ConstantField), panel_flip)
  # pull back the H(div) test functions into global coordinates
  m             = Gridap.ReferenceFEs.ContraVariantPiolaMap()
  sqrt_det_JtxJ = lazy_map(Operation(Gridap.TensorValues.meas), Jt)
  ϕₖs   = lazy_map(Broadcasting(Operation(m)),
                                grad_perp_αₖ,
                                Jt,
                                sqrt_det_JtxJ,
                                fpanel_flip)
  uq    = lazy_map(evaluate, Gridap.CellData.get_data(u), qₖ)
  ϕₖsq  = lazy_map(evaluate, ϕₖs, qₖ)
  intcq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(⋅), ϕₖsq, uq)
  iwqc  = lazy_map(Gridap.Fields.IntegrationMap(), intcq, wₖ, Jq)
  -1.0*iwqc
end

function diagnose_potential_vorticity(model, order, Ω, dΩ, qₖ, wₖ, f, h, u, U, V, R, S)
  # solve the system:
  #
  # ∫αhqdΩ = -∫∇⟂α⋅udΩ + ∫αfdΩ, ∀α∈ H₁(Ω)
  # where:
  #
  # q : potential vorticity; q = (∇×u + f)/h
  # f : coriolis force (∈ H₁(Ω)
  # h : fluid depth
  # u : velocity
  #
  # order      : polynomial order
  # Ω          : domain
  # dΩ         : measure of the elements
  # qₖ         : quadrature points
  # wₖ         : quadrature weights

  # the bilinear form left hand side
  r      = get_trial_fe_basis(R)
  s      = get_fe_basis(S)
  a(r,s) = ∫(s*h*r)*dΩ
  lhsdc  = a(r,s)

  # the linear form right hand side
  grad_perp = grad_perp_ref_domain(model, order, Ω, R, S, U, V, u, qₖ, wₖ)
  b(s)  = ∫(s*f)*dΩ
  rhsdc = b(s)
  # subtract the weak form curl as evaluated using the low level API
  Gridap.CellData.add_contribution!(rhsdc, get_triangulation(dΩ.quad), grad_perp)
  # assemble the right hand side
  data  = Gridap.FESpaces.collect_cell_vector(S, rhsdc)
  assem = SparseMatrixAssembler(R, S)
  rhs   = assemble_vector(assem, data)
  # assemble the left hand side
  data  = Gridap.FESpaces.collect_cell_matrix(R, S, lhsdc)
  assem = SparseMatrixAssembler(R, S)
  lhs   = assemble_matrix(assem, data)

  op = AffineFEOperator(R, S, lhs, rhs)
  q  = solve(op)
end

function assemble_rhs_vector(A, B, dc)
  data  = Gridap.FESpaces.collect_cell_vector(B, dc)
  assem = SparseMatrixAssembler(A, B)
  rhs   = assemble_vector(assem, data)
end

function shallow_water_explicit(model, order, Ω, dΩ, dω, qₖ, wₖ, f, g, h₁, u₁, hₘ, uₘ, RTMM, L2MM, dt, leap_frog, τ, P, Q, U, V, R, S)
  # energetically balanced explicit second order shallow water solver
  # reference: eqns (21-24) of
  # https://github.com/BOM-Monash-Collaborations/articles/blob/main/energetically_balanced_time_integration/EnergeticallyBalancedTimeIntegration_SW.tex
  #
  # f          : coriolis force (field)
  # g          : gravity (constant)
  # h₁         : fluid depth at current time level
  # u₁         : fluid velocity at current time level
  # hₘ         : fluid depth at previous time level (for leap-frogging the first step)
  # uₘ         : fluid velocity at previous time level (for leap-frogging the first step)
  # RTMM       : H(div) mass matrix, ∫β⋅βdΩ, ∀β∈ H(div,Ω)
  # L2MM       : L² mass matrix, ∫γγdΩ, ∀γ∈ L²(Ω)
  # dt         : time step
  # leap_frog  : do leap frog time integration for the first step (boolean)
  # order      : polynomial order
  # dΩ         : measure of the elements

  u = get_trial_fe_basis(U)
  v = get_fe_basis(V)
  p = get_trial_fe_basis(P)
  q = get_fe_basis(Q)
  n = get_normal_vector(model)

  # explicit step for provisional velocity, uₚ
  dt1 = dt
  if leap_frog
    dt1 = 2.0*dt
  end

  # 1.1: the mass flux
  b₁(v)  = ∫(v⋅u₁*h₁)*dΩ
  rhs1   = assemble_rhs_vector(U, V, b₁(v))
  op     = AffineFEOperator(U, V, RTMM, rhs1)
  F      = solve(op)
  # 1.2: the bernoulli function
  b₂(q)  = ∫(q*(0.5*u₁⋅u₁ + g*h₁))*dΩ
  rhs2   = assemble_rhs_vector(P, Q, b₂(q))
  op     = AffineFEOperator(P, Q, L2MM, rhs2)
  ϕ      = solve(op)
  # 1.3: the potential vorticity
  q₁     = diagnose_potential_vorticity(model, order, Ω, dΩ, qₖ, wₖ, f, h₁, u₁, U, V, R, S)
  # 1.4: solve for the provisional velocity
  b₃(v)  = ∫(v⋅uₘ - dt1*(q₁ - τ*u₁⋅∇(q₁))*(v⋅⟂(F,n)) + dt1*DIV(v)*ϕ)*dω
  rhs3   = assemble_rhs_vector(U, V, b₃(v))
  op     = AffineFEOperator(U, V, RTMM, rhs3)
  uₚ     = solve(op)
  # 1.5: solve for the provisional depth
  b₄(q)  = ∫(q*hₘ - dt1*q*DIV(F))*dω
  rhs4   = assemble_rhs_vector(P, Q, b₄(q))
  op     = AffineFEOperator(P, Q, L2MM, rhs4)
  hₚ     = solve(op)

  # 2.1: the mass flux
  b₅(v)  = ∫(v⋅u₁*(2.0*h₁ + hₚ)/6.0 + v⋅uₚ*(h₁ + 2.0*hₚ)/6.0)*dΩ
  rhs5   = assemble_rhs_vector(U, V, b₅(v))
  op     = AffineFEOperator(U, V, RTMM, rhs5)
  F      = solve(op)
  # 2.2: the bernoulli function
  b₆(q)  = ∫(q*((u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/6.0 + 0.5*g*(h₁ + hₚ)))*dΩ
  rhs6   = assemble_rhs_vector(P, Q, b₆(q))
  op     = AffineFEOperator(P, Q, L2MM, rhs6)
  ϕ      = solve(op)
  # 2.3: the potential vorticity
  q₂     = diagnose_potential_vorticity(model, order, Ω, dΩ, qₖ, wₖ, f, hₚ, uₚ, U, V, R, S)
  # 2.4: solve for the final velocity
  b₇(v)  = ∫(v⋅u₁ - 0.5*dt*(q₁ - τ*u₁⋅∇(q₁) + q₂ - τ*uₚ⋅∇(q₂))*(v⋅⟂(F,n)) + dt*DIV(v)*ϕ)*dω
  rhs7   = assemble_rhs_vector(U, V, b₇(v))
  op     = AffineFEOperator(U, V, RTMM, rhs7)
  u₂     = solve(op)
  # 2.5: solve for the final depth
  b₈(q)  = ∫(q*h₁ - dt*q*DIV(F))*dω
  rhs8   = assemble_rhs_vector(P, Q, b₈(q))
  op     = AffineFEOperator(P, Q, L2MM, rhs8)
  h₂     = solve(op)

  h₂, u₂, ϕ, F
end

function total_vorticity(model, order, Ω, qₖ, wₖ, R, S, U, V, H1MM, u)
  # ∫∇×udΩ
  iwqc  = grad_perp_ref_domain(model, order, Ω, R, S, U, V, u, qₖ, wₖ)
  assem = SparseMatrixAssembler(U, S)
  dc    = Gridap.CellData.DomainContribution()
  Gridap.CellData.add_contribution!(dc, Ω, iwqc)
  data  = Gridap.FESpaces.collect_cell_vector(S, dc)
  rhs   = assemble_vector(assem, data)
  op    = AffineFEOperator(R, S, H1MM, rhs)
  w     = solve(op)

  w_dof = Gridap.FESpaces.get_free_dof_values(w)
  sum(H1MM*w_dof)
end

function compute_diagnostics(model, order, Ω, dΩ, dω, qₖ, wₖ, U, V, R, S, L2MM, H1MM, g, h, u, ϕ, F, mass, vort, kin, pot, pow, step)
  mass_i = sum(L2MM*Gridap.FESpaces.get_free_dof_values(h))
  vort_i = total_vorticity(model, order, Ω, qₖ, wₖ, R, S, U, V, H1MM, u)
  kin_i  = 0.5*sum(∫(h*(u⋅u))dΩ)
  pot_i  = 0.5*g*sum(∫(h*h)dΩ)
  pow_i  = sum(∫(ϕ*DIV(F))dω)

  append!(mass, mass_i)
  append!(vort, vort_i)
  append!(kin, kin_i)
  append!(pot, pot_i)
  append!(pow, pow_i)

  # normalised conservation errors
  mass_norm = (mass_i-mass[1])/mass[1]
  vort_norm = vort_i-vort[1]
  en_norm   = (kin_i+pot_i-kin[1]-pot[1])/(kin[1]+pot[1])
  println(step, "\t", mass_norm, "\t", vort_norm, "\t", kin_i, "\t", pot_i, "\t", en_norm, "\t", pow_i)
end

function new_field(A, a)
  a_dof  = Gridap.FESpaces.get_free_dof_values(a)
  b_dof  = similar(a_dof)
  b_dof .= a_dof
  b      = FEFunction(A, b_dof)
end

function shallow_water_explicit_time_stepper(model, order, Ω, dΩ, dω, qₖ, wₖ, f, g, hn, un, dt, nstep, dump_freq, τ, P, Q, U, V, R, S)
  # assemble the mass matrices
  amm(a,b) = ∫(a⋅b)dΩ
  H1MM = assemble_matrix(amm, R, S)
  RTMM = assemble_matrix(amm, U, V)
  L2MM = assemble_matrix(amm, P, Q)

  # initialise the diagnostics arrays
  mass = zeros(0)
  vort = zeros(0)
  kin  = zeros(0)
  pot  = zeros(0)
  pow  = zeros(0)

  # first step, no leap frog integration
  hm1          = new_field(Q, hn)
  um1          = new_field(V, un)
  istep        = 1
  hn, un, ϕ, F = shallow_water_explicit(model, order, Ω, dΩ, dω, qₖ, wₖ, f, g, hm1, um1, hm1, um1, RTMM, L2MM, dt, false, τ, P, Q, U, V, R, S)

  compute_diagnostics(model, order, Ω, dΩ, dω, qₖ, wₖ, U, V, R, S, L2MM, H1MM, g, hn, un, ϕ, F, mass, vort, kin, pot, pow, istep)
  
  # subsequent steps, do leap frog integration (now that we have the state at two previous time levels)
  for istep in 2:nstep
    hm2          = new_field(Q, hm1)
    um2          = new_field(V, um1)
    hm1          = new_field(Q, hn)
    um1          = new_field(V, un)
    hn, un, ϕ, F = shallow_water_explicit(model, order, Ω, dΩ, dω, qₖ, wₖ, f, g, hm1, um1, hm2, um2, RTMM, L2MM, dt, true, τ, P, Q, U, V, R, S)

    compute_diagnostics(model, order, Ω, dΩ, dω, qₖ, wₖ, U, V, R, S, L2MM, H1MM, g, hn, un, ϕ, F, mass, vort, kin, pot, pow, istep)

    if mod(istep, dump_freq) == 0
      iwqc  = grad_perp_ref_domain(model, order, Ω, R, S, U, V, un, qₖ, wₖ)
      assem = SparseMatrixAssembler(U, S)
      dc    = Gridap.CellData.DomainContribution()
      Gridap.CellData.add_contribution!(dc, Ω, iwqc)
      data  = Gridap.FESpaces.collect_cell_vector(S, dc)
      rhs   = assemble_vector(assem, data)
      op    = AffineFEOperator(R, S, H1MM, rhs)
      wn    = solve(op)
      writevtk(Ω,"local/shallow_water_exp_n=$(istep)",cellfields=["hn"=>hn, "un"=>un, "wn"=>wn])
    end
  end

  hn, un
end

function forward_step(n)
  order = 1
  degree = 4

  model = CubedSphereDiscreteModel(n, order, radius=rₑ)

  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())
  quad_cell_point = get_cell_points(dΩ.quad)
  qₖ = Gridap.CellData.get_data(quad_cell_point)
  wₖ = dΩ.quad.cell_weight
  ξₖ = get_cell_map(Ω)

  # Setup the trial and test spaces
  reffe_rt  = ReferenceFE(raviart_thomas, Float64, order)
  V = FESpace(model, reffe_rt ; conformity=:HDiv)
  U = TrialFESpace(V)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order)
  Q = FESpace(model, reffe_lgn; conformity=:L2)
  P = TrialFESpace(Q)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order+1)
  S = FESpace(model, reffe_lgn; conformity=:H1)
  R = TrialFESpace(S)

  # Project the initial conditions onto the trial spaces
  hc      = CellField(h₀, Ω)
  a₁(p,q) = ∫(q*p)dΩ
  b₁(q)   = ∫(q*hc)dΩ
  op      = AffineFEOperator(a₁, b₁, P, Q)
  hp      = solve(op)

  uc      = CellField(u₀, Ω)
  a₂(u,v) = ∫(v⋅u)dΩ
  b₂(v)   = ∫(v⋅uc)dΩ
  op      = AffineFEOperator(a₂, b₂, U, V)
  up      = solve(op)

  fc      = CellField(f₀, Ω)
  a₃(r,s) = ∫(s*r)*dΩ
  b₃(s)   = ∫(s*fc)*dΩ
  op      = AffineFEOperator(a₃, b₃, R, S)
  fp      = solve(op)

  nstep  = 5*n
  Uc     = sqrt(g*H₀)
  dx     = 2.0*π*rₑ/(4*n)
  dt     = 0.05*dx/Uc
  println("timestep: ", dt)   # gravity wave time step
  hf, uf = shallow_water_explicit_time_stepper(model, order, Ω, dΩ, dω, qₖ, wₖ, fp, g, hp, up, dt, nstep, 20, 0.0*dt, P, Q, U, V, R, S)

  e = hc-hf
  err_h = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(hc⋅hc)*dΩ))
  e = uc-uf
  err_u = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(uc⋅uc)*dΩ))
  println("n=", n, ",\terr_u: ", err_u, ",\terr_h: ", err_h)
end

for nn in 1:4
  n = 2*2^nn
  forward_step(n)
end

end
