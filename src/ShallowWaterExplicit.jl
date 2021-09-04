function shallow_water_explicit_time_step!(model, dΩ, dω, f, g, h₁, u₁, hₘ, uₘ, hₚ, uₚ, RTMMchol, L2MMchol, dt, leap_frog, τ, Q, V, R, S, h₂, u₂, ϕ, F)
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
  # dΩ         : measure of the elements

  n = get_normal_vector(model)
  # explicit step for provisional velocity, uₚ
  dt1 = dt
  if leap_frog
    dt1 = 2.0*dt
  end

  # 1.1: the mass flux
  b₁(v)  = ∫(v⋅u₁*h₁)*dΩ
  rhs1   = assemble_vector(b₁, V)
  copy!(get_free_dof_values(F), rhs1)
  ldiv!(RTMMchol, get_free_dof_values(F))
  # 1.2: the bernoulli function
  b₂(q)  = ∫(q*(0.5*u₁⋅u₁ + g*h₁))*dΩ
  rhs2   = assemble_vector(b₂, Q)
  copy!(get_free_dof_values(ϕ), rhs2)
  ldiv!(L2MMchol, get_free_dof_values(ϕ))
  # 1.3: the potential vorticity
  a₁(r,s) = ∫(s*h₁*r)dΩ
  c₁(s)   = ∫(perp(∇(s),n)⋅(u₁) + s*f)dΩ
  H1h     = assemble_matrix(a₁, R, S)
  rhs_q₁  = assemble_vector(c₁, S)
  op      = AffineFEOperator(R, S, H1h, rhs_q₁)
  q₁      = solve(op)
  # 1.4: solve for the provisional velocity
  b₃(v)  = ∫(v⋅uₘ - dt1*(q₁ - τ*u₁⋅∇(q₁))*(v⋅⟂(F,n)))dΩ + ∫(dt1*DIV(v)*ϕ)*dω
  rhs3   = assemble_vector(b₃, V)
  copy!(get_free_dof_values(uₚ), rhs3)
  ldiv!(RTMMchol, get_free_dof_values(uₚ))
  # 1.5: solve for the provisional depth
  b₄(q)  = ∫(q*hₘ)dΩ - ∫(dt1*q*DIV(F))*dω
  rhs4   = assemble_vector(b₄, Q)
  copy!(get_free_dof_values(hₚ), rhs4)
  ldiv!(L2MMchol, get_free_dof_values(hₚ))

  # 2.1: the mass flux
  b₅(v)  = ∫(v⋅u₁*(2.0*h₁ + hₚ)/6.0 + v⋅uₚ*(h₁ + 2.0*hₚ)/6.0)*dΩ
  rhs5   = assemble_vector(b₅, V)
  copy!(get_free_dof_values(F), rhs5)
  ldiv!(RTMMchol, get_free_dof_values(F))
  # 2.2: the bernoulli function
  b₆(q)  = ∫(q*((u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/6.0 + 0.5*g*(h₁ + hₚ)))*dΩ
  rhs6   = assemble_vector(b₆, Q)
  copy!(get_free_dof_values(ϕ), rhs6)
  ldiv!(L2MMchol, get_free_dof_values(ϕ))
  # 2.3: the potential vorticity
  a₂(r,s) = ∫(s*hₚ*r)dΩ
  c₂(s)   = ∫(perp(∇(s),n)⋅(uₚ) + s*f)dΩ
  H2h     = assemble_matrix(a₂, R, S)
  rhs_q₂  = assemble_vector(c₂, S)
  op      = AffineFEOperator(R, S, H2h, rhs_q₂)
  q₂      = solve(op)
  # 2.4: solve for the final velocity
  b₇(v)  = ∫(v⋅u₁ - 0.5*dt*(q₁ - τ*u₁⋅∇(q₁) + q₂ - τ*uₚ⋅∇(q₂))*(v⋅⟂(F,n)))dΩ + ∫(dt*DIV(v)*ϕ)*dω
  rhs7   = assemble_vector(b₇, V)
  copy!(get_free_dof_values(u₂), rhs7)
  ldiv!(RTMMchol, get_free_dof_values(u₂))
  # 2.5: solve for the final depth
  b₈(q)  = ∫(q*h₁)dΩ - ∫(dt*q*DIV(F))*dω
  rhs8   = assemble_vector(b₈, Q)
  copy!(get_free_dof_values(h₂), rhs8)
  ldiv!(L2MMchol, get_free_dof_values(h₂))
end

function shallow_water_time_stepper(model, order, degree, h₀, u₀, f₀, g, nstep, diag_freq, dump_freq, dt, τ)
  # Forward integration of the shallow water equations using a supplied method
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

  # assemble the mass matrices
  amm(a,b) = ∫(a⋅b)dΩ
  H1MM = assemble_matrix(amm, R, S)
  RTMM = assemble_matrix(amm, U, V)
  L2MM = assemble_matrix(amm, P, Q)
  H1MMchol = lu(H1MM)
  RTMMchol = lu(RTMM)
  L2MMchol = lu(L2MM)

  # Project the initial conditions onto the trial spaces
  b₁(q)   = ∫(q*h₀)dΩ
  rhs1    = assemble_vector(b₁, Q)
  hn      = FEFunction(Q, copy(rhs1))
  ldiv!(L2MMchol, get_free_dof_values(hn))

  b₂(v)   = ∫(v⋅u₀)dΩ
  rhs2    = assemble_vector(b₂, V)
  un      = FEFunction(V, copy(rhs2))
  ldiv!(RTMMchol, get_free_dof_values(un))

  b₃(s)   = ∫(s*f₀)*dΩ
  rhs3    = assemble_vector(b₃, S)
  f       = FEFunction(S, copy(rhs3))
  ldiv!(H1MMchol, get_free_dof_values(f))

  # initialise the diagnostics arrays
  mass = Vector{Float64}(undef, nstep)
  vort = Vector{Float64}(undef, nstep)
  kin  = Vector{Float64}(undef, nstep)
  pot  = Vector{Float64}(undef, nstep)
  pow  = Vector{Float64}(undef, nstep)

  # work arrays
  h_tmp = copy(get_free_dof_values(hn))
  w_tmp = copy(get_free_dof_values(f))

  hm1    = FEFunction(Q, copy(get_free_dof_values(hn)))
  um1    = FEFunction(V, copy(get_free_dof_values(un)))
  hm2    = FEFunction(Q, copy(get_free_dof_values(hn)))
  um2    = FEFunction(V, copy(get_free_dof_values(un)))
  hp     = FEFunction(Q, copy(get_free_dof_values(hn)))
  up     = FEFunction(V, copy(get_free_dof_values(un)))
  ϕ      = FEFunction(Q, copy(get_free_dof_values(hn)))
  F      = FEFunction(V, copy(get_free_dof_values(un)))
  wn     = FEFunction(S, copy(get_free_dof_values(f)))
  # first step, no leap frog integration
  shallow_water_explicit_time_step!(model, dΩ, dω, f, g, hm1, um1, hm2, um2, hp, up, RTMMchol, L2MMchol, dt, false, τ, Q, V, R, S, hn, un, ϕ, F)
  if mod(1, diag_freq) == 0
    compute_diagnostics_shallow_water!(model, dΩ, dω, S, L2MM, H1MM, H1MMchol, h_tmp, w_tmp, g, hn, un, ϕ, F, mass, vort, kin, pot, pow, 1, true, wn)
  end
  
  # subsequent steps, do leap frog integration (now that we have the state at two previous time levels)
  for istep in 2:nstep
    get_free_dof_values(hm2) .= get_free_dof_values(hm1)
    get_free_dof_values(um2) .= get_free_dof_values(um1)
    get_free_dof_values(hm1) .= get_free_dof_values(hn)
    get_free_dof_values(um1) .= get_free_dof_values(un)

    shallow_water_explicit_time_step!(model, dΩ, dω, f, g, hm1, um1, hm2, um2, hp, up, RTMMchol, L2MMchol, dt, true, τ, Q, V, R, S, hn, un, ϕ, F)
    if mod(istep, diag_freq) == 0
      compute_diagnostics_shallow_water!(model, Ω, dΩ, dω, S, L2MM, H1MM, H1MMchol, h_tmp, w_tmp, g, hn, un, ϕ, F, mass, vort, kin, pot, pow, istep, true, wn)
    end
    if mod(istep, dump_freq) == 0
      writevtk(Ω,"local/shallow_water_exp_n=$(istep)",cellfields=["hn"=>hn, "un"=>un, "wn"=>wn])
    end
  end

  hn, un
end
