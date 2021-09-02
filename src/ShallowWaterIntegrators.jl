function shallow_water_explicit_time_step(model, order, dΩ, dω, qₖ, wₖ, f, g, h₁, u₁, hₘ, uₘ, RTMM, L2MM, dt, leap_frog, τ, P, Q, U, V, R, S)
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

  n = get_normal_vector(model)
  # explicit step for provisional velocity, uₚ
  dt1 = dt
  if leap_frog
    dt1 = 2.0*dt
  end

  # 1.1: the mass flux
  b₁(v)  = ∫(v⋅u₁*h₁)*dΩ
  rhs1   = assemble_vector(b₁, V)
  op     = AffineFEOperator(U, V, RTMM, rhs1)
  F      = solve(op)
  # 1.2: the bernoulli function
  b₂(q)  = ∫(q*(0.5*u₁⋅u₁ + g*h₁))*dΩ
  rhs2   = assemble_vector(b₂, Q)
  op     = AffineFEOperator(P, Q, L2MM, rhs2)
  ϕ      = solve(op)
  # 1.3: the potential vorticity
  q₁     = diagnose_potential_vorticity(model, order, dΩ, qₖ, wₖ, f, h₁, u₁, U, V, R, S)
  # 1.4: solve for the provisional velocity
  b₃(v)  = ∫(v⋅uₘ - dt1*(q₁ - τ*u₁⋅∇(q₁))*(v⋅⟂(F,n)))dΩ + ∫(dt1*DIV(v)*ϕ)*dω
  rhs3   = assemble_vector(b₃, V)
  op     = AffineFEOperator(U, V, RTMM, rhs3)
  uₚ     = solve(op)
  # 1.5: solve for the provisional depth
  b₄(q)  = ∫(q*hₘ)dΩ - ∫(dt1*q*DIV(F))*dω
  rhs4   = assemble_vector(b₄, Q)
  op     = AffineFEOperator(P, Q, L2MM, rhs4)
  hₚ     = solve(op)

  # 2.1: the mass flux
  b₅(v)  = ∫(v⋅u₁*(2.0*h₁ + hₚ)/6.0 + v⋅uₚ*(h₁ + 2.0*hₚ)/6.0)*dΩ
  rhs5   = assemble_vector(b₅, V)
  op     = AffineFEOperator(U, V, RTMM, rhs5)
  F      = solve(op)
  # 2.2: the bernoulli function
  b₆(q)  = ∫(q*((u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/6.0 + 0.5*g*(h₁ + hₚ)))*dΩ
  rhs6   = assemble_vector(b₆, Q)
  op     = AffineFEOperator(P, Q, L2MM, rhs6)
  ϕ      = solve(op)
  # 2.3: the potential vorticity
  q₂     = diagnose_potential_vorticity(model, order, dΩ, qₖ, wₖ, f, hₚ, uₚ, U, V, R, S)
  # 2.4: solve for the final velocity
  b₇(v)  = ∫(v⋅u₁ - 0.5*dt*(q₁ - τ*u₁⋅∇(q₁) + q₂ - τ*uₚ⋅∇(q₂))*(v⋅⟂(F,n)))dΩ + ∫(dt*DIV(v)*ϕ)*dω
  rhs7   = assemble_vector(b₇, V)
  op     = AffineFEOperator(U, V, RTMM, rhs7)
  u₂     = solve(op)
  # 2.5: solve for the final depth
  b₈(q)  = ∫(q*h₁)dΩ - ∫(dt*q*DIV(F))*dω
  rhs8   = assemble_vector(b₈, Q)
  op     = AffineFEOperator(P, Q, L2MM, rhs8)
  h₂     = solve(op)

  h₂, u₂, ϕ, F
end

function shallow_water_time_stepper(model, order, Ω, dΩ, dω, qₖ, wₖ, f, g, hn, un, dt, nstep, dump_freq, τ, P, Q, U, V, R, S, method)
  # Forward integration of the shallow water equations using a supplied method

  # assemble the mass matrices
  amm(a,b) = ∫(a⋅b)dΩ
  H1MM = assemble_matrix(amm, R, S)
  RTMM = assemble_matrix(amm, U, V)
  L2MM = assemble_matrix(amm, P, Q)

  # initialise the diagnostics arrays
  mass = Vector{Float64}(undef, nstep)
  vort = Vector{Float64}(undef, nstep)
  kin  = Vector{Float64}(undef, nstep)
  pot  = Vector{Float64}(undef, nstep)
  pow  = Vector{Float64}(undef, nstep)

  # work arrays
  h_tmp = copy(get_free_dof_values(hn))
  w_tmp = copy(get_free_dof_values(f))

  # first step, no leap frog integration
  hm1          = FEFunction(Q, copy(get_free_dof_values(hn)))
  um1          = FEFunction(V, copy(get_free_dof_values(un)))
  hm2          = FEFunction(Q, copy(get_free_dof_values(hn)))
  um2          = FEFunction(V, copy(get_free_dof_values(un)))
  hm1_dof      = get_free_dof_values(hm1)
  hm2_dof      = get_free_dof_values(hm2)
  um1_dof      = get_free_dof_values(um1)
  um2_dof      = get_free_dof_values(um2)
  hn, un, ϕ, F = method(model, order, dΩ, dω, qₖ, wₖ, f, g, hm1, um1, hm2, um2, RTMM, L2MM, dt, false, τ, P, Q, U, V, R, S)

  wn = compute_diagnostics_shallow_water!(model, order, Ω, dΩ, dω, qₖ, wₖ, U, V, R, S, L2MM, H1MM, h_tmp, w_tmp, g, hn, un, ϕ, F, mass, vort, kin, pot, pow, 1, true)
  
  # subsequent steps, do leap frog integration (now that we have the state at two previous time levels)
  for istep in 2:nstep
    hm2_dof     .= hm1_dof
    um2_dof     .= um1_dof
    hm1_dof     .= hn_dof
    um1_dof     .= un_dof
    hn, un, ϕ, F = method(model, order, dΩ, dω, qₖ, wₖ, f, g, hm1, um1, hm2, um2, RTMM, L2MM, dt, true, τ, P, Q, U, V, R, S)

    wn = compute_diagnostics_shallow_water!(model, order, Ω, dΩ, dω, qₖ, wₖ, U, V, R, S, L2MM, H1MM, h_tmp, w_tmp, g, hn, un, ϕ, F, mass, vort, kin, pot, pow, istep, true)
    if mod(istep, dump_freq) == 0
      writevtk(Ω,"local/shallow_water_exp_n=$(istep)",cellfields=["hn"=>hn, "un"=>un, "wn"=>wn])
    end
  end

  hn, un
end
