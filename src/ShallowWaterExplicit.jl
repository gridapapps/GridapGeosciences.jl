function shallow_water_explicit_time_step!(h₂, u₂, ϕ, F, q₁, q₂, model, dΩ, dω, f, g, h₁, u₁, hₘ, uₘ, hₚ, uₚ, RTMMchol, L2MMchol, H1h, H1hchol, dt, leap_frog, τ, Q, V, R, S)
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
  b₁(v)  = ∫(v⋅u₁*h₁)dΩ
  Gridap.FESpaces.assemble_vector!(b₁, get_free_dof_values(F), V)
  ldiv!(RTMMchol, get_free_dof_values(F))
  # 1.2: the bernoulli function
  b₂(q)  = ∫(q*(0.5*u₁⋅u₁ + g*h₁))*dΩ
  Gridap.FESpaces.assemble_vector!(b₂, get_free_dof_values(ϕ), Q)
  ldiv!(L2MMchol, get_free_dof_values(ϕ))
  # 1.3: the potential vorticity
  a₁(r,s) = ∫(s*h₁*r)dΩ
  c₁(s)   = ∫(perp(n,∇(s))⋅(u₁) + s*f)dΩ
  Gridap.FESpaces.assemble_matrix_and_vector!(a₁, c₁, H1h, get_free_dof_values(q₁), R, S)
  lu!(H1hchol, H1h)
  ldiv!(H1hchol, get_free_dof_values(q₁))
  # 1.4: solve for the provisional velocity
  b₃(v)  = ∫(v⋅uₘ - dt1*(q₁ - τ*u₁⋅∇(q₁))*(v⋅⟂(F,n)))dΩ + ∫(dt1*DIV(v)*ϕ)dω
  Gridap.FESpaces.assemble_vector!(b₃, get_free_dof_values(uₚ), V)
  ldiv!(RTMMchol, get_free_dof_values(uₚ))
  # 1.5: solve for the provisional depth
  b₄(q)  = ∫(q*hₘ)dΩ - ∫(dt1*q*DIV(F))dω
  Gridap.FESpaces.assemble_vector!(b₄, get_free_dof_values(hₚ), Q)
  ldiv!(L2MMchol, get_free_dof_values(hₚ))

  # 2.1: the mass flux
  b₅(v)  = ∫(v⋅u₁*(2.0*h₁ + hₚ)/6.0 + v⋅uₚ*(h₁ + 2.0*hₚ)/6.0)dΩ
  Gridap.FESpaces.assemble_vector!(b₅, get_free_dof_values(F), V)
  ldiv!(RTMMchol, get_free_dof_values(F))
  # 2.2: the bernoulli function
  b₆(q)  = ∫(q*((u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/6.0 + 0.5*g*(h₁ + hₚ)))dΩ
  Gridap.FESpaces.assemble_vector!(b₆, get_free_dof_values(ϕ), Q)
  ldiv!(L2MMchol, get_free_dof_values(ϕ))
  # 2.3: the potential vorticity
  a₂(r,s) = ∫(s*hₚ*r)dΩ
  c₂(s)   = ∫(perp(n,∇(s))⋅(uₚ) + s*f)dΩ
  Gridap.FESpaces.assemble_matrix_and_vector!(a₂, c₂, H1h, get_free_dof_values(q₂), R, S)
  lu!(H1hchol, H1h)
  ldiv!(H1hchol, get_free_dof_values(q₂))
  # 2.4: solve for the final velocity
  b₇(v)  = ∫(v⋅u₁ - 0.5*dt*(q₁ - τ*u₁⋅∇(q₁) + q₂ - τ*uₚ⋅∇(q₂))*(v⋅⟂(F,n)))dΩ + ∫(dt*DIV(v)*ϕ)dω
  Gridap.FESpaces.assemble_vector!(b₇, get_free_dof_values(u₂), V)
  ldiv!(RTMMchol, get_free_dof_values(u₂))
  # 2.5: solve for the final depth
  b₈(q)  = ∫(q*h₁)dΩ - ∫(dt*q*DIV(F))dω
  Gridap.FESpaces.assemble_vector!(b₈, get_free_dof_values(h₂), Q)
  ldiv!(L2MMchol, get_free_dof_values(h₂))
end

function shallow_water_time_stepper(model, order, degree, h₀, u₀, f₀, g, dt, τ, nstep, out_dir, diag_freq=1, dump_freq=100)
  # Forward integration of the shallow water equations
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())

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

  # work arrays
  h_tmp = copy(get_free_dof_values(hn))
  w_tmp = copy(get_free_dof_values(f))
  # build the potential vorticity lhs operator once just to initialise
  bmm(a,b) = ∫(a*hn*b)dΩ
  H1h      = assemble_matrix(bmm, R, S)
  H1hchol  = lu(H1h)

  hm1    = FEFunction(Q, copy(get_free_dof_values(hn)))
  um1    = FEFunction(V, copy(get_free_dof_values(un)))
  hm2    = FEFunction(Q, copy(get_free_dof_values(hn)))
  um2    = FEFunction(V, copy(get_free_dof_values(un)))
  hp     = FEFunction(Q, copy(get_free_dof_values(hn)))
  up     = FEFunction(V, copy(get_free_dof_values(un)))
  ϕ      = FEFunction(Q, copy(get_free_dof_values(hn)))
  F      = FEFunction(V, copy(get_free_dof_values(un)))
  wn     = FEFunction(S, copy(get_free_dof_values(f)))
  q1     = FEFunction(S, copy(get_free_dof_values(f)))
  q2     = FEFunction(S, copy(get_free_dof_values(f)))
  # first step, no leap frog integration
  shallow_water_explicit_time_step!(hn, un, ϕ, F, q1, q2, model, dΩ, dω, f, g, hm1, um1, hm2, um2, hp, up, RTMMchol, L2MMchol, H1h, H1hchol, dt, false, τ, Q, V, R, S)
  initialize_csv(joinpath(out_dir,"swe_diagnostics.csv"), "time", "mass", "vorticity", "kinetic", "potential", "power")
  if mod(1, diag_freq) == 0
    compute_diagnostics_shallow_water!(wn, model, dΩ, dω, S, L2MM, H1MM, H1MMchol, h_tmp, w_tmp, g, hn, un, ϕ, F, 1, dt, true, out_dir)
  end
  
  # subsequent steps, do leap frog integration (now that we have the state at two previous time levels)
  for istep in 2:nstep
    get_free_dof_values(hm2) .= get_free_dof_values(hm1)
    get_free_dof_values(um2) .= get_free_dof_values(um1)
    get_free_dof_values(hm1) .= get_free_dof_values(hn)
    get_free_dof_values(um1) .= get_free_dof_values(un)

    shallow_water_explicit_time_step!(hn, un, ϕ, F, q1, q2, model, dΩ, dω, f, g, hm1, um1, hm2, um2, hp, up, RTMMchol, L2MMchol, H1h, H1hchol, dt, true, τ, Q, V, R, S)
    if mod(istep, diag_freq) == 0
      compute_diagnostics_shallow_water!(wn, model, dΩ, dω, S, L2MM, H1MM, H1MMchol, h_tmp, w_tmp, g, hn, un, ϕ, F, istep, dt, true, out_dir)
    end
    if mod(istep, dump_freq) == 0
      writevtk(Ω,"local/shallow_water_exp_n=$(istep)",cellfields=["hn"=>hn, "un"=>un, "wn"=>wn])
    end
  end

  hn, un
end
