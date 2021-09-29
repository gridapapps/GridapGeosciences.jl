function compute_temperature_gradient!(dT,dω,V,RTMMchol,T)
  b(v) = ∫(-1.0*DIV(v)*T)dω
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(dT), V)
  ldiv!(RTMMchol, get_free_dof_values(dT))
end

function compute_buoyancy_flux!(eF,dΩ,V,RTMMchol,e,F)
  b(v) = ∫((v⋅F)*e)dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(eF), V)
  ldiv!(RTMMchol, get_free_dof_values(eF))
end

# assume that H1chol factorization has already been performed
function compute_buoyancy!(e,dΩ,S,H1chol,E)
  b(s) = ∫(s*E)dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(e), S)
  ldiv!(H1chol, get_free_dof_values(e))
end

function compute_velocity_tswe!(u1,dΩ,dω,V,RTMMchol,u2,qAPVM,eAPVM,F,ϕ,dT,n,dt1,dt2)
  b(v) = ∫(v⋅u2 - dt1*(qAPVM)*(v⋅⟂(F,n)) - dt1*(eAPVM)*v⋅dT)dΩ + ∫(dt2*DIV(v)*ϕ)dω
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(u1), V)
  ldiv!(RTMMchol, get_free_dof_values(u1))
end

function thermal_shallow_water_explicit_time_step!(
     h₂, u₂, E₂, hₚ, uₚ, Eₚ, ϕ, F, q₁, q₂, e₁, e₂, H1h, H1hchol, dT,  # in/out args
     model, dΩ, dω, V, Q, R, S, f, h₁, u₁, E₁, hₘ, uₘ, Eₘ,            # in args
     RTMMchol, L2MMchol, dt, τ, leap_frog)                            # more in args

  # energetically balanced explicit second order thermal shallow water solver
  #
  # f          : coriolis force (field)
  # h₁         : fluid depth at current time level
  # u₁         : fluid velocity at current time level
  # E₁         : fluid buoyancy at current time level
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
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*h₁)
  # 1.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,u₁⋅u₁,S₁,0.5)
  # 1.3: materially advected quantities (potential vorticity and buoyancy)
  compute_potential_vorticity!(q₁,H1h,H1hchol,dΩ,R,S,h₁,u₁,f,n)
  compute_buoyancy!(e₁,dΩ,S,H1chol,E₁)
  # 1.4: solve for the provisional velocity
  compute_temperature_gradient!(dT,dω,V,RTMMchol,0.5*h₁)
  compute_velocity_tswe!(uₚ,dΩ,dω,V,RTMMchol,uₘ,q₁-τ*u₁⋅∇(q₁),e₁-τ*u₁⋅∇(e₁),F,ϕ,dT,n,dt1,dt1)
  # 1.5: solve for the provisional depth
  compute_depth!(hₚ,dΩ,dω,Q,L2MMchol,hₘ,F,dt1)
  # 1.6: solve for the buoyancy weighted mass flux
  compute_buoyancy_flux!(dT,dΩ,V,RTMMchol,e₁-τ*u₁⋅∇(e₁),F)
  compute_depth!(Eₚ,dΩ,dω,Q,L2MMchol,Eₘ,dT,dt1)

  # 2.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*(2.0*h₁ + hₚ)/6.0+uₚ*(h₁ + 2.0*hₚ)/6.0)
  # 2.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,(u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/3.0,0.5*(S₁ + Sₚ),0.5)
  # 2.3: materially advected quantities (potential vorticity and buoyancy)
  compute_potential_vorticity!(q₂,H1h,H1hchol,dΩ,R,S,hₚ,uₚ,f,n)
  compute_buoyancy!(e₂,dΩ,S,H1chol,Eₚ)
  # 2.4: solve for the final velocity
  compute_temperature_gradient!(dT,dω,V,RTMMchol,0.25*(h₁+hₚ))
  compute_velocity_tswe!(u₂,dΩ,dω,V,RTMMchol,u₁,q₁-τ*u₁⋅∇(q₁)+q₂-τ*uₚ⋅∇(q₂),e₁-τ*u₁⋅∇(e₁)+e₂-τ*uₚ⋅∇(e₂),F,ϕ,dT,n,0.5*dt,dt)
  # 2.5: solve for the final depth
  compute_depth!(h₂,dΩ,dω,Q,L2MMchol,h₁,F,dt)
  # 2.6: solve for the buoyancy weighted mass flux
  compute_buoyancy_flux!(dT,dΩ,V,RTMMchol,e₁-τ*u₁⋅∇(e₁)+e₂-τ*uₚ⋅∇(e₂),F)
  compute_depth!(E₂,dΩ,dω,Q,L2MMchol,E₁,dT,dt)
end

function thermal_shallow_water_explicit_time_stepper(model, order, degree,
                        h₀, u₀, E₀, f₀,
                        dt, τ, N;
                        write_diagnostics=true,
                        write_diagnostics_freq=1,
                        dump_diagnostics_on_screen=true,
                        write_solution=false,
                        write_solution_freq=N/10,
                        output_dir="tswe_ncells_$(num_cells(model))_order_$(order)_explicit")

  # Forward integration of the shallow water equations
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())

  # Setup the trial and test spaces
  R, S, U, V, P, Q = setup_mixed_spaces(model, order)

  # assemble the mass matrices
  H1MM, RTMM, L2MM, H1MMchol, RTMMchol, L2MMchol = setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)

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

  b₄(q)   = ∫(q*E₀)dΩ
  rhs4    = assemble_vector(b₄, Q)
  En      = FEFunction(Q, copy(rhs4))
  ldiv!(L2MMchol, get_free_dof_values(En))

  # work arrays
  h_tmp = copy(get_free_dof_values(hn))
  w_tmp = copy(get_free_dof_values(f))
  # build the potential vorticity lhs operator once just to initialise
  bmm(a,b) = ∫(a*hn*b)dΩ
  H1h      = assemble_matrix(bmm, R, S)
  H1hchol  = lu(H1h)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"tswe_diagnostics.csv")

    hm1    = clone_fe_function(Q,hn)
    hm2    = clone_fe_function(Q,hn)
    hp     = clone_fe_function(Q,hn)
    ϕ      = clone_fe_function(Q,hn)
    Em1    = clone_fe_function(Q,hn)
    Em2    = clone_fe_function(Q,hn)

    um1    = clone_fe_function(V,un)
    um2    = clone_fe_function(V,un)
    up     = clone_fe_function(V,un)
    F      = clone_fe_function(V,un)
    eF     = clone_fe_function(V,un)

    wn     = clone_fe_function(S,f)
    q1     = clone_fe_function(S,f)
    q2     = clone_fe_function(S,f)
    e1     = clone_fe_function(S,f)
    e2     = clone_fe_function(S,f)

    # first step, no leap frog integration
    thermal_shallow_water_explicit_time_step!(hn, un, En, hp, up, Ep, ϕ, F, q1, q2, e1, e2, H1h, H1hchol, eF,
                                              model, dΩ, dω, V, Q, R, S, f, hm1, um1, Em1, hm2, um2, Em2,
                                              RTMMchol, L2MMchol, dt, τ, false)

    if (write_diagnostics)
      initialize_csv(diagnostics_file,"time", "mass", "vorticity", "kinetic", "internal", "power-k2p","power-k2i")
    end

    if (write_diagnostics && write_diagnostics_freq==1)
      compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
      dump_diagnostics_thermal_shallow_water!(h_tmp, w_tmp,
                                              model, dΩ, dω, S, L2MM, H1MM,
                                              hn, un, En, wn, ϕ, F, eF, 1, dt,
                                              diagnostics_file,
                                              dump_diagnostics_on_screen)
    end

    # subsequent steps, do leap frog integration
    # (now that we have the state at two previous time levels)
    for istep in 2:N
      h_aux = hm2
      hm2   = hm1
      hm1   = hn
      hn    = h_aux
      u_aux = um2
      um2   = um1
      um1   = un
      un    = u_aux
      E_aux = Em2
      Em2   = Em1
      Em1   = En
      En    = E_aux

      thermal_shallow_water_explicit_time_step!(hn, un, En, hp, up, Ep, ϕ, F, q1, q2, e1, e2, H1h, H1hchol, eF,
                                                model, dΩ, dω, V, Q, R, S, f, hm1, um1, Em1, hm2, um2, Em2,
                                                RTMMchol, L2MMchol, dt, τ, true)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
        dump_diagnostics_thermal_shallow_water!(h_tmp, w_tmp,
                                                model, dΩ, dω, S, L2MM, H1MM,
                                                hn, un, En, wn, ϕ, F, eF, istep, dt,
                                                diagnostics_file,
                                                dump_diagnostics_on_screen)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
        pvd[Float64(istep)] = new_vtk_step_tswe(Ω,joinpath(output_dir,"n=$(istep)"),hn,un,wn,e2)
      end
    end
    hn, un, En
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,"tswe_ncells_$(num_cells(model))_order_$(order)_explicit")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
