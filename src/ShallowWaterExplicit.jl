function compute_potential_vorticity!(q,H1h,H1hchol,dΩ,R,S,h,u,f,n)
  a(r,s) = ∫(s*h*r)dΩ
  c(s)   = ∫(perp(n,∇(s))⋅(u) + s*f)dΩ
  Gridap.FESpaces.assemble_matrix_and_vector!(a, c, H1h, get_free_dof_values(q), R, S)
  lu!(H1hchol, H1h)
  ldiv!(H1hchol, get_free_dof_values(q))
end

function compute_potential_vorticity_bis!(q,H1h,H1hchol,dΩ,R,S,h,u,f,n)
  a(r,s) = ∫(s*h*r)dΩ
  c(s)   = ∫(perp(n,∇(s))⋅(u) + s*f)dΩ
  Gridap.FESpaces.assemble_matrix_and_vector!(a, c, H1h, get_free_dof_values(q), R, S)
  numerical_setup!(H1hchol,H1h)
  solve!(get_free_dof_values(q),H1hchol,get_free_dof_values(q))
end

function compute_velocity!(u1,dΩ,dω,V,RTMMchol,u2,qAPVM,F,ϕ,n,dt1,dt2)
  b(v) = ∫(v⋅u2 - dt1*(qAPVM)*(v⋅⟂(F,n)))dΩ + ∫(dt2*DIV(v)*ϕ)dω
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(u1), V)
  ldiv!(RTMMchol, get_free_dof_values(u1))
end

function compute_mass_flux!(F,dΩ,V,RTMMchol,u)
  b(v) = ∫(v⋅u)dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(F), V)
  ldiv!(RTMMchol, get_free_dof_values(F))
end

function compute_mass_flux_bis!(F,dΩ,V,RTMMchol,u)
  b(v) = ∫(v⋅u)dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(F), V)
  solve!(get_free_dof_values(F),RTMMchol,get_free_dof_values(F))
end

function compute_depth!(h1,dΩ,dω,Q,L2MMchol,h2,F,dt)
  b(q)  = ∫(q*h2)dΩ - ∫(dt*q*DIV(F))dω
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(h1), Q)
  ldiv!(L2MMchol, get_free_dof_values(h1))
end

function compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,uu,h,g)
  b(q)  = ∫(q*(0.5*uu + g*h))*dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(ϕ), Q)
  ldiv!(L2MMchol, get_free_dof_values(ϕ))
end

function compute_bernoulli_potential_bis!(ϕ,dΩ,Q,L2MMchol,uu,h,g)
  b(q)  = ∫(q*(0.5*uu + g*h))*dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(ϕ), Q)
  solve!(get_free_dof_values(ϕ),L2MMchol,get_free_dof_values(ϕ))
end

function compute_diagnostic_vorticity!(w,dΩ,S,H1MMchol,u,n)
  b(s) = ∫(⟂(n,∇(s))⋅(u))dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(w), S)
  ldiv!(H1MMchol, get_free_dof_values(w))
end

function compute_diagnostic_vorticity_bis!(w,dΩ,S,H1MMchol,u,n)
  b(s) = ∫(⟂(n,∇(s))⋅(u))dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(w), S)
  solve!(get_free_dof_values(w),H1MMchol,get_free_dof_values(w))
end

function shallow_water_explicit_time_step!(
     h₂, u₂, hₚ, uₚ, ϕ, F, q₁, q₂, H1h, H1hchol,      # in/out args
     model, dΩ, dω, V, Q, R, S, f, g, h₁, u₁, hₘ, uₘ, # in args
     RTMMchol, L2MMchol, dt, τ, leap_frog)            # more in args

  # energetically balanced explicit second order shallow water solver
  # reference: eqns (21-24) of
  # https://github.com/BOM-Monash-Collaborations/articles/blob/main/energetically_balanced_time_integration/EnergeticallyBalancedTimeIntegration_SW.tex

  n = get_normal_vector(Triangulation(model))
  # explicit step for provisional velocity, uₚ
  dt1 = dt
  if leap_frog
    dt1 = 2.0*dt
  end

  # 1.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*h₁)
  # 1.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,u₁⋅u₁,h₁,g)
  # 1.3: the potential vorticity
  compute_potential_vorticity!(q₁,H1h,H1hchol,dΩ,R,S,h₁,u₁,f,n)
  # 1.4: solve for the provisional velocity
  compute_velocity!(uₚ,dΩ,dω,V,RTMMchol,uₘ,q₁-τ*u₁⋅∇(q₁),F,ϕ,n,dt1,dt1)
  # 1.5: solve for the provisional depth
  compute_depth!(hₚ,dΩ,dω,Q,L2MMchol,hₘ,F,dt1)

  # 2.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*(2.0*h₁ + hₚ)/6.0+uₚ*(h₁ + 2.0*hₚ)/6.0)
  # 2.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,(u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/3.0,0.5*(h₁ + hₚ),g)
  # 2.3: the potential vorticity
  compute_potential_vorticity!(q₂,H1h,H1hchol,dΩ,R,S,hₚ,uₚ,f,n)
  # 2.4: solve for the final velocity
  compute_velocity!(u₂,dΩ,dω,V,RTMMchol,u₁,q₁-τ*u₁⋅∇(q₁)+q₂-τ*uₚ⋅∇(q₂),F,ϕ,n,0.5*dt,dt)
  # 2.5: solve for the final depth
  compute_depth!(h₂,dΩ,dω,Q,L2MMchol,h₁,F,dt)
end

function project_shallow_water_initial_conditions(dΩ, Q, V, S, L2MMchol, RTMMchol, H1MMchol, h₀, u₀, f₀)
  b₁(q)   = ∫(q*h₀)dΩ
  rhs1    = assemble_vector(b₁, Q)
  hn      = FEFunction(Q, copy(rhs1))

  b₂(v)   = ∫(v⋅u₀)dΩ
  rhs2    = assemble_vector(b₂, V)
  un      = FEFunction(V, copy(rhs2))

  b₃(s)   = ∫(s*f₀)*dΩ
  rhs3    = assemble_vector(b₃, S)
  f       = FEFunction(S, copy(rhs3))

  hnv,unv,fv=get_free_dof_values(hn,un,f)
  ldiv!(L2MMchol, hnv)
  ldiv!(RTMMchol, unv)
  ldiv!(H1MMchol, fv)

  hn, un, f, hnv, unv, fv
end

function shallow_water_explicit_time_stepper(model, order, degree,
                        h₀, u₀, f₀, g,
                        dt, τ, N;
                        write_diagnostics=true,
                        write_diagnostics_freq=1,
                        dump_diagnostics_on_screen=true,
                        write_solution=false,
                        write_solution_freq=N/10,
                        output_dir="nswe_eq_ncells_$(num_cells(model))_order_$(order)_explicit")

  # Forward integration of the shallow water equations
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())

  # Setup the trial and test spaces
  R, S, U, V, P, Q = setup_mixed_spaces(model, order)

  # assemble the mass matrices
  H1MM, _, L2MM, H1MMchol, RTMMchol, L2MMchol = setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)

  # Project the initial conditions onto the trial spaces
  hn, un, f, hnv, unv, fv =  project_shallow_water_initial_conditions(dΩ, Q, V, S,
                               L2MMchol, RTMMchol, H1MMchol, h₀, u₀, f₀)

  # work arrays
  h_tmp = copy(hnv)
  w_tmp = copy(fv)
  # build the potential vorticity lhs operator once just to initialise
  bmm(a,b) = ∫(a*hn*b)dΩ
  H1h      = assemble_matrix(bmm, R, S)
  H1hchol  = lu(H1h)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"nswe_diagnostics.csv")

    hm1    = clone_fe_function(Q,hn)
    hm2    = clone_fe_function(Q,hn)
    hp     = clone_fe_function(Q,hn)
    ϕ      = clone_fe_function(Q,hn)

    um1    = clone_fe_function(V,un)
    um2    = clone_fe_function(V,un)
    up     = clone_fe_function(V,un)
    F      = clone_fe_function(V,un)

    wn     = clone_fe_function(S,f)
    q1     = clone_fe_function(S,f)
    q2     = clone_fe_function(S,f)

    # first step, no leap frog integration
    shallow_water_explicit_time_step!(hn, un, hp, up, ϕ, F, q1, q2, H1h, H1hchol,
                                      model, dΩ, dω, V, Q, R, S, f, g, hm1, um1, hm2, um2,
                                      RTMMchol, L2MMchol, dt, τ, false)

    if (write_diagnostics)
      initialize_csv(diagnostics_file,"time", "mass", "vorticity", "kinetic", "potential", "power")
    end

    if (write_diagnostics && write_diagnostics_freq==1)
      compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(Ω))
      dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                      model, dΩ, dω, S, L2MM, H1MM,
                                      hn, un, wn, ϕ, F, g, 1, dt,
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

      shallow_water_explicit_time_step!(hn, un, hp, up, ϕ, F, q1, q2, H1h, H1hchol,
                                        model, dΩ, dω, V, Q, R, S, f, g, hm1, um1, hm2, um2,
                                        RTMMchol, L2MMchol, dt, τ, true)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(Ω))
        dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                        model, dΩ, dω, S, L2MM, H1MM,
                                        hn, un, wn, ϕ, F, g, istep, dt,
                                        diagnostics_file,
                                        dump_diagnostics_on_screen)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(Ω))
        pvd[dt*Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),["hn"=>hn,"un"=>un,"wn"=>wn])
      end
    end
    hn, un
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,"nswe_eq_ncells_$(num_cells(model))_order_$(order)_explicit")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
