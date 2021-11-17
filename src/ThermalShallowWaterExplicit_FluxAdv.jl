function update_velocity_tswe_ff!(u1,dΩ,dω,dΓ,V,RTMMchol,u2,q,e,F,ϕ,T,n,dt)
  b(v) = ∫(v⋅u2 - dt*q*(v⋅⟂(F,n)) + (∇⋅(e*v))*dt*T)dΩ - ∫(dt*mean(e*T)*jump(v⋅n))dΓ + ∫(dt*DIV(v)*ϕ)dω
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(u1), V)
  ldiv!(RTMMchol, get_free_dof_values(u1))
end

function update_buoyancy_ff!(E1,dΩ,dω,dΓ,S,H1MMchol,E2,e,F,n,dt)
  b(s) = ∫(s*E2 - dt*s*(∇⋅(e*F)))dΩ + ∫(dt*mean(e*s)*jump(F⋅n))dΓ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(E1), S)
  ldiv!(H1MMchol, get_free_dof_values(E1))
end

function compute_buoyancy_ff!(e,dΩ,R,S,H1MM,H1chol,E,h,u,τ)
  b(s) = ∫(s*E - τ*s*u⋅∇(E))dΩ
  Gridap.FESpaces.assemble_vector!(b, get_free_dof_values(e), S)
  ldiv!(H1chol, get_free_dof_values(e))
end

function thermal_shallow_water_flux_adv_explicit_time_step!(
     h₂, u₂, E₂, hₚ, uₚ, Eₚ, ϕ, F, q₁, q₂, e₁, e₂, H1h, H1hchol, T,  # in/out args
     model, dΩ, dω, dΓ, V, Q, R, S, f, h₁, u₁, E₁, hₘ, uₘ, Eₘ,       # in args
     H1MMchol, RTMMchol, L2MMchol, dt, τ, τₑ, leap_frog)             # more in args

  # energetically balanced explicit second order thermal shallow water solver.
  # extends the explicit shallow water solver with the an additional buoyancy 
  # modulated forcing term, and flux form equation for the density weighted 
  # buoyancy

  n = get_normal_vector(model)
  # explicit step for provisional velocity, uₚ
  dt1 = dt
  if leap_frog
    dt1 = 2.0*dt
  end

  # 1.1: the mass flux ∈ H(div,Ω)
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*h₁)
  # 1.2: the bernoulli function ∈ L²(Ω)
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,u₁⋅u₁,E₁,0.5)
  # 1.3: temperature ∈ H¹(Ω)
  compute_temperature!(T,dΩ,S,H1MMchol,0.5*h₁)
  # 1.4: materially advected quantities (potential vorticity and buoyancy)
  compute_potential_vorticity!(q₁,H1h,H1hchol,dΩ,R,S,h₁,u₁,f,n)
  compute_buoyancy_ff!(e₁,dΩ,R,S,H1h,H1hchol,E₁,h₁,u₁,τₑ)
  # 1.5: solve for the provisional velocity
  update_velocity_tswe_ff!(uₚ,dΩ,dω,dΓ,V,RTMMchol,uₘ,q₁-τ*u₁⋅∇(q₁),e₁,F,ϕ,T,n,dt1)
  # 1.6: solve for the provisional depth
  compute_depth!(hₚ,dΩ,dω,Q,L2MMchol,hₘ,F,dt1)
  # 1.7: solve for the buoyancy weighted mass flux
  update_buoyancy_ff!(Eₚ,dΩ,dω,dΓ,S,H1MMchol,Eₘ,e₁,F,n,dt1)

  # 2.1: the mass flux ∈ H(div,Ω)
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*(2.0*h₁ + hₚ)/6.0+uₚ*(h₁ + 2.0*hₚ)/6.0)
  # 2.2: the bernoulli function ∈ L²(Ω)
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,(u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/3.0,0.5*(E₁+Eₚ),0.5)
  # 2.3: temperature ∈ H¹(Ω)
  compute_temperature!(T,dΩ,S,H1MMchol,0.25*(h₁+hₚ))
  # 2.4: materially advected quantities (potential vorticity and buoyancy)
  compute_potential_vorticity!(q₂,H1h,H1hchol,dΩ,R,S,hₚ,uₚ,f,n)
  compute_buoyancy_ff!(e₂,dΩ,R,S,H1h,H1hchol,Eₚ,hₚ,uₚ,τₑ)
  # 2.5: solve for the final velocity
  update_velocity_tswe_ff!(u₂,dΩ,dω,dΓ,V,RTMMchol,u₁,0.5*(q₁-τ*u₁⋅∇(q₁)+q₂-τ*uₚ⋅∇(q₂)),0.5*(e₁+e₂),F,ϕ,T,n,dt)
  # 2.6: solve for the final depth
  compute_depth!(h₂,dΩ,dω,Q,L2MMchol,h₁,F,dt)
  # 2.7: solve for the buoyancy weighted mass flux
  update_buoyancy_ff!(E₂,dΩ,dω,dΓ,S,H1MMchol,Eₘ,0.5*(e₁+eₑ),F,n,dt)
end

function thermal_shallow_water_explicit_time_stepper(model, order, degree,
                        h₀, u₀, E₀, f₀,
                        dt, τ, τₑ, N;
                        write_diagnostics=true,
                        write_diagnostics_freq=1,
                        dump_diagnostics_on_screen=true,
                        write_solution=false,
                        write_solution_freq=N/10,
                        output_dir="tswe_ncells_$(num_cells(model))_order_$(order)_explicit")

  # Forward integration of the shallow water equations
  Ω  = Triangulation(model)
  Γ  = SkeletonTriangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())
  dΓ = Measure(Γ, degree)

  # Setup the trial and test spaces
  R, S, U, V, P, Q = setup_mixed_spaces(model, order)

  # assemble the mass matrices
  H1MM, _, L2MM, H1MMchol, RTMMchol, L2MMchol = setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)

  # Project the initial conditions onto the trial spaces
  hn, un, f, hnv, unv, fv =  project_shallow_water_initial_conditions(dΩ, Q, V, S, 
                               L2MMchol, RTMMchol, H1MMchol, h₀, u₀, f₀)

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
    Em1    = clone_fe_function(S,En)
    Em2    = clone_fe_function(S,En)
    Ep     = clone_fe_function(S,En)
    T      = clone_fe_function(S,En)

    um1    = clone_fe_function(V,un)
    um2    = clone_fe_function(V,un)
    up     = clone_fe_function(V,un)
    F      = clone_fe_function(V,un)

    wn     = clone_fe_function(S,f)
    q1     = clone_fe_function(S,f)
    q2     = clone_fe_function(S,f)
    e1     = clone_fe_function(S,f)
    e2     = clone_fe_function(S,f)

    # first step, no leap frog integration
    thermal_shallow_water_flux_adv_explicit_time_step!(hn, un, En, hp, up, Ep, ϕ, F, q1, q2, e1, e2, H1h, H1hchol, T,
                                              model, dΩ, dω, dΓ, V, Q, R, S, f, hm1, um1, Em1, hm2, um2, Em2,
                                              H1MMchol, RTMMchol, L2MMchol, dt, τ, τₑ, false)

    if (write_diagnostics)
      initialize_csv(diagnostics_file,"time", "mass", "vorticity", "buoyancy", "kinetic", "internal", "power_k2p", "power_k2i")
    end

    if (write_diagnostics && write_diagnostics_freq==1)
      compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
      dump_diagnostics_thermal_shallow_water!(h_tmp, w_tmp,
                                              model, dΩ, dω, S, L2MM, H1MM,
					      hn, un, En, wn, ϕ, F, 0.5*(e1+e2)*F, 1, dt,
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

      thermal_shallow_water_flux_adv_explicit_time_step!(hn, un, En, hp, up, Ep, ϕ, F, q1, q2, e1, e2, H1h, H1hchol, T,
                                                model, dΩ, dω, dΓ, V, Q, R, S, f, hm1, um1, Em1, hm2, um2, Em2,
                                                H1MMchol, RTMMchol, L2MMchol, dt, τ, τₑ, true)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
        dump_diagnostics_thermal_shallow_water!(h_tmp, w_tmp,
                                                model, dΩ, dω, S, L2MM, H1MM,
						hn, un, En, wn, ϕ, F, 0.5*(e1+e2)*F, istep, dt,
                                                diagnostics_file,
                                                dump_diagnostics_on_screen)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
	pvd[dt*Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),["hn"=>hn,"un"=>un,"wn"=>wn,"en"=>e2])
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
