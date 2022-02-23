function assemble_residuals_implicit!(duh, dΩ, dω, Y, u₁, u₂, h₁, h₂, q, ϕ, F, n, dt)
  bᵤ(v) = ∫(v⋅(u₂ - u₁) + dt*q*(v⋅⟂(F,n)))dΩ - ∫(dt*DIV(v)*ϕ)dω
  bₕ(p) = ∫(p*(h₂ - h₁))dΩ + ∫(dt*p*DIV(F))dω
  bₕᵤ((v,p)) = bᵤ(v) + bₕ(p)
  Gridap.FESpaces.assemble_vector!(bₕᵤ, duh, Y)
end

function shallow_water_implicit_time_step!(
     y₂, ϕ, F, q₁, q₂, duh, 
     H1h, H1h_1, H1h_2, H1hchol, H1hchol_1, H1hchol_2,     # in/out args
     model, dΩ, dω, Y, V, Q, R, S, R1_up, R2_up, f, g, y₁, # in args
     RTMMchol, L2MMchol, Achol, dt, τ)                     # more in args

  n = get_normal_vector(Triangulation(model))

  y₁v, y₂v = get_free_dof_values(y₁,y₂)
  y₁v .= y₂v

  # multifield terms
  u₁, h₁ = y₁
  u₂, h₂ = y₂

  compute_potential_vorticity!(q₁,H1h,H1hchol,dΩ,R,S,h₁,u₁,f,n)
  # q₁_up = compute_potential_vorticity_downtrial!(q₁,H1h_1,H1hchol_1,dΩ,R,R1_up,S,h₁,u₁,f,n,τ,model)

  iter = 0
  while iter < 40
    compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*(2.0*h₁ + h₂)/6.0+u₂*(h₁ + 2.0*h₂)/6.0)
    compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,(u₁⋅u₁ + u₁⋅u₂ + u₂⋅u₂)/3.0,0.5*(h₁ + h₂),g)
    compute_potential_vorticity!(q₂,H1h,H1hchol,dΩ,R,S,h₂,u₂,f,n)
    # q₂_up = compute_potential_vorticity_downtrial!(q₂,H1h_2,H1hchol_2,dΩ,R,R2_up,S,h₂,u₂,f,n,τ,model)

    assemble_residuals_implicit!(duh, dΩ, dω, Y, u₁, u₂, h₁, h₂,
				 0.5*(q₁ - τ*u₁⋅∇(q₁) + q₂ - τ*u₂⋅∇(q₂)) - τ*(q₂ - q₁)/dt, ϕ, F, n, dt)
				 #0.5*(q₁ - τ*u₁⋅∇(q₁) + q₂ - τ*u₂⋅∇(q₂)), ϕ, F, n, dt)
    ldiv!(Achol, duh)

    y₂v .=  y₂v .- duh

    norm_x = norm(y₂v)
    norm_dx = norm(duh)
    @printf("%d:\t|x|:\t%14.9e\t|dx|:\t%14.9e\t|dx|/|x|:\t%14.9e\n", iter, norm_x, norm_dx, norm_dx/norm_x)

    if norm_dx/norm_x < 1.0e-13
      break
    end
 
    iter = iter + 1
  end

  y₁v .= y₂v
end

function shallow_water_implicit_time_stepper(model, order, degree,
                        h₀, u₀, f₀, g, H₀, dt, τ, N;
                        write_diagnostics=true,
                        write_diagnostics_freq=1,
                        dump_diagnostics_on_screen=true,
                        write_solution=false,
                        write_solution_freq=N/10,
                        output_dir="nswe_nc_$(num_cells(model))_ord_$(order)")

  # Forward integration of the shallow water equations
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())

  # Setup the trial and test spaces
  R, S, U, V, P, Q = setup_mixed_spaces(model, order)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  # assemble the mass matrices (RTMM mass matrix not needed)
  H1MM, _, L2MM, H1MMchol, RTMMchol, L2MMchol = setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)

  # Project the initial conditions onto the trial spaces
  b₁(q)   = ∫(q*h₀)dΩ
  b₂(v)   = ∫(v⋅u₀)dΩ
  b₃(s)   = ∫(s*f₀)*dΩ
  rhs1    = assemble_vector(b₃, S)
  f       = FEFunction(S, copy(rhs1))
  ldiv!(H1MMchol, get_free_dof_values(f))

  # assemble the approximate MultiFieldFESpace Jacobian
  n = get_normal_vector(Ω)

  Amat((u,p),(v,q)) = ∫(v⋅u + 0.5*dt*f*(v⋅⟂(u,n)))dΩ - ∫(0.5*dt*g*(DIV(v)*p))dω + ∫(0.5*dt*H₀*(q*DIV(u)))dω + ∫(p*q)dΩ
  Mmat((u,p),(v,q)) = ∫(u⋅v)dΩ + ∫(p*q)dΩ # block mass matrix
  A = assemble_matrix(Amat, X,Y)
  M = assemble_matrix(Mmat, X,Y)
  Achol = lu(A)
  Mchol = lu(M)

  # multifield initial condtions
  b₄((v,q)) = b₁(q) + b₂(v)
  rhs2 = assemble_vector(b₄, Y)
  ldiv!(Mchol, rhs2)
  yn   = FEFunction(Y, rhs2)

  un, hn = yn

  hnv, fv, ynv = get_free_dof_values(hn,f,yn)

  # work arrays
  h_tmp = copy(hnv)
  w_tmp = copy(fv)

  # upwinded trial function spaces
  R1_up = TrialFESpace(S)
  R2_up = TrialFESpace(S)

  # build the potential vorticity lhs operator once just to initialise
  bmm(a,b)  = ∫(a*hn*b)dΩ
  H1h       = assemble_matrix(bmm, R, S)
  H1hchol   = lu(H1h)
  H1h_1     = assemble_matrix(bmm, R1_up, S)
  H1hchol_1 = lu(H1h_1)
  H1h_2     = assemble_matrix(bmm, R2_up, S)
  H1hchol_2 = lu(H1h_2)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"nswe_diag.csv")

    ϕ      = clone_fe_function(Q,hn)
    F      = clone_fe_function(V,un)
    wn     = clone_fe_function(S,f)
    q1     = clone_fe_function(S,f)
    q2     = clone_fe_function(S,f)

    # mulifield fe functions
    ym1    = clone_fe_function(Y,yn)
    duh    = copy(ynv)

    if (write_diagnostics)
      initialize_csv(diagnostics_file,"time", "mass", "vorticity", "kinetic", "potential", "power", "enstrophy")
    end

    # time step iteration loop
    for istep in 1:N
      aux=ym1
      ym1=yn
      yn=aux
      shallow_water_implicit_time_step!(yn, ϕ, F, q1, q2, duh, 
					H1h, H1h_1, H1h_2, H1hchol, H1hchol_1, H1hchol_2,
                                        model, dΩ, dω, Y, V, Q, R, S, R1_up, R2_up, f, g, ym1,
                                        RTMMchol, L2MMchol, Achol, dt, τ)

      # IMPORTANT NOTE: We need to extract un, hn out of yn at each iteration because
      #                 the association of yn with its object instance changes at the beginning of
      #                 each iteration
      un, hn = yn
      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(Ω))
        dump_diagnostics_shallow_water!(h_tmp, w_tmp, model, dΩ, dω, S, L2MM, H1MM,
                                        hn, un, wn, ϕ, F, q2, g, istep, dt,
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
    pvdfile=joinpath(output_dir,"nswe_nc_$(num_cells(model))_ord_$(order)")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
