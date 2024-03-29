function assemble_residuals!(duh, dΩ, dω, Y, qAPVM, ϕ, F, n)
  bᵤ(v) = ∫(-1.0*qAPVM*(v⋅⟂(F,n)))dΩ + ∫(DIV(v)*ϕ)dω
  bₕ(q) = ∫(-q*DIV(F))dω
  bₕᵤ((v,q)) = bᵤ(v) + bₕ(q)
  Gridap.FESpaces.assemble_vector!(bₕᵤ, duh, Y)
end

function shallow_water_rosenbrock_time_step!(
  y₂, ϕ, F, q₁, q₂, duh₁, duh₂, H1h, H1hchol, y_wrk,  # in/out args
  model, dΩ, dω, Y, V, Q, R, S, f, g, y₁, y₀,         # in args
  RTMMchol, L2MMchol, Amat, Bchol, Blfchol,           # more in args
  dt, τ, leap_frog,
  assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},R,S))                                   # ...yet more in args
  # energetically balanced second order rosenbrock shallow water solver
  # reference: eqns (24) and (39) of
  # https://github.com/BOM-Monash-Collaborations/articles/blob/main/energetically_balanced_time_integration/EnergeticallyBalancedTimeIntegration_SW.tex

  n = get_normal_vector(Triangulation(model))
  dt₁ = dt
  if leap_frog
    dt₁ = 2.0*dt
  end

  y₀v, y₁v, y₂v = get_free_dof_values(y₀,y₁,y₂)

  # multifield terms
  u₁, h₁ = y₁

  # 1.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*h₁)
  # 1.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,u₁⋅u₁,h₁,g)
  # 1.3: the potential vorticity
  compute_potential_vorticity!(q₁,H1h,H1hchol,dΩ,R,S,h₁,u₁,f,n,assem)
  # 1.4: assemble the momentum and continuity equation residuals
  assemble_residuals!(duh₁, dΩ, dω, Y, q₁ - τ*u₁⋅∇(q₁), ϕ, F, n)

  # Solve for du₁, dh₁ over a MultiFieldFESpace
  solve!(duh₁, Blfchol, duh₁)

  # update
  y₂v .=  y₀v .+ dt₁ .* duh₁

  u₂, h₂ = y₂
  # 2.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*(2.0*h₁ + h₂)/6.0+u₂*(h₁ + 2.0*h₂)/6.0)
  # 2.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,(u₁⋅u₁ + u₁⋅u₂ + u₂⋅u₂)/3.0,0.5*(h₁ + h₂),g)
  # 2.3: the potential vorticity
  compute_potential_vorticity!(q₂,H1h,H1hchol,dΩ,R,S,h₂,u₂,f,n,assem)
  # 2.4: assemble the momentum and continuity equation residuals
  assemble_residuals!(duh₂, dΩ, dω, Y, 0.5*(q₁ - τ*u₁⋅∇(q₁) + q₂ - τ*u₂⋅∇(q₂)), ϕ, F, n)

  # subtract A*[du₁,dh₁] from [du₂,dh₂] vector
  mul!(y_wrk, Amat, duh₁)
  duh₂ .= duh₂ .- y_wrk

  # solve for du₂, dh₂
  solve!(duh₂, Bchol, duh₂)

  # update yⁿ⁺¹
  y₂v .= y₁v .+ dt .* duh₂
end

function compute_mean_depth!(wrk, L2MM, h)
  # compute the mean depth over the sphere, for use in the approximate Jacobian
  mul!(wrk, L2MM, get_free_dof_values(h))
  h_int = sum(wrk)
  a_int = sum(L2MM,dims=[1,2])
  h_avg = h_int/a_int
  h_avg
end

function shallow_water_rosenbrock_time_stepper(
  model, order, degree,
  h₀, u₀, f₀, g, H₀,
  λ, dt, τ, N;
  mass_matrix_solver::Gridap.Algebra.LinearSolver=Gridap.Algebra.BackslashSolver(),
  jacobian_matrix_solver::Gridap.Algebra.LinearSolver=Gridap.Algebra.BackslashSolver(),
  leap_frog=false,
  write_diagnostics=true,
  write_diagnostics_freq=1,
  dump_diagnostics_on_screen=true,
  write_solution=false,
  write_solution_freq=N/10,
  output_dir="nswe_eq_ncells_$(num_cells(model))_order_$(order)_rosenbrock")

  # Forward integration of the shallow water equations
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())

  # Setup the trial and test spaces
  R, S, U, V, P, Q = setup_mixed_spaces(model, order)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  # assemble the mass matrices (RTMM mass matrix not needed)
  H1MM, _, L2MM, H1MMchol, RTMMchol, L2MMchol =
    setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q;
                                      mass_matrix_solver=mass_matrix_solver)

  # Project the initial conditions onto the trial spaces
  b₁(q)   = ∫(q*h₀)dΩ
  b₂(v)   = ∫(v⋅u₀)dΩ
  b₃(s)   = ∫(s*f₀)*dΩ
  rhs1    = assemble_vector(b₃, S)
  f       = FEFunction(S, copy(rhs1))
  solve!(get_free_dof_values(f),H1MMchol,get_free_dof_values(f))

  # assemble the approximate MultiFieldFESpace Jacobian
  n = get_normal_vector(Ω)

  Amat((u,p),(v,q)) = ∫(-dt*λ*f*(v⋅⟂(u,n)))dΩ + ∫(dt*λ*g*(DIV(v)*p))dω - ∫(dt*λ*H₀*(q*DIV(u)))dω
  Mmat((u,p),(v,q)) = ∫(u⋅v)dΩ + ∫(p*q)dΩ # block mass matrix
  A = assemble_matrix(Amat, X,Y)
  M = assemble_matrix(Mmat, X,Y)
  B = M - A

  Bchol = numerical_setup(symbolic_setup(jacobian_matrix_solver,B),B)
  Mchol = numerical_setup(symbolic_setup(mass_matrix_solver,M),M)
  # leap frog matrices, using 2×dt
  lf = 1.0
  if leap_frog
    lf = 2.0
  end
  Blf = M - lf*A
  Blfchol = numerical_setup(symbolic_setup(jacobian_matrix_solver,Blf),Blf)

  # multifield initial condtions
  b₄((v,q)) = b₁(q) + b₂(v)
  rhs2  = assemble_vector(b₄, Y)
  solve!(rhs2, Mchol, rhs2)
  yn  = FEFunction(Y, rhs2)

  un, hn = yn

  hnv, fv, ynv = get_free_dof_values(hn,f,yn)

  # work arrays
  h_tmp = copy(hnv)
  w_tmp = copy(fv)

  # build the potential vorticity lhs operator once just to initialise
  bmm(a,b) = ∫(a*hn*b)dΩ
  H1h      = assemble_matrix(bmm, R, S)
  H1hchol  = numerical_setup(symbolic_setup(mass_matrix_solver,H1h),H1h)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"nswe__rosenbrock_diagnostics.csv")

    ϕ      = clone_fe_function(Q,hn)
    F      = clone_fe_function(V,un)
    wn     = clone_fe_function(S,f)
    q1     = clone_fe_function(S,f)
    q2     = clone_fe_function(S,f)

    # mulifield fe functions
    ym1     = clone_fe_function(Y,yn)
    ym2     = clone_fe_function(Y,yn)
    duh1    = copy(ynv)
    duh2    = copy(ynv)
    y_wrk   = copy(ynv)

    if (write_diagnostics)
      initialize_csv(diagnostics_file,"time", "mass", "vorticity", "kinetic", "potential", "power")
    end

    # first step, no leap frog
    istep = 1
    shallow_water_rosenbrock_time_step!(yn, ϕ, F, q1, q2, duh1, duh2, H1h, H1hchol, y_wrk,
                                        model, dΩ, dω, Y, V, Q, R, S, f, g, ym1, ym2,
                                        RTMMchol, L2MMchol, A, Bchol, Blfchol, dt, τ, false)

    if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
      compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(Ω))
      dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                      model, dΩ, dω, S, L2MM, H1MM,
                                      hn, un, wn, ϕ, F, g, istep, dt,
                                      diagnostics_file,
                                      dump_diagnostics_on_screen)
    end
    # time step iteration loop
    for istep in 2:N
      aux=ym2
      ym2=ym1
      ym1=yn
      yn=aux
      shallow_water_rosenbrock_time_step!(yn, ϕ, F, q1, q2, duh1, duh2, H1h, H1hchol, y_wrk,
                                          model, dΩ, dω, Y, V, Q, R, S, f, g, ym1, ym2,
                                          RTMMchol, L2MMchol, A, Bchol, Blfchol, dt, τ, leap_frog)

      # IMPORTANT NOTE: We need to extract un, hn out of yn at each iteration because
      #                 the association of yn with its object instance changes at the beginning of
      #                 each iteration
      un, hn = yn
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
    pvdfile=joinpath(output_dir,"nswe_eq_ncells_$(num_cells(model))_order_$(order)_rosenbrock")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
