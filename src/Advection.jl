function advect!(q,H1h,H1hchol,dΩ,R,S,S_up,q0,qh,u,dt,τ,model)
  rh_trial    = get_trial_fe_basis(R)
  sh_test     = get_fe_basis(S)
  sh_test_up  = upwind_test_functions(sh_test,u,τ,model)

  # step 1
  a1(r,s) = ∫(s*r)dΩ
  c1(s)   = ∫(s*q0 - dt*s*(u⋅∇(q0)))dΩ

  mat_contrib = a1(rh_trial,sh_test_up)
  vec_contrib = c1(sh_test_up)

  assem = SparseMatrixAssembler(R,S_up)
  data  = Gridap.FESpaces.collect_cell_matrix_and_vector(R,S_up,mat_contrib,vec_contrib)

  Gridap.FESpaces.assemble_matrix_and_vector!(H1h,get_free_dof_values(q),assem,data)

  lu!(H1hchol, H1h)
  ldiv!(H1hchol, get_free_dof_values(qh))

  # step 2
  a2(r,s) = ∫(s*r)dΩ
  c2(s)   = ∫(s*q0 - 0.5*dt*s*(u⋅∇(q0)) - 0.5*dt*s*(u⋅∇(qh)))dΩ

  mat_contrib = a2(rh_trial,sh_test_up)
  vec_contrib = c2(sh_test_up)

  assem = SparseMatrixAssembler(R,S_up)
  data  = Gridap.FESpaces.collect_cell_matrix_and_vector(R,S_up,mat_contrib,vec_contrib)

  Gridap.FESpaces.assemble_matrix_and_vector!(H1h,get_free_dof_values(q),assem,data)

  lu!(H1hchol, H1h)
  ldiv!(H1hchol, get_free_dof_values(q))
end

function advect_solid_body(model, order, degree,
                        q₀, u₀, dt, τ, N;
                        write_solution=false,
                        write_solution_freq=N/10,
                        output_dir="adv_ncells_$(num_cells(model))_order_$(order)")

  # Forward integration of the shallow water equations
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)

  # Setup the trial and test spaces
  reffe_rt  = ReferenceFE(raviart_thomas, Float64, order)
  V = FESpace(model, reffe_rt ; conformity=:HDiv)
  U = TrialFESpace(V)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order+1)
  S = FESpace(model, reffe_lgn; conformity=:H1)
  R = TrialFESpace(S)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order+1)
  S_up = FESpace(model, reffe_lgn; conformity=:H1)

  # assemble the mass matrices
  amm(a,b) = ∫(a⋅b)dΩ
  H1MM = assemble_matrix(amm, R, S)
  RTMM = assemble_matrix(amm, U, V)
  H1MMchol = lu(H1MM)
  RTMMchol = lu(RTMM)

  # Project the initial conditions onto the trial spaces
  b₂(v)   = ∫(v⋅u₀)dΩ
  rhsu    = assemble_vector(b₂, V)
  un      = FEFunction(V, copy(rhsu))
  ldiv!(RTMMchol, get_free_dof_values(un))
  b₃(s)   = ∫(s*q₀)*dΩ
  rhsq    = assemble_vector(b₃, S)
  qn      = FEFunction(S, copy(rhsq))
  ldiv!(H1MMchol, get_free_dof_values(qn))

  q_tmp = copy(get_free_dof_values(qn))

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"advection_diagnostics.csv")

    qp     = clone_fe_function(S,f)
    qm     = clone_fe_function(S,f)
    get_free_dof_values(qm) .= get_free_dof_values(qn)

    initialize_csv(diagnostics_file,"time", "mass", "mass_sq")

    for istep in 1:N
      q_aux = qm
      qm    = qn
      qn    = q_aux

      advect!(qn,H1h,H1hchol,dΩ,R,S,S_up,qm,qp,un,dt,τ,model)

      mul!(q_tmp,L2MM,get_free_dof_values(qn))
      mass_i    = sum(q_tmp)
      mass_sq_i = q_tmp⋅get_free_dof_values(qn)
      append_to_csv(diagnostics_file;
                time     = step*dt/24/60/60,
                mass     = mass_i,
		mass_sq  = mass_sq_i)
      @printf("%5d %14.9e %14.9e\n", step, mass_i, mass_sq_i)

      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        pvd[dt*Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),["qn"=>qn])
      end
    end
    qn
  end
  if (write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
    pvdfile=joinpath(output_dir,"adv_ncells_$(num_cells(model))_order_$(order)")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end

