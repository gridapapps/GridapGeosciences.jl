function advection_hdg_time_step!(
     pn, pm, ph, un, un, model, dΩ, ∂K, d∂K, X, Y, dt, τ,
     assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X,Y))

  # Stiffly stable RK2 time integration

  n  = get_cell_normal_vector(∂K)
  nₒ = get_cell_owner_normal_vector(∂K)

  # First stage
  b(q,m) = ∫(q*pn + ∇q⋅un)dΩ - ∫(q*pn*(un⋅n))d∂K - ∫(τ*m*...)d∂K

  # Second stage

end

function project_initial_conditions(dΩ, P, Q, p₀, U, V, u₀, mass_matrix_solver)
  # the tracer
  b₁(q)    = ∫(q*p₀)dΩ
  a₁(p,q)  = ∫(q*p)dΩ
  rhs₁     = assemble_vector(b₁, Q)
  L2MM     = assemble_matrix(a₁, P, Q)
  L2MMchol = numerical_setup(symbolic_setup(mass_matrix_solver,L2MM),L2MM)
  pn       = FEFunction(Q, copy(rhs₁))
  pnv      = get_free_dof_values(pn)

  solve!(pnv, L2MMchol, pnv)

  # the velocity field
  b₂(v)    = ∫(v*u₀)dΩ
  a₂(u,v)  = ∫(v*u)dΩ
  rhs₂     = assemble_vector(b₂, V)
  MM       = assemble_matrix(a₂, U, V)
  MMchol   = numerical_setup(symbolic_setup(mass_matrix_solver,MM),MM)
  un       = FEFunction(V, copy(rhs₂))
  unv      = get_free_dof_values(un)

  solve!(unv, MMchol, unv)

  pn, pnv, L2MM, L2MMchol, un
end

function advection_hdg(
        model, order, degree,
        u₀, p₀, dt, N;
        mass_matrix_solver::Gridap.Algebra.LinearSolver=Gridap.Algebra.BackslashSolver(),
        write_diagnostics=true,
        write_diagnostics_freq=1,
        dump_diagnostics_on_screen=true,
        write_solution=false,
        write_solution_freq=N/10,
        output_dir="adv_eq_ncells_$(num_cells(model))_order_$(order)_explicit")

  # Forward integration of the advection equation
  D = num_cell_dims(model)
  Ω = Triangulation(ReferenceFE{D},model)
  Γ = Triangulation(ReferenceFE{D-1},model)
  ∂K = GridapHybrid.Skeleton(model)

  reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order;space=:P)
  reffeₚ = ReferenceFE(lagrangian,Float64,order;space=:P)
  reffeₗ = ReferenceFE(lagrangian,Float64,order;space=:P)

  # Define test FESpaces
  V = TestFESpace(Ω, reffeᵤ; conformity=:L2)
  Q = TestFESpace(Ω, reffeₚ; conformity=:L2)
  M = TestFESpace(Γ, reffeₗ; conformity=:L2)
  Y = MultiFieldFESpace([Q,M])

  U = TrialFESpace(V)
  P = TrialFESpace(Q)
  L = TrialFESpace(M)
  X = MultiFieldFESpace([P,L])

  dΩ  = Measure(Ω,degree)
  d∂K = Measure(∂K,degree)

  # HDG stabilisation parameter
  τ = 1.0

  # Project the initial conditions onto the trial spaces
  pn, pnv, L2MM, L2MMchol, un = project_initial_conditions(dΩ, P, Q, p₀, U, V, u₀, mass_matrix_solver)

  # Work array
  p_tmp = copy(pnv)

  # Initial states
  mul!(p_tmp, L2MM, pnv)
  p01 = sum(p_tmp)  # total mass
  p02 = p_tmp⋅pnv   # total entropy

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"adv_diagnostics.csv")

    ph  = clone_fe_function(Q,pn)
    pm  = clone_fe_function(Q,pn)
    pmv = get_free_dof_values(pm)

    for istep in 1:N
      pmv .= pnv

      advection_hdg_time_step!(pn, pm, ph, un, un, model, dΩ, ∂K, d∂K, X, Y, dt, τ)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        # compute mass and entropy conservation
        mul!(p_tmp, L2MM, pnv)
        pn1 = sum(p_tmp)
        pn2 = p_tmp⋅pnv
	pn1 = (pn1 - p01)/p01
	pn2 = (pn2 - p02)/p02
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        pvd[dt*Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),["hn"=>hn,"un"=>un])
      end
    end
    hn, un
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,"adv_eq_ncells_$(num_cells(model))_order_$(order)_explicit")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
