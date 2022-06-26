function advection_hdg_time_step!(
     q₂, u₂, qₚ, uₚ, H1h, H1hchol,      # in/out args
     model, dΩ, V, Q, R, S, # in args
     RTMMchol, L2MMchol, dt,
     assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},R,S))            # more in args

  n = get_normal_vector(Triangulation(model))

  # 1.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*h₁)
  # 1.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,u₁⋅u₁,h₁,g)
  # 1.3: the potential vorticity
  compute_potential_vorticity!(q₁,H1h,H1hchol,dΩ,R,S,h₁,u₁,f,n,assem)
  # 1.4: solve for the provisional velocity
  compute_velocity!(uₚ,dΩ,dω,V,RTMMchol,uₘ,q₁-τ*u₁⋅∇(q₁),F,ϕ,n,dt1,dt1)
  # 1.5: solve for the provisional depth
  compute_depth!(hₚ,dΩ,dω,Q,L2MMchol,hₘ,F,dt1)

  # 2.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*(2.0*h₁ + hₚ)/6.0+uₚ*(h₁ + 2.0*hₚ)/6.0)
  # 2.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,(u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/3.0,0.5*(h₁ + hₚ),g)
  # 2.3: the potential vorticity
  compute_potential_vorticity!(q₂,H1h,H1hchol,dΩ,R,S,hₚ,uₚ,f,n,assem)
  # 2.4: solve for the final velocity
  compute_velocity!(u₂,dΩ,dω,V,RTMMchol,u₁,q₁-τ*u₁⋅∇(q₁)+q₂-τ*uₚ⋅∇(q₂),F,ϕ,n,0.5*dt,dt)
  # 2.5: solve for the final depth
  compute_depth!(h₂,dΩ,dω,Q,L2MMchol,h₁,F,dt)
end

function project_initial_conditions(dΩ, Q, V, S, L2MMchol, RTMMchol, H1MMchol, u₀, q₀)
  b₁(q)   = ∫(q*q₀)dΩ
  rhs1    = assemble_vector(b₁, Q)
  hn      = FEFunction(Q, copy(rhs1))

  b₂(v)   = ∫(v⋅u₀)dΩ
  rhs2    = assemble_vector(b₂, V)
  un      = FEFunction(V, copy(rhs2))

  qnv,unv=get_free_dof_values(qn,un)

  solve!(hnv, L2MMchol, hnv)
  solve!(unv, RTMMchol, unv)

  un, qn
end

function advection_hdg(
        model, order, degree,
        u₀, q₀, dt, N;
        mass_matrix_solver::Gridap.Algebra.LinearSolver=Gridap.Algebra.BackslashSolver(),
        write_diagnostics=true,
        write_diagnostics_freq=1,
        dump_diagnostics_on_screen=true,
        write_solution=false,
        write_solution_freq=N/10,
        output_dir="nswe_eq_ncells_$(num_cells(model))_order_$(order)_explicit")

  # Forward integration of the advection equation
  D = num_cell_dims(model)
  Ω = Triangulation(ReferenceFE{D},model)
  Γ = Triangulation(ReferenceFE{D-1},model)
  ∂K = GridapHybrid.Skeleton(model)

  reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order;space=:P)
  reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
  reffeₗ = ReferenceFE(lagrangian,Float64,order;space=:P)

  # Define test FESpaces
  V = TestFESpace(Ω, reffeᵤ; conformity=:L2)
  Q = TestFESpace(Ω, reffeₚ; conformity=:L2)
  M = TestFESpace(Γ, reffeₗ; conformity=:L2)

  U = TrialFESpace(V)
  P = TrialFESpace(Q)

  dΩ  = Measure(Ω,degree)
  n   = get_cell_normal_vector(∂K)
  nₒ  = get_cell_owner_normal_vector(∂K)
  d∂K = Measure(∂K,degree)

  # assemble the mass matrices
  H1MM, _, L2MM, H1MMchol, RTMMchol, L2MMchol =
    setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q;
                                      mass_matrix_solver=mass_matrix_solver)

  # Project the initial conditions onto the trial spaces
  un, qn =  project_initial_conditions(dΩ, Q, V, S,
                               L2MMchol, RTMMchol, H1MMchol, u₀, q₀)

  # work arrays
  h_tmp = copy(hnv)
  # build the potential vorticity lhs operator once just to initialise
  bmm(a,b) = ∫(a*hn*b)dΩ
  H1h      = assemble_matrix(bmm, R, S)
  H1hchol  = numerical_setup(symbolic_setup(mass_matrix_solver,H1h),H1h)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"nswe_diagnostics.csv")

    hm1    = clone_fe_function(Q,hn)
    hm2    = clone_fe_function(Q,hn)
    hp     = clone_fe_function(Q,hn)

    um1    = clone_fe_function(V,un)
    um2    = clone_fe_function(V,un)
    up     = clone_fe_function(V,un)

    for istep in 1:N
      h_aux = hm2
      hm2   = hm1
      hm1   = hn
      hn    = h_aux
      u_aux = um2
      um2   = um1
      um1   = un
      un    = u_aux

      advection_hdg_time_step!(qn, un, qp, up, H1h, H1hchol,
                                        model, dΩ, V, Q, R, S,
                                        RTMMchol, L2MMchol, dt)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        # compute mass and entropy conservation
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(Ω))
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
    pvdfile=joinpath(output_dir,"nswe_eq_ncells_$(num_cells(model))_order_$(order)_explicit")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
