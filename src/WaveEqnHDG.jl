function wave_eqn_hdg_time_step!(
     pn, un, model, dΩ, ∂K, d∂K, X, Y, dt,
     assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X,Y))

  # Second order implicit advection
  # References:
  #   Muralikrishnan, Tran, Bui-Thanh, JCP, 2020 vol. 367
  #   Kang, Giraldo, Bui-Thanh, JCP, 2020 vol. 401
  γ     = 0.5*(2.0 - sqrt(2.0))
  γdt   = γ*dt
  γm1dt = (1.0 - γ)*dt

  n  = get_cell_normal_vector(∂K)
  nₒ = get_cell_owner_normal_vector(∂K)

  τ = 1.0

  # First stage
  b₁((q,v,m)) = ∫(q*pn)dΩ +
                ∫(v*un)dΩ - 
                ∫(m*0.0)d∂K

  a₁((p,u,l),(q,v,m)) = ∫(q*p)dΩ + ∫(γdt*τ*(nₒ⋅n)*q*p)d∂K -       # [q,p] block
                        ∫(γdt*(∇(q)*u))dΩ + ∫(γdt*q*(u⋅n))d∂K -   # [q,u] block
                        ∫(γdt*τ*(nₒ⋅n)*q*l)d∂Ω -                  # [q,l] block
                        ∫(γdt*(∇⋅v)*p)dΩ +                        # [v,p] block
                        ∫(v⋅u)dΩ +                                # [v,u] block
                        ∫(γdt*(v⋅n)*l)d∂Ω +                       # [v,l] block
                        ∫(τ*(nₒ⋅n)*m*p)d∂Ω +                      # [m,p] block
                        ∫((u⋅n)*m)d∂Ω -                           # [m,u] block
                        ∫(τ*(nₒ⋅n)*m*l)d∂Ω                        # [m,l] block

  op₁      = HybridAffineFEOperator((x,y)->(a₁(x,y),b₁(y)), X, Y, [1,2], [3])
  Xh       = solve(op₁)
  ph,uh,lh = Xh

  # Second stage
  b₂((q,v,m)) = ∫(q*pn)dΩ + ∫(γm1dt*τ*(nₒ⋅n)*q*ph)d∂K -              # [q] rhs
                ∫(γm1dt*(∇(q)*uh))dΩ + ∫(γm1dt*q*(uh⋅n))d∂K -        # [q] rhs
                ∫(γm1dt*τ*(nₒ⋅n)*q*lh)d∂Ω -                          # [q] rhs
                ∫(γm1dt*(∇⋅v)*ph)dΩ +                                # [v] rhs
                ∫(v⋅un)dΩ +                                          # [v] rhs
                ∫(γm1dt*(v⋅n)*lh)d∂Ω +                               # [v] rhs
                ∫(γm1*τ*(nₒ⋅n)*m*ph)d∂Ω +                            # [m] rhs
                ∫(γm1*(uh⋅n)*m)d∂Ω -                                 # [m] rhs
                ∫(γm1*τ*(nₒ⋅n)*m*lh)d∂Ω                              # [m] rhs

  a₂((p,u,l),(q,v,m)) = ∫(q*p)dΩ + ∫(γdt*τ*(nₒ⋅n)*q*p)d∂K -      # [q,p] block
                        ∫(γdt*(∇(q)*u))dΩ + ∫(γdt*q*(u⋅n))d∂K -  # [q,u] block
                        ∫(γdt*τ*(nₒ⋅n)*q*l)d∂Ω -                 # [q,l] block
                        ∫(γdt*(∇⋅v)*p)dΩ +                       # [v,p] block
                        ∫(v⋅u)dΩ +                               # [v,u] block
                        ∫(γdt*(v⋅n)*l)d∂Ω +                      # [v,l] block
                        ∫(γ*τ*(nₒ⋅n)*m*p)d∂Ω +                   # [m,p] block
                        ∫(γ*(u⋅n)*m)d∂Ω -                        # [m,u] block
                        ∫(γ*τ*(nₒ⋅n)*m*l)d∂Ω                     # [m,l] block

  op₂      = HybridAffineFEOperator((x,y)->(a₂(x,y),b₂(y)), X, Y, [1,2], [3])
  Xm       = solve(op₂)
  pm,um,_  = Xm

  get_free_dof_values(pn) .= get_free_dof_values(pm)
  get_free_dof_values(un) .= get_free_dof_values(um)
end

function conservation_wave_eqn(L2MM, U2MM, pnv, unv, p_tmp, u_tmp, mass_0, energy_0)
  mul!(p_tmp, L2MM, pnv)
  mul!(u_tmp, U2MM, unv)
  mass_con   = sum(p_tmp)
  energy_con = p_tmp⋅pnv + u_tmp⋅unv
  mass_con   = (mass_con - mass_0)/mass_0
  energy_con = (energy_con - energy_0)/energy_0
  mass_con, energy_con
end

function wave_eqn_hdg(
        model, order, degree,
        u₀, p₀, dt, N;
        mass_matrix_solver::Gridap.Algebra.LinearSolver=Gridap.Algebra.BackslashSolver(),
        write_diagnostics=true,
        write_diagnostics_freq=1,
        dump_diagnostics_on_screen=true,
        write_solution=false,
        write_solution_freq=N/10,
        output_dir="wave_eq_ncells_$(num_cells(model))_order_$(order)")

  # Forward integration of the wave equation
  D = num_cell_dims(model)
  Ω = Triangulation(ReferenceFE{D},model)
  Γ = Triangulation(ReferenceFE{D-1},model)
  ∂K = GridapHybrid.Skeleton(model)

  reffeᵤ = ReferenceFE(lagrangian,VectorValue{3,Float64},order;space=:P)
  reffeₚ = ReferenceFE(lagrangian,Float64,order;space=:P)
  reffeₗ = ReferenceFE(lagrangian,Float64,order;space=:P)

  # Define test FESpaces
  V = TestFESpace(Ω, reffeᵤ; conformity=:L2)
  Q = TestFESpace(Ω, reffeₚ; conformity=:L2)
  M = TestFESpace(Γ, reffeₗ; conformity=:L2)
  Y = MultiFieldFESpace([Q,V,M])

  U = TrialFESpace(V)
  P = TrialFESpace(Q)
  L = TrialFESpace(M)
  X = MultiFieldFESpace([P,U,L])

  dΩ  = Measure(Ω,degree)
  d∂K = Measure(∂K,degree)

  @printf("time step: %14.9e\n", dt)
  @printf("number of time steps: %u\n", N)

  # Project the initial conditions onto the trial spaces
  pn, pnv, L2MM, un, unv, U2MM = project_initial_conditions(dΩ, P, Q, p₀, U, V, u₀, mass_matrix_solver)

  # Work array
  p_tmp = copy(pnv)
  u_tmp = copy(unv)
  p_ic  = copy(pnv)
  u_ic  = copy(unv)

  # Initial states
  mul!(p_tmp, L2MM, pnv)
  p01 = sum(p_tmp)             # total mass
  mul!(u_tmp, U2MM, unv)
  p02 = p_tmp⋅pnv + u_tmp⋅unv  # total energy

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"wave_diagnostics.csv")
    if (write_diagnostics)
      initialize_csv(diagnostics_file,"step", "mass", "entropy")
    end
    if (write_solution && write_solution_freq>0)
      pvd[dt*Float64(0)] = new_vtk_step(Ω,joinpath(output_dir,"n=0"),["pn"=>pn,"un"=>un])
    end

    for istep in 1:N
      wave_eqn_hdg_time_step!(pn, un, model, dΩ, ∂K, d∂K, X, Y, dt)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        # compute mass and entropy conservation
        pn1, pn2 = conservation_wave_eqn(L2MM, pnv, p_tmp, U2MM, unv, u_tmp, p01, p02)
        if dump_diagnostics_on_screen
          @printf("%5d\t%14.9e\t%14.9e\n", istep, pn1, pn2)
        end
        append_to_csv(diagnostics_file; step=istep, mass=pn1, entropy=pn2)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        pvd[dt*Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),["pn"=>pn,"un"=>un])
      end
    end
    pn1, pn2 = conservation_wave_eqn(L2MM, pnv, p_tmp, U2MM, unv, u_tmp, p01, p02)
    pn1, pn2
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,"wave_eq_ncells_$(num_cells(model))_order_$(order)")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
