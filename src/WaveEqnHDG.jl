function wave_eqn_hdg_time_step!(
     pn, un, model, dΩ, ∂K, d∂K, X, Y, dt,
     assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X,Y))

  # Second order implicit linear wave equation
  # References:
  #   Muralikrishnan, Tran, Bui-Thanh, JCP, 2020 vol. 367
  #   Kang, Giraldo, Bui-Thanh, JCP, 2020 vol. 401
  γ     = 0.5*(2.0 - sqrt(2.0))
  γm1   = (1.0 - γ)
  γdt   = γ*dt
  γm1dt = γm1*dt

  n = get_cell_normal_vector(∂K)

  τ = 1.0

  # First stage
  b₁((q,v,m)) = ∫(q*pn)dΩ +
                ∫(v⋅un)dΩ -
                ∫(m*0.0)d∂K

  a₁((p,u,l),(q,v,m)) = ∫(q*p)dΩ + ∫(γdt*τ*q*p)d∂K -             # [q,p] block
                        ∫(γdt*(∇(q)⋅u))dΩ + ∫(γdt*q*(u⋅n))d∂K -  # [q,u] block
                        ∫(γdt*τ*q*l)d∂K -                        # [q,l] block
                        ∫(γdt*(∇⋅v)*p)dΩ +                       # [v,p] block
                        ∫(v⋅u)dΩ +                               # [v,u] block
                        ∫(γdt*(v⋅n)*l)d∂K +                      # [v,l] block
                        ∫(τ*m*p)d∂K +                            # [m,p] block
                        ∫((u⋅n)*m)d∂K -                          # [m,u] block
                        ∫(τ*m*l)d∂K                              # [m,l] block

  op₁      = HybridAffineFEOperator((x,y)->(a₁(x,y),b₁(y)), X, Y, [1,2], [3])
  Xh       = solve(op₁)
  ph,uh,lh = Xh

  # Second stage
  b₂((q,v,m)) = ∫(q*pn)dΩ - ∫(γm1dt*τ*q*ph)d∂K +                     # [q] rhs
                ∫(γm1dt*(∇(q)⋅uh))dΩ - ∫(γm1dt*q*(uh⋅n))d∂K +        # [q] rhs
                ∫(γm1dt*τ*q*lh)d∂K +                                 # [q] rhs
                ∫(γm1dt*(∇⋅v)*ph)dΩ +                                # [v] rhs
                ∫(v⋅un)dΩ -                                          # [v] rhs
                ∫(γm1dt*(v⋅n)*lh)d∂K -                               # [v] rhs
                ∫(γm1*τ*m*ph)d∂K -                                   # [m] rhs
                ∫(γm1*(uh⋅n)*m)d∂K +                                 # [m] rhs
                ∫(γm1*τ*m*lh)d∂K                                     # [m] rhs

  a₂((p,u,l),(q,v,m)) = ∫(q*p)dΩ + ∫(γdt*τ*q*p)d∂K -             # [q,p] block
                        ∫(γdt*(∇(q)⋅u))dΩ + ∫(γdt*q*(u⋅n))d∂K -  # [q,u] block
                        ∫(γdt*τ*q*l)d∂K -                        # [q,l] block
                        ∫(γdt*(∇⋅v)*p)dΩ +                       # [v,p] block
                        ∫(v⋅u)dΩ +                               # [v,u] block
                        ∫(γdt*(v⋅n)*l)d∂K +                      # [v,l] block
                        ∫(γ*τ*m*p)d∂K +                          # [m,p] block
                        ∫(γ*(u⋅n)*m)d∂K -                        # [m,u] block
                        ∫(γ*τ*m*l)d∂K                            # [m,l] block

  op₂      = HybridAffineFEOperator((x,y)->(a₂(x,y),b₂(y)), X, Y, [1,2], [3])
  Xm       = solve(op₂)
  pm,um,_  = Xm

  get_free_dof_values(pn) .= get_free_dof_values(pm)
  get_free_dof_values(un) .= get_free_dof_values(um)
end

function wave_eqn_hdg_time_step_2!(
     pn, un, model, dΩ, ∂K, d∂K, X, Y, dt, uo,
     assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X,Y))

  # Second order implicit linear wave equation
  # References:
  #   Muralikrishnan, Tran, Bui-Thanh, JCP, 2020 vol. 367
  #   Kang, Giraldo, Bui-Thanh, JCP, 2020 vol. 401
  γ     = 0.5*(2.0 - sqrt(2.0))
  γm1   = (1.0 - γ)
  γdt   = γ*dt
  γm1dt = γm1*dt

  nₑ = get_cell_normal_vector(∂K)
  nᵣ = get_normal_vector(Triangulation(model))
  nₒ = get_cell_owner_normal_vector(∂K)

  τ = 1.0

  # First stage
  b₁((q,v,m,s)) = ∫(q*pn)dΩ +
                  ∫(v⋅un)dΩ +
                  ∫(m*0.0)d∂K +
                  ∫(s⋅uo)d∂K

  a₁((p,u,l,r),(q,v,m,s)) = ∫(q*p)dΩ + ∫(γdt*τ*q*p)d∂K -             # [q,p] block
                            ∫(γdt*(∇(q)⋅u))dΩ + ∫(γdt*q*(u⋅nₑ))d∂K - # [q,u] block
                            ∫(γdt*τ*q*l)d∂K -                        # [q,l] block
                            ∫(γdt*(∇⋅v)*p)dΩ +                       # [v,p] block
                            ∫(v⋅u)dΩ +                               # [v,u] block
                            ∫(γdt*(v⋅nₑ)*l)d∂K +                     # [v,l] block
                            ∫(τ*m*p)d∂K +                            # [m,p] block
                            ∫((u⋅nₑ)*m)d∂K -                         # [m,u] block
                            ∫(τ*m*l)d∂K +                            # [m,l] block
			    ∫(τ*((v×nₑ)⋅((nₑ×(u×nₑ))×nₑ)))d∂K -
			    ∫(τ*((v×nₑ)⋅((nₑ×(r×nₑ))×nₑ)))d∂K -
			    ∫(τ*((s×nₑ)⋅((nₑ×(u×nₑ))×nₑ)))d∂K +
			    ∫(τ*((s×nₑ)⋅((nₑ×(r×nₑ))×nₑ)))d∂K

  op₁         = HybridAffineFEOperator((x,y)->(a₁(x,y),b₁(y)), X, Y, [1,2], [3,4])
  Xh          = solve(op₁)
  ph,uh,lh,rh = Xh

  # Second stage
  b₂((q,v,m,s)) = ∫(q*pn)dΩ - ∫(γm1dt*τ*q*ph)d∂K +                       # [q] rhs
                  ∫(γm1dt*(∇(q)⋅uh))dΩ - ∫(γm1dt*q*(uh⋅nₑ))d∂K +         # [q] rhs
                  ∫(γm1dt*τ*q*lh)d∂K +                                   # [q] rhs
                  ∫(γm1dt*(∇⋅v)*ph)dΩ +                                  # [v] rhs
                  ∫(v⋅un)dΩ -                                            # [v] rhs
                  ∫(γm1dt*(v⋅nₑ)*lh)d∂K -                                # [v] rhs
                  ∫(γm1*τ*m*ph)d∂K -                                     # [m] rhs
                  ∫(γm1*(uh⋅nₑ)*m)d∂K +                                  # [m] rhs
                  ∫(γm1*τ*m*lh)d∂K -                                     # [m] rhs
		  ∫(γm1*τ*((v×nₑ)⋅((nₑ×(uh×nₑ))×nₑ)))d∂K +
		  ∫(γm1*τ*((v×nₑ)⋅((nₑ×(rh×nₑ))×nₑ)))d∂K +
		  ∫(γm1*τ*((s×nₑ)⋅((nₑ×(uh×nₑ))×nₑ)))d∂K -
		  ∫(γm1*τ*((s×nₑ)⋅((nₑ×(rh×nₑ))×nₑ)))d∂K

  a₂((p,u,l,r),(q,v,m,s)) = ∫(q*p)dΩ + ∫(γdt*τ*q*p)d∂K -             # [q,p] block
                            ∫(γdt*(∇(q)⋅u))dΩ + ∫(γdt*q*(u⋅nₑ))d∂K - # [q,u] block
                            ∫(γdt*τ*q*l)d∂K -                        # [q,l] block
                            ∫(γdt*(∇⋅v)*p)dΩ +                       # [v,p] block
                            ∫(v⋅u)dΩ +                               # [v,u] block
                            ∫(γdt*(v⋅nₑ)*l)d∂K +                     # [v,l] block
                            ∫(γ*τ*m*p)d∂K +                          # [m,p] block
                            ∫(γ*(u⋅nₑ)*m)d∂K -                       # [m,u] block
                            ∫(γ*τ*m*l)d∂K +                          # [m,l] block
			    ∫(γ*τ*((v×nₑ)⋅((nₑ×(u×nₑ))×nₑ)))d∂K -
			    ∫(γ*τ*((v×nₑ)⋅((nₑ×(r×nₑ))×nₑ)))d∂K -
			    ∫(γ*τ*((s×nₑ)⋅((nₑ×(u×nₑ))×nₑ)))d∂K +
			    ∫(γ*τ*((s×nₑ)⋅((nₑ×(r×nₑ))×nₑ)))d∂K

  op₂        = HybridAffineFEOperator((x,y)->(a₂(x,y),b₂(y)), X, Y, [1,2], [3,4])
  Xm         = solve(op₂)
  pm,um,_,_  = Xm

  get_free_dof_values(pn) .= get_free_dof_values(pm)
  get_free_dof_values(un) .= get_free_dof_values(um)
end

function wave_eqn_hdg_time_step_3!(
     pn, un, model, dΩ, ∂K, d∂K, X, Y, dt, uo,
     assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X,Y))

  # Second order implicit linear wave equation
  # References:
  #   Muralikrishnan, Tran, Bui-Thanh, JCP, 2020 vol. 367
  #   Kang, Giraldo, Bui-Thanh, JCP, 2020 vol. 401
  γ     = 0.5*(2.0 - sqrt(2.0))
  γm1   = (1.0 - γ)
  γdt   = γ*dt
  γm1dt = γm1*dt

  nₑ = get_cell_normal_vector(∂K)
  nᵣ = get_normal_vector(Triangulation(model))
  nₒ = get_cell_owner_normal_vector(∂K)

  τ = 1.0

  # First stage
  b₁((q,v,m,s)) = ∫(q*pn)dΩ +
                  ∫(v⋅un)dΩ +
                  ∫(m*0.0)d∂K +
                  ∫(s⋅uo)d∂K

  a₁((p,u,l,r),(q,v,m,s)) = ∫(q*p)dΩ + ∫(γdt*τ*q*p)d∂K -             # [q,p] block
                            ∫(γdt*(∇(q)⋅u))dΩ + ∫(γdt*q*(u⋅nₑ))d∂K - # [q,u] block
                            ∫(γdt*τ*q*l)d∂K -                        # [q,l] block
                            ∫(γdt*(∇⋅v)*p)dΩ +                       # [v,p] block
                            ∫(v⋅u)dΩ +                               # [v,u] block
                            ∫(γdt*(v⋅nₑ)*l)d∂K +                     # [v,l] block
                            ∫(τ*m*p)d∂K +                            # [m,p] block
                            ∫((u⋅nₑ)*m)d∂K -                         # [m,u] block
                            ∫(τ*m*l)d∂K +                            # [m,l] block
			    ∫((v×nₑ)⋅((nₑ×(∇(p)×nₑ))×nₑ))d∂K +
			    ∫(τ*((v×nₑ)⋅((nₑ×(u×nₑ))×nₑ)))d∂K -
			    ∫(τ*((v×nₑ)⋅((nₑ×(r×nₑ))×nₑ)))d∂K -
			    ∫(((s×nₑ)⋅((nₑ×(∇(p)×nₑ))×nₑ)))d∂K -
			    ∫(τ*((s×nₑ)⋅((nₑ×(u×nₑ))×nₑ)))d∂K +
			    ∫(τ*((s×nₑ)⋅((nₑ×(r×nₑ))×nₑ)))d∂K

  op₁         = HybridAffineFEOperator((x,y)->(a₁(x,y),b₁(y)), X, Y, [1,2], [3,4])
  Xh          = solve(op₁)
  ph,uh,lh,rh = Xh

  # Second stage
  b₂((q,v,m,s)) = ∫(q*pn)dΩ - ∫(γm1dt*τ*q*ph)d∂K +                       # [q] rhs
                  ∫(γm1dt*(∇(q)⋅uh))dΩ - ∫(γm1dt*q*(uh⋅nₑ))d∂K +         # [q] rhs
                  ∫(γm1dt*τ*q*lh)d∂K +                                   # [q] rhs
                  ∫(γm1dt*(∇⋅v)*ph)dΩ +                                  # [v] rhs
                  ∫(v⋅un)dΩ -                                            # [v] rhs
                  ∫(γm1dt*(v⋅nₑ)*lh)d∂K -                                # [v] rhs
                  ∫(γm1*τ*m*ph)d∂K -                                     # [m] rhs
                  ∫(γm1*(uh⋅nₑ)*m)d∂K +                                  # [m] rhs
                  ∫(γm1*τ*m*lh)d∂K -                                     # [m] rhs
		  ∫(γm1*((v×nₑ)⋅((nₑ×(∇(ph)×nₑ))×nₑ)))d∂K -
		  ∫(γm1*τ*((v×nₑ)⋅((nₑ×(uh×nₑ))×nₑ)))d∂K +
		  ∫(γm1*τ*((v×nₑ)⋅((nₑ×(rh×nₑ))×nₑ)))d∂K +
		  ∫(γm1*((s×nₑ)⋅((nₑ×(∇(ph)×nₑ))×nₑ)))d∂K +
		  ∫(γm1*τ*((s×nₑ)⋅((nₑ×(uh×nₑ))×nₑ)))d∂K -
		  ∫(γm1*τ*((s×nₑ)⋅((nₑ×(rh×nₑ))×nₑ)))d∂K

  a₂((p,u,l,r),(q,v,m,s)) = ∫(q*p)dΩ + ∫(γdt*τ*q*p)d∂K -             # [q,p] block
                            ∫(γdt*(∇(q)⋅u))dΩ + ∫(γdt*q*(u⋅nₑ))d∂K - # [q,u] block
                            ∫(γdt*τ*q*l)d∂K -                        # [q,l] block
                            ∫(γdt*(∇⋅v)*p)dΩ +                       # [v,p] block
                            ∫(v⋅u)dΩ +                               # [v,u] block
                            ∫(γdt*(v⋅nₑ)*l)d∂K +                     # [v,l] block
                            ∫(γ*τ*m*p)d∂K +                          # [m,p] block
                            ∫(γ*(u⋅nₑ)*m)d∂K -                       # [m,u] block
                            ∫(γ*τ*m*l)d∂K +                          # [m,l] block
			    ∫(γ*((v×nₑ)⋅((nₑ×(∇(p)×nₑ))×nₑ)))d∂K +
			    ∫(γ*τ*((v×nₑ)⋅((nₑ×(u×nₑ))×nₑ)))d∂K -
			    ∫(γ*τ*((v×nₑ)⋅((nₑ×(r×nₑ))×nₑ)))d∂K -
			    ∫(γ*τ*((s×nₑ)⋅((nₑ×(∇(p)×nₑ))×nₑ)))d∂K -
			    ∫(γ*τ*((s×nₑ)⋅((nₑ×(u×nₑ))×nₑ)))d∂K +
			    ∫(γ*τ*((s×nₑ)⋅((nₑ×(r×nₑ))×nₑ)))d∂K

  op₂        = HybridAffineFEOperator((x,y)->(a₂(x,y),b₂(y)), X, Y, [1,2], [3,4])
  Xm         = solve(op₂)
  pm,um,_,_  = Xm

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

function compute_vorticity!(L2MM, un, p_tmp, model, P, Q, U, V, dΩ, ∂K, d∂K, L2MMchol, U2MMchol)
  nₑ = get_cell_normal_vector(∂K)
  nᵣ = get_normal_vector(Triangulation(model))

  #b₁(v)   = ∫(v⋅(∇×(un)))dΩ
  #a₁(u,v) = ∫(u⋅v)dΩ
  #op₁     = AffineFEOperator(a₁,b₁,U,V)
  #w3      = solve(op₁)

  #b₂(v)   = ∫(v*(w3⋅nᵣ))dΩ
  #a₂(u,v) = ∫(u*v)dΩ
  #op₂     = AffineFEOperator(a₂,b₂,P,Q)
  #wr      = solve(op₂)

  b₁(v) = ∫((∇×v)⋅un)dΩ + ∫((v×nₑ)⋅((nₑ×(un×nₑ))×nₑ))d∂K
  rhs₁  = assemble_vector(b₁, V)
  ω3    = FEFunction(V, copy(rhs₁))
  ω3v   = get_free_dof_values(ω3)
  solve!(ω3v, U2MMchol, ω3v)

  b₂(q) = ∫(q*(ω3⋅nᵣ))dΩ
  rhs₂  = assemble_vector(b₂, Q)
  ωr    = FEFunction(Q, copy(rhs₂))
  ωrv   = get_free_dof_values(ωr)
  solve!(ωrv, L2MMchol, ωrv)

  mul!(p_tmp, L2MM, ωrv)
  vort_tot = sum(p_tmp)

  vort_tot, ωr, ω3
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
  reffeᵣ = ReferenceFE(lagrangian,VectorValue{3,Float64},order;space=:P)

  # Define test FESpaces
  V = TestFESpace(Ω, reffeᵤ; conformity=:L2)
  Q = TestFESpace(Ω, reffeₚ; conformity=:L2)
  M = TestFESpace(Γ, reffeₗ; conformity=:L2)
  S = TestFESpace(Γ, reffeᵣ; conformity=:L2)
  Y = MultiFieldFESpace([Q,V,M])
  Y2 = MultiFieldFESpace([Q,V,M,S])

  U = TrialFESpace(V)
  P = TrialFESpace(Q)
  L = TrialFESpace(M)
  R = TrialFESpace(S)
  X = MultiFieldFESpace([P,U,L])
  X2 = MultiFieldFESpace([P,U,L,R])

  dΩ  = Measure(Ω,degree)
  d∂K = Measure(∂K,degree)

  @printf("time step: %14.9e\n", dt)
  @printf("number of time steps: %u\n", N)

  # Project the initial conditions onto the trial spaces
  pn, pnv, L2MM, un, unv, U2MM, L2MMchol, U2MMchol = project_initial_conditions_hdg(dΩ, P, Q, p₀, U, V, u₀, mass_matrix_solver)
  uo = FEFunction(V, copy(unv))

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
      initialize_csv(diagnostics_file,"step", "mass", "energy", "vorticity")
    end
    if (write_solution && write_solution_freq>0)
      pvd[dt*Float64(0)] = new_vtk_step(Ω,joinpath(output_dir,"n=0"),["pn"=>pn,"un"=>un])
    end

    for istep in 1:N
      #wave_eqn_hdg_time_step!(pn, un, model, dΩ, ∂K, d∂K, X, Y, dt)
      wave_eqn_hdg_time_step_2!(pn, un, model, dΩ, ∂K, d∂K, X2, Y2, dt, uo)
      #wave_eqn_hdg_time_step_3!(pn, un, model, dΩ, ∂K, d∂K, X2, Y2, dt, uo)

      vort_tot, ωr, ω3 = compute_vorticity!(L2MM, un, p_tmp, model, P, Q, U, V, dΩ, ∂K, d∂K, L2MMchol, U2MMchol)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        # compute mass and energy conservation
        pn1, pn2 = conservation_wave_eqn(L2MM, U2MM, pnv, unv, p_tmp, u_tmp, p01, p02)
        if dump_diagnostics_on_screen
          @printf("%5d\t%14.9e\t%14.9e\t%14.9e\n", istep, pn1, pn2, vort_tot)
        end
        append_to_csv(diagnostics_file; step=istep, mass=pn1, energy=pn2, vorticity=vort_tot)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        pvd[dt*Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),["pn"=>pn,"un"=>un,"wr"=>ωr,"w3"=>ω3])
      end
    end
    pn1, pn2 = conservation_wave_eqn(L2MM, U2MM, pnv, unv, p_tmp, u_tmp, p01, p02)
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
