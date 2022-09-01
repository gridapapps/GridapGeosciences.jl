function tang(n,v)
  (n×(v×n))
end

function quasi_geostrophic_hdg_time_step!(
     wn, un, fn, uo, model, dΩ, ∂K, d∂K, X1, Y1, X2, Y2, dt,
     assem1=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X1,Y1),
     assem2=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X2,Y2))

  # References:
  #   Muralikrishnan, Tran, Bui-Thanh, JCP, 2020 vol. 367
  #   Kang, Giraldo, Bui-Thanh, JCP, 2020 vol. 401
  #γ     = 0.5*(2.0 - sqrt(2.0))
  γ     = 1.0
  γm1   = (1.0 - γ)
  γdt   = γ*dt
  γm1dt = γm1*dt

  τ = 1.0

  nₑ = get_cell_normal_vector(∂K)
  nᵣ = get_normal_vector(Triangulation(model))

  # First stage

  # Elliptic equation for the velocity (u), stream function (a), velocity tangent multiplier (r)
  B₁((v,b,s)) = ∫(v⋅uo)dΩ + ∫(b*wn)dΩ + ∫(s⋅uo)d∂Ω
  A₁((u,a,r),(v,b,s)) = ∫(v⋅u)dΩ -
                        ∫((∇×v)*a)dΩ - ∫((v×nₑ)*a)d∂K -
			∫((v×nₑ)*τ*(tang(nₑ,u)⋅nᵣ))d∂K +           
			∫((v×nₑ)*τ*(tang(nₑ,r)⋅nᵣ))d∂K + 
                        ∫(perp(nᵣ,∇(b))⋅u)dΩ + 
                        ∫(b*(nₑ×(r×nₑ)))d∂K + 
			∫(tang(nₑ,s)*a)d∂K +
			∫(tang(nₑ,s)*τ*(tang(nₑ,u)⋅nᵣ))d∂K -
			∫(tang(nₑ,s)*τ*(tang(nₑ,r)⋅nᵣ))d∂K
  #A₁((u,a,r),(v,b,s)) = ∫(v⋅u)dΩ -
  #                      ∫((∇×v)*a)dΩ - ∫((v×nₑ)*a)d∂K -
  #                      ∫((v×nₑ)*τ*((nₑ×(u×nₑ))⋅nᵣ))d∂K +           
  #                      ∫((v×nₑ)*τ*((nₑ×(r×nₑ))⋅nᵣ))d∂K + 
  #                      ∫(perp(nᵣ,∇(b))⋅u)dΩ + 
  #                      ∫(b*(nₑ×(r×nₑ)))d∂K + 
  #                      ∫((nₑ×(s×nₑ))*a)d∂K +
  #                      ∫((nₑ×(s×nₑ))*τ*((nₑ×(u×nₑ))⋅nᵣ))d∂K -
  #                      ∫((nₑ×(s×nₑ))*τ*((nₑ×(r×nₑ))⋅nᵣ))d∂K

  OP₁     = HybridAffineFEOperator((u,v)->(A₁(u,v),B₁(v)), X2, Y2, [1,2], [3])
  Xh      = solve(OP₁)
  uh,ψh,_ = Xh

  # Advection equation for the 
  b₁((q,m)) = ∫(q*wn + γdt*(∇(q)⋅uh)*fn)dΩ - ∫(γdt*(uh⋅nₑ)*q*fn)d∂K
              ∫(m*0.0)d∂K

  a₁((p,l),(q,m)) = ∫(q*p - γdt*(∇(q)⋅uh)*p)dΩ + ∫(((uh⋅nₑ) + abs(uh⋅nₑ))*γdt*q*p)d∂K -  # [q,p] block
                    ∫(γdt*abs(uh⋅nₑ)*q*l)d∂K +                                           # [q,l] block
                    ∫(((uh⋅nₑ) + abs(uh⋅nₑ))*p*m)d∂K -                                   # [m,p] block
                    ∫(abs(uh⋅nₑ)*l*m)d∂K                                                 # [m,l] block

  op₁   = HybridAffineFEOperator((u,v)->(a₁(u,v),b₁(v)), X1, Y1, [1], [2])
  xh    = solve(op₁)
#  wh,lh = xh
  wm,lm = xh
  um    = uh

  # Second stage
  #b₂((q,m)) = ∫(q*wn + γm1dt*(∇(q)⋅un)*wh)dΩ - ∫(((un⋅n) + abs(un⋅n))*γm1dt*wh*q - abs(un⋅n)*γm1dt*lh*q)d∂K -
  #            ∫(γm1*((un⋅n) + abs(un⋅n))*wh*m - γm1*abs(un⋅n)*lh*m)d∂K
#
#  a₂((p,l),(q,m)) = ∫(q*p - γdt*(∇(q)⋅un)*p)dΩ + ∫(((un⋅n) + abs(un⋅n))*γdt*q*p)d∂K -    # [q,p] block
#                    ∫(γdt*abs(un⋅n)*q*l)d∂K +                                            # [q,l] block
#                    ∫(((un⋅n) + abs(un⋅n))*γ*p*m)d∂K -                                   # [m,p] block
#                    ∫(γ*abs(un⋅n)*l*m)d∂K                                                # [m,l] block

#  op₂   = HybridAffineFEOperator((u,v)->(a₂(u,v),b₂(v)), X1, Y1, [1], [2])
#  Xm    = solve(op₂)
#  wm,_  = Xm

  get_free_dof_values(wn) .= get_free_dof_values(wm)
  get_free_dof_values(un) .= get_free_dof_values(um)
end

function project_initial_conditions_hdg(dΩ, P, Q, p₀, U, V, u₀, mass_matrix_solver)
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
  b₂(v)    = ∫(v⋅u₀)dΩ
  a₂(u,v)  = ∫(v⋅u)dΩ
  rhs₂     = assemble_vector(b₂, V)
  U2MM     = assemble_matrix(a₂, U, V)
  U2MMchol = numerical_setup(symbolic_setup(mass_matrix_solver,U2MM),U2MM)
  un       = FEFunction(V, copy(rhs₂))
  unv      = get_free_dof_values(un)

  solve!(unv, U2MMchol, unv)

  pn, pnv, L2MM, un, unv, U2MM, L2MMchol, U2MMchol
end

function qg_conservation(L2MM, pnv, p_tmp, mass_0, entropy_0)
  mul!(p_tmp, L2MM, pnv)
  mass_con    = sum(p_tmp)
  entropy_con = p_tmp⋅pnv
  mass_con    = (mass_con - mass_0)
  entropy_con = (entropy_con - entropy_0)
  mass_con, entropy_con
end

function quasi_geostrophic_hdg(
        model, order, degree,
        p₀, u₀, f₀, dt, N;
        mass_matrix_solver::Gridap.Algebra.LinearSolver=Gridap.Algebra.BackslashSolver(),
        write_diagnostics=true,
        write_diagnostics_freq=1,
        dump_diagnostics_on_screen=true,
        write_solution=false,
        write_solution_freq=N/10,
        output_dir="qg_eq_ncells_$(num_cells(model))_order_$(order)")

  # Forward integration of the advection equation
  D = num_cell_dims(model)
  Ω = Triangulation(ReferenceFE{D},model)
  Γ = Triangulation(ReferenceFE{D-1},model)
  ∂K = GridapHybrid.Skeleton(model)

  reffeᵤ = ReferenceFE(lagrangian,VectorValue{3,Float64},order;space=:P)
  reffeₐ = ReferenceFE(lagrangian,Float64,order;space=:P)
  reffeₚ = ReferenceFE(lagrangian,Float64,order;space=:P)
  reffeₗ = ReferenceFE(lagrangian,Float64,order;space=:P)
  reffeᵣ = ReferenceFE(lagrangian,VectorValue{3,Float64},order;space=:P)

  # Define test FESpaces
  V = TestFESpace(Ω, reffeᵤ; conformity=:L2)
  Q = TestFESpace(Ω, reffeₚ; conformity=:L2)
  M = TestFESpace(Γ, reffeₗ; conformity=:L2)
  Y1 = MultiFieldFESpace([Q,M])
  B = TestFESpace(Ω, reffeₐ; conformity=:L2)
  S = TestFESpace(Γ, reffeᵣ; conformity=:L2)
  Y2 = MultiFieldFESpace([V,B,S])

  U = TrialFESpace(V)
  P = TrialFESpace(Q)
  L = TrialFESpace(M)
  X1 = MultiFieldFESpace([P,L])
  A = TrialFESpace(B)
  R = TrialFESpace(S)
  X2 = MultiFieldFESpace([U,A,R])

  dΩ  = Measure(Ω,degree)
  d∂K = Measure(∂K,degree)

  @printf("time step: %14.9e\n", dt)
  @printf("number of time steps: %u\n", N)

  # Project the initial conditions onto the trial spaces
  pn, pnv, L2MM, un, unv, U2MM, L2MMchol, U2MMchol = project_initial_conditions_hdg(dΩ, P, Q, p₀, U, V, u₀, mass_matrix_solver)

  # Work array
  p_tmp = copy(pnv)
  p_ic = copy(pnv)
  uo = FEFunction(V, copy(unv))
  get_free_dof_values(uo) .= 0.0

  # Initial states
  mul!(p_tmp, L2MM, pnv)
  p01 = sum(p_tmp)  # total mass
  p02 = p_tmp⋅pnv   # total entropy
  @printf("Initial vorticity and enstrophy:\t%14.9e\t%14.9e\n", p01, p02)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"qg_diagnostics.csv")
    if (write_diagnostics)
      initialize_csv(diagnostics_file,"step", "mass", "entropy")
    end
    if (write_solution && write_solution_freq>0)
      pvd[dt*Float64(0)] = new_vtk_step(Ω,joinpath(output_dir,"n=0"),["pn"=>pn,"un"=>un])
    end

    for istep in 1:N
      quasi_geostrophic_hdg_time_step!(pn, un, f₀, uo, model, dΩ, ∂K, d∂K, X1, Y1, X2, Y2, dt)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        # compute mass and entropy conservation
        pn1, pn2 = qg_conservation(L2MM, pnv, p_tmp, p01, p02)
        if dump_diagnostics_on_screen
          @printf("%5d\t%14.9e\t%14.9e\n", istep, pn1, pn2)
        end
        append_to_csv(diagnostics_file; step=istep, mass=pn1, entropy=pn2)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        pvd[dt*Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),["pn"=>pn,"un"=>un])
      end
    end
    # compute the L2 error with respect to the inital condition after one rotation
    mul!(p_tmp, L2MM, p_ic)
    l2_norm_sq = p_tmp⋅p_ic
    p_ic      .= p_ic .- pnv
    mul!(p_tmp, L2MM, p_ic)
    l2_err_sq  = p_ic⋅p_tmp
    l2_err     = sqrt(l2_err_sq/l2_norm_sq)
    @printf("L2 error: %14.9e\n", l2_err)
    pn1, pn2   = qg_conservation(L2MM, pnv, p_tmp, p01, p02)
    l2_err, pn1, pn2
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,"qg_eq_ncells_$(num_cells(model))_order_$(order)")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
