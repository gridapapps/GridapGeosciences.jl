function shallow_water_hdg_time_step!(
     pn, un, wn, rn, fn, bₒ, grav, model, dΩ, ∂K, d∂K, X, Y, dt,
     assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X,Y))

  # Second order implicit shallow_water
  # References:
  #   Muralikrishnan, Tran, Bui-Thanh, JCP, 2020 vol. 367
  #   Kang, Giraldo, Bui-Thanh, JCP, 2020 vol. 401
  #   Nguyen, Peraire, Cockburn, JCP, 2011 vol. 230
  γ     = 0.5*(2.0 - sqrt(2.0))
  γm1   = (1.0 - γ)
  γdt   = γ*dt
  γm1dt = γm1*dt

  nₑ = get_cell_normal_vector(∂K)
  τ  = 1.0

  # Trial/test spaces:
  # A/B: vorticity
  # U/V: velocity
  # P/Q: depth
  # L/M: lagrange multiplier (depth)
  # R/S: lagrange multiplier (velocity)

  # First stage
  b₁((b,v,q,m,s)) = ∫(b⋅bₒ)dΩ +
                    ∫(v⋅un - γdt*(v⋅(wn×un)) + γdt*(∇⋅v)*(un⋅un))dΩ - ∫(γdt*(v⋅nₑ)*0.5*(rn⋅rn))d∂K +
                    ∫(q*pn)dΩ +
                    ∫(m*0.0)d∂K +
                    ∫(s⋅bₒ)d∂K

  a₁((a,u,p,l,r),(b,v,q,m,s)) = ∫(b⋅a)dΩ - ∫((∇×b)⋅u)dΩ - ∫((b×nₑ)⋅(nₑ×(r×nₑ)))d∂K    + # b equation
                                ∫(v⋅u + γdt*(v⋅(fn×u)))dΩ                             - # v equation
                                ∫(γdt*(∇⋅v)*grav*p)dΩ                                 + # ...
                                ∫(γdt*(v⋅nₑ)*grav*l)d∂K                               + # ...
                                ∫(q*p)dΩ - ∫(γdt*(∇(q)⋅u)*pn)dΩ                       + # q equation
                                ∫(γdt*pn*(u⋅nₑ)*q)d∂K + ∫(γdt*abs(un⋅nₑ)*q*p)d∂K      - # ...
                                ∫(γdt*abs(un⋅nₑ)*q*l)d∂K                              + # ...
                                ∫(pn*(u⋅nₑ)*m)d∂K + ∫(abs(un⋅nₑ)*p*m)d∂K              - # m equation
                                ∫(abs(un⋅nₑ)*l*m)d∂K                                  + # ...
                                ∫((nₑ×(s×nₑ))⋅(nₑ×a))d∂K                              + # s equation
                                ∫(τ*((nₑ×(s×nₑ))⋅(nₑ×(u×nₑ))))d∂K                     - # ...
                                ∫(τ*((nₑ×(s×nₑ))⋅(nₑ×(r×nₑ))))d∂K                       # ...

  op₁            = HybridAffineFEOperator((x,y)->(a₁(x,y),b₁(y)), X, Y, [1,2,3], [4,5])
  Xh             = solve(op₁)
  wh,uh,ph,lh,rh = Xh

  # Second stage
  b₂((b,v,q,m,s)) = ∫(b⋅bₒ)dΩ                                                         + # b equation
                    ∫(v⋅un - γm1dt*(v⋅(wh×uh)) - γdt*(v⋅(wn×un)))dΩ                   + # v equation
                    ∫(γm1dt*(∇⋅v)*0.5*(uh⋅uh))dΩ                                      + # ...
                    ∫(γdt*(∇⋅v)*0.5*(un⋅un))dΩ                                        + # ...
                    ∫(γm1dt*(∇⋅v)*grav*ph)dΩ                                          - # ...
                    ∫(γm1dt*(v⋅nₑ)*0.5*(rh⋅rh))d∂K                                    - # ...
                    ∫(γdt*(v⋅nₑ)*0.5*(rn⋅rn))d∂K                                      - # ...
                    ∫(γm1dt*(v⋅nₑ)*grav*lh)d∂K                                        + # ...
                    ∫(q*pn - γm1dt*(∇(q)⋅uh)*ph)dΩ                                    - # q equation
                    ∫(γm1dt*(uh⋅nₑ)*q*ph + γm1dt*abs(uh⋅nₑ)*q*ph)d∂K                  + # ...
                    ∫(γm1dt*abs(uh⋅nₑ)*q*lh)d∂K                                       - # ...
                    ∫(γm1*((uh⋅nₑ) + abs(uh⋅nₑ))*ph*m)d∂K + ∫(γm1*abs(uh⋅nₑ)*lh*m)d∂K - # m equation
                    ∫(γm1*((nₑ×(s×nₑ))⋅(nₑ×wh)))d∂K                                   - # s equation
                    ∫(τ*γm1*((nₑ×(s×nₑ))⋅(nₑ×(uh×nₑ))))d∂K                            + # ...
                    ∫(τ*γm1*((nₑ×(s×nₑ))⋅(nₑ×(rh×nₑ))))d∂K                              # ...

  a₂((a,u,p,l,r),(b,v,q,m,s)) = ∫(b⋅a)dΩ - ∫((∇×b)⋅u)dΩ - ∫((b×nₑ)⋅(nₑ×(r×nₑ)))d∂K    + # b equation
                                ∫(v⋅u + dt*(v⋅(fn×u)))dΩ                              - # v equation
                                ∫(γdt*(∇⋅v)*grav*p)dΩ                                 + # ...
                                ∫(γdt*(v⋅nₑ)*grav*l)d∂K                               + # ...
                                ∫(q*p)dΩ - ∫(γdt*(∇(q)⋅u)*ph)dΩ                       + # q equation
                                ∫(γdt*ph*(u⋅nₑ)*q)d∂K + ∫(γdt*abs(uh⋅nₑ)*q*p)d∂K      - # ...
                                ∫(γdt*abs(uh⋅nₑ)*q*l)d∂K                              + # ...
                                ∫(γ*ph*(u⋅nₑ)*m)d∂K + ∫(γ*abs(uh⋅nₑ)*p*m)d∂K          - # m equation
                                ∫(γ*abs(uh⋅nₑ)*l*m)d∂K                                + # ...
                                ∫(γ*((nₑ×(s×nₑ))⋅(nₑ×a)))d∂K                          + # s equation
                                ∫(τ*γ*((nₑ×(s×nₑ))⋅(nₑ×(u×nₑ))))d∂K                   - # ...
                                ∫(τ*γ*((nₑ×(s×nₑ))⋅(nₑ×(r×nₑ))))d∂K                     # ...

  op₂            = HybridAffineFEOperator((x,y)->(a₂(x,y),b₂(y)), X, Y, [1,2,3], [4,5])
  Xm             = solve(op₂)
  wm,um,pm,lm,rm = Xm

  get_free_dof_values(un) .= get_free_dof_values(um)
  get_free_dof_values(pn) .= get_free_dof_values(pm)
  get_free_dof_values(wn) .= get_free_dof_values(wm)

  rm
end

function shallow_water_hdg_time_step_2!(
     pn, un, wn, rn, fn, bₒ, grav, model, dΩ, ∂K, d∂K, X, Y, dt,
     assem=SparseMatrixAssembler(SparseMatrixCSC{Float64,Int},Vector{Float64},X,Y))

  # Second order implicit shallow_water
  # References:
  #   Muralikrishnan, Tran, Bui-Thanh, JCP, 2020 vol. 367
  #   Kang, Giraldo, Bui-Thanh, JCP, 2020 vol. 401
  #   Nguyen, Peraire, Cockburn, JCP, 2011 vol. 230
  nₑ = get_cell_normal_vector(∂K)
  τ  = 1.0

  # Trial/test spaces:
  # A/B: vorticity
  # U/V: velocity
  # P/Q: depth
  # L/M: lagrange multiplier (depth)
  # R/S: lagrange multiplier (velocity)

  # First stage
  b₁((b,v,q,m,s)) = ∫(b⋅bₒ)dΩ +
                    ∫(v⋅un - dt*(v⋅(wn×un)) + dt*(∇⋅v)*0.5*(un⋅un))dΩ - ∫(dt*(v⋅nₑ)*0.5*(rn⋅rn))d∂K +
                    ∫(q*pn)dΩ +
                    ∫(m*0.0)d∂K +
                    ∫(s⋅bₒ)d∂K

  a₁((a,u,p,l,r),(b,v,q,m,s)) = ∫(b⋅a)dΩ - ∫((∇×b)⋅u)dΩ - ∫((b×nₑ)⋅(nₑ×(r×nₑ)))d∂K      + # b equation
                                ∫(v⋅u + dt*(v⋅(fn×u)))dΩ                                - # v equation
                                ∫(dt*(∇⋅v)*grav*p)dΩ                                    + # ...
                                ∫(dt*(v⋅nₑ)*grav*l)d∂K                                  + # ...
                                ∫(q*p)dΩ - ∫(dt*(∇(q)⋅u)*pn)dΩ                          + # q equation
                                ∫(dt*pn*(u⋅nₑ)*q)d∂K + ∫(dt*abs(un⋅nₑ)*q*p)d∂K          - # ...
                                ∫(dt*abs(un⋅nₑ)*q*l)d∂K                                 + # ...
				∫(pn*(u⋅nₑ)*m)d∂K + ∫(abs(un⋅nₑ)*p*m)d∂K                - # m equation
				∫(abs(un⋅nₑ)*l*m)d∂K                                    + # ...
                                ∫((nₑ×(s×nₑ))⋅(nₑ×a))d∂K                                + # s equation
                                ∫(τ*((nₑ×(s×nₑ))⋅(nₑ×(u×nₑ))))d∂K                       - # ...
                                ∫(τ*((nₑ×(s×nₑ))⋅(nₑ×(r×nₑ))))d∂K                         # ...

  op₁            = HybridAffineFEOperator((x,y)->(a₁(x,y),b₁(y)), X, Y, [1,2,3], [4,5])
  Xh             = solve(op₁)
  wm,um,pm,lm,rm = Xh

  get_free_dof_values(un) .= get_free_dof_values(um)
  get_free_dof_values(pn) .= get_free_dof_values(pm)
  get_free_dof_values(wn) .= get_free_dof_values(wm)

  rm
end

function project_initial_conditions_sw_hdg(dΩ, ∂K, d∂K, p₀, u₀, P, Q, U, V, model, mass_matrix_solver)
  # the depth field
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

  # the vorticity field
  nₑ = get_cell_normal_vector(∂K)

  b₄(v) = ∫((∇×v)⋅un)dΩ + ∫((v×nₑ)⋅((nₑ×(un×nₑ))))d∂K
  rhs₄  = assemble_vector(b₄, V)
  ω3    = FEFunction(V, copy(rhs₄))
  ω3v   = get_free_dof_values(ω3)
  solve!(ω3v, U2MMchol, ω3v)

  pn, pnv, un, unv, ω3, ω3v, L2MM, U2MM, L2MMchol, U2MMchol
end

function project_vorticity_hdg(dΩ, ∂K, d∂K, u₀, b₀, A, B, R, S, model, mass_matrix_solver)
  nₑ = get_cell_normal_vector(∂K)
  τ  = 1.0

  X = MultiFieldFESpace([A,R])
  Y = MultiFieldFESpace([B,S])

  b₁((b,s)) = ∫((∇×b)⋅u₀)dΩ + ∫(τ*((nₑ×(s×nₑ))⋅(nₑ×(u₀×nₑ))))d∂K
  a₁((a,r),(b,s)) = ∫(b⋅a)dΩ - ∫((b×nₑ)⋅(nₑ×(r×nₑ)))d∂K    + # b equation
                    ∫((nₑ×(s×nₑ))⋅(nₑ×a))d∂K               + # s equation
                    ∫(τ*((nₑ×(s×nₑ))⋅(nₑ×(r×nₑ))))d∂K        # ...

  op₁   = HybridAffineFEOperator((x,y)->(a₁(x,y),b₁(y)), X, Y, [1], [2])
  Xh    = solve(op₁)
  wh,rh = Xh

  wh,rh
end

function get_radial_vorticity!(ωr, ω3, dΩ, Q, L2MMchol, model)
  nᵣ   = get_normal_vector(Triangulation(model))
  b(q) = ∫(q*(ω3⋅nᵣ))dΩ
  rhs  = assemble_vector(b, Q)
  ωrv  = get_free_dof_values(ωr)
  ωrv .= rhs
  solve!(ωrv, L2MMchol, ωrv)
end

function sw_conservation_hdg(L2MM, U2MM, pnv, unv, wnv, p_tmp, u_tmp, w_tmp, mass_0, vort_0)
  # mass conservation
  mul!(p_tmp, L2MM, pnv)
  mass_con = sum(p_tmp)
  mass_con = (mass_con - mass_0)/mass_0

  # 3d vorticity conservation
  mul!(w_tmp, U2MM, unv)
  vort_con = sum(w_tmp)
  vort_con = (vort_con - vort_0)

  mass_con, vort_con
end

function l2_error_norm(nv, ic, tmp, MM)
  mul!(tmp, MM, ic)
  l2_norm_sq = tmp⋅ic
  ic        .= ic .- nv
  mul!(tmp, MM, ic)
  l2_err_sq  = ic⋅tmp
  sqrt(l2_err_sq/l2_norm_sq)
end

function shallow_water_hdg(
        model, order, degree,
        p₀, u₀, f₀, grav, dt, N;
        mass_matrix_solver::Gridap.Algebra.LinearSolver=Gridap.Algebra.BackslashSolver(),
        write_diagnostics=true,
        write_diagnostics_freq=1,
        dump_diagnostics_on_screen=true,
        write_solution=false,
        write_solution_freq=N/10,
        output_dir="sw_eq_ncells_$(num_cells(model))_order_$(order)")

  # Forward integration of the shallow_water equation
  D = num_cell_dims(model)
  Ω = Triangulation(ReferenceFE{D},model)
  Γ = Triangulation(ReferenceFE{D-1},model)
  ∂K = GridapHybrid.Skeleton(model)

  reffeₐ = ReferenceFE(lagrangian,VectorValue{3,Float64},order;space=:P)
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{3,Float64},order;space=:P)
  reffeₚ = ReferenceFE(lagrangian,Float64,order;space=:P)
  reffeₗ = ReferenceFE(lagrangian,Float64,order;space=:P)
  reffeᵣ = ReferenceFE(lagrangian,VectorValue{3,Float64},order;space=:P)

  # Define test FESpaces
  B = TestFESpace(Ω, reffeₐ; conformity=:L2)
  V = TestFESpace(Ω, reffeᵤ; conformity=:L2)
  Q = TestFESpace(Ω, reffeₚ; conformity=:L2)
  M = TestFESpace(Γ, reffeₗ; conformity=:L2)
  S = TestFESpace(Γ, reffeᵣ; conformity=:L2)
  Y = MultiFieldFESpace([B,V,Q,M,S])

  A = TrialFESpace(B)
  U = TrialFESpace(V)
  P = TrialFESpace(Q)
  L = TrialFESpace(M)
  R = TrialFESpace(S)
  X = MultiFieldFESpace([A,U,P,L,R])

  dΩ  = Measure(Ω,degree)
  d∂K = Measure(∂K,degree)

  @printf("time step: %14.9e\n", dt)
  @printf("number of time steps: %u\n", N)

  # Project the initial conditions onto the trial spaces
  pn, pnv, un, unv, wn, wnv, L2MM, U2MM, L2MMchol, U2MMchol = 
    project_initial_conditions_sw_hdg(dΩ, ∂K, d∂K, p₀, u₀, P, Q, U, V, model, mass_matrix_solver)

  rn = FEFunction(V, copy(unv))
  bo = FEFunction(V, copy(unv))
  get_free_dof_values(bo) .= 0.0

  wn2,_ = project_vorticity_hdg(dΩ, ∂K, d∂K, u₀, bo, A, B, R, S, model, mass_matrix_solver)

  wr = FEFunction(Q, copy(pnv))
  get_radial_vorticity!(wr, wn, dΩ, Q, L2MMchol, model)
  wr2 = FEFunction(Q, copy(pnv))
  get_radial_vorticity!(wr2, wn2, dΩ, Q, L2MMchol, model)

  # Work array
  p_tmp = copy(pnv)
  u_tmp = copy(unv)
  w_tmp = copy(wnv)

  p_ic = copy(pnv)
  u_ic = copy(unv)
  w_ic = copy(wnv)

  # Initial states
  mul!(p_tmp, L2MM, pnv)
  mass_0 = sum(p_tmp)  # total mass
  mul!(w_tmp, U2MM, wnv)
  vort_0 = sum(w_tmp)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"sw_diagnostics.csv")
    if (write_diagnostics)
      initialize_csv(diagnostics_file,"step", "mass", "vorticity")
    end
    if (write_solution && write_solution_freq>0)
      pvd[dt*Float64(0)] = new_vtk_step(Ω,joinpath(output_dir,"n=0"),["pn"=>pn,"un"=>un,"wn"=>wn,"wr"=>wr,"wn2"=>wn2,"wr2"=>wr2])
    end

    for istep in 1:N
      rn = shallow_water_hdg_time_step!(pn, un, wn, rn, f₀, bo, grav, model, dΩ, ∂K, d∂K, X, Y, dt)
      #rn = shallow_water_hdg_time_step_2!(pn, un, wn, rn, f₀, bo, grav, model, dΩ, ∂K, d∂K, X, Y, dt)
      get_radial_vorticity!(wr, wn, dΩ, Q, L2MMchol, model)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        # compute mass and entropy conservation
        cons_mass, cons_vort = sw_conservation_hdg(L2MM, U2MM, pnv, unv, wnv, p_tmp, u_tmp, w_tmp, mass_0, vort_0)
        if dump_diagnostics_on_screen
          @printf("%5d\t%14.9e\t%14.9e\n", istep, cons_mass, cons_vort)
        end
        append_to_csv(diagnostics_file; step=istep, mass=cons_mass, vorticity=cons_vort)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        pvd[dt*Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),["pn"=>pn,"un"=>un,"wn"=>wn,"wr"=>wr])
      end
    end
    # compute the L2 error with respect to the inital condition after one rotation
    l2_err_u = l2_error_norm(unv, u_ic, u_tmp, U2MM)
    l2_err_p = l2_error_norm(pnv, p_ic, p_tmp, L2MM)
    l2_err_w = l2_error_norm(wnv, w_ic, w_tmp, U2MM)
    @printf("L2 errors: %14.9e\t%14.9e\t%14.9e\n", l2_err_u, l2_err_p, l2_err_w)
    cons_mass, cons_vort = sw_conservation_hdg(L2MM, U2MM, pnv, unv, wnv, p_tmp, u_tmp, w_tmp, mass_0, vort_0)
    l2_err_u, l2_err_p, l2_err_w, cons_mass, cons_vort
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,"sw_eq_ncells_$(num_cells(model))_order_$(order)")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
