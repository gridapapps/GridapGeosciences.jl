function setup_subassembled_RTMM(U,V,dΩ)
  a(a,b) = ∫(a⋅b)dΩ
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  dc=a(u,v)
  collect(dc[dΩ.quad.trian])
end

function setup_subassembled_invL2MM(P,Q,dΩ)
  a(a,b) = ∫(a⋅b)dΩ
  q = get_fe_basis(Q)
  p = get_trial_fe_basis(P)
  dc=a(p,q)
  lazy_map(inv,dc[dΩ.quad.trian])
end

function setup_subassembled_D(U,Q,dω)
  a(u,q) = ∫(q*DIV(u))*dω
  q = get_fe_basis(Q)
  u = get_trial_fe_basis(U)
  dc=a(u,q)
  dc[dω.quad.trian]
end

function setup_subassembled_invL2MMD(U,P,Q,dΩ,dω)
  invL2MM=setup_subassembled_invL2MM(P,Q,dΩ)
  D=setup_subassembled_D(U,Q,dω)
  collect(lazy_map(*,invL2MM,D))
end

function build_subassembled_A_dc(P,V,u,dΩ)
  a(p,v) = ∫(v⋅(u*p))dΩ
  p=get_trial_fe_basis(P)
  v=get_fe_basis(V)
  a(p,v)
end

function setup_subassembled_A(P,V,u,dΩ)
  dc=build_subassembled_A_dc(P,V,u,dΩ)
  collect(dc[dΩ.quad.trian])
end

function setup_subassembled_A!(A,P,V,u,dΩ)
  dc=build_subassembled_A_dc(P,V,u,dΩ)
  subassembled_A=dc[dΩ.quad.trian]
  c=array_cache(subassembled_A)
  for cell=1:length(A)
     A[cell] .= getindex!(c,subassembled_A,cell)
  end
end


function build_implicit_compound_operator_dc(P,V,dt,dΩ,A,RTMM,invL2MMD)
  dtA=lazy_map(Gridap.Fields.BroadcastingFieldOpMap(*), Fill(dt,length(RTMM)), A)
  dtAL2MMinvD=lazy_map(*,dtA,invL2MMD)
  RTMM_plus_dtAL2MMinvD=lazy_map(Gridap.Fields.BroadcastingFieldOpMap(+),RTMM,dtAL2MMinvD)
  dc = Gridap.CellData.DomainContribution()
  Gridap.CellData.add_contribution!(dc, dΩ.quad.trian, RTMM_plus_dtAL2MMinvD)
end

function assemble_implicit_compound_operator(P,U,V,dt,dΩ,A,RTMM,invL2MMD)
  dc=build_implicit_compound_operator_dc(P,V,dt,dΩ,A,RTMM,invL2MMD)
  a=SparseMatrixAssembler(U,V)
  assemble_matrix(a,Gridap.FESpaces.collect_cell_matrix(U,V,dc))
end

function assemble_implicit_compound_operator!(B,P,U,V,dt,dΩ,A,RTMM,invL2MMD)
  dc=build_implicit_compound_operator_dc(P,V,dt,dΩ,A,RTMM,invL2MMD)
  a=SparseMatrixAssembler(U,V)
  Gridap.FESpaces.assemble_matrix!(B,a,Gridap.FESpaces.collect_cell_matrix(U,V,dc))
end

function assemble_Asub_mul_u!(p, U, P, dΩ, A, u)
  ucell=Gridap.FESpaces.scatter_free_and_dirichlet_values(U,u,Float64[])
  A_mul_u=lazy_map(*, A, ucell)
  dc = Gridap.CellData.DomainContribution()
  Gridap.CellData.add_contribution!(dc, dΩ.quad.trian, A_mul_u)
  a = SparseMatrixAssembler(P,P)
  Gridap.FESpaces.assemble_vector!(p,a,Gridap.FESpaces.collect_cell_vector(P,dc))
end

function shallow_water_imex_time_step!(
     h₂, u₂, uₚ, ϕ, F, q₁, q₂,                               # in/out args
     H1h, H1hchol, h_wrk, u_wrk, A, B, Bchol,                # more in/out args
     model, dΩ, dω, U, V, P, Q, R, S, f, g, h₁, u₁, u₀,         # in args
     RTMMchol, L2MMchol, RTMM, L2MMinvD, dt, τ, leap_frog)   # more in args

  # energetically balanced implicit-explicit second order shallow water solver
  # reference: eqns (31-33) of
  # https://github.com/BOM-Monash-Collaborations/articles/blob/main/energetically_balanced_time_integration/EnergeticallyBalancedTimeIntegration_SW.tex

  n = get_normal_vector(model)
  # explicit step for provisional velocity, uₚ
  dt1 = dt
  if leap_frog
    dt1 = 2.0*dt
  end

  Fv,h₁v,h₂v = get_free_dof_values(F,h₁,h₂)

  # 1.1: the mass flux
  compute_mass_flux!(F,dΩ,V,RTMMchol,u₁*h₁)
  # 1.2: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,u₁⋅u₁,h₁,g)
  # 1.3: the potential vorticity
  compute_potential_vorticity!(q₁,H1h,H1hchol,dΩ,R,S,h₁,u₁,f,n)
  # 1.4: solve for the provisional velocity
  compute_velocity!(uₚ,dΩ,dω,V,RTMMchol,u₀,q₁-τ*u₁⋅∇(q₁),F,ϕ,n,dt1,dt1)

  # solve for the second order, semi-implicit mass flux
  # 2.1: mass flux component using the previous depth (explicit)
  compute_mass_flux!(F,dΩ,V,RTMMchol,h₁*(2.0*u₁ + uₚ)/6.0)
  # 2.2: mass flux component using the current depth (implicit)
  setup_subassembled_A!(A,P,V,(u₁ + 2.0*uₚ)/6.0,dΩ)
  assemble_implicit_compound_operator!(B,P,U,V,dt,dΩ,A,RTMM,L2MMinvD) # B = RTMM + dt*A*L2MMinv*D
  assemble_Asub_mul_u!(h_wrk,U,P,dΩ,L2MMinvD,Fv)                  # h_wrk = L2MMinvD * Fv
  h_wrk .= h₁v .- dt .* h_wrk
  assemble_Asub_mul_u!(u_wrk, P, U, dΩ, A, h_wrk)                        # u_wrk = A * h_wrk
  lu!(Bchol, B)
  ldiv!(Bchol, u_wrk)
  # 2.3: combine the two mass flux components
  Fv .= Fv .+ u_wrk
  # 2.4: compute the divergence of the total mass flux
  assemble_Asub_mul_u!(h_wrk,U,P,dΩ,L2MMinvD,Fv)                  # h_wrk = L2MMinvD * Fv
  # 2.5: depth field at new time level
  h₂v .= h₁v .- dt .* h_wrk

  # 3.1: the bernoulli function
  compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,(u₁⋅u₁ + u₁⋅uₚ + uₚ⋅uₚ)/3.0,0.5*(h₁ + h₂),g)
  # 3.2: the potential vorticity
  compute_potential_vorticity!(q₂,H1h,H1hchol,dΩ,R,S,h₂,uₚ,f,n)
  # 3.3: solve for the final velocity
  compute_velocity!(u₂,dΩ,dω,V,RTMMchol,u₁,q₁-τ*u₁⋅∇(q₁)+q₂-τ*uₚ⋅∇(q₂),F,ϕ,n,0.5*dt,dt)
end



function shallow_water_imex_time_stepper(model, order, degree,
                        h₀, u₀, f₀, g,
                        dt, τ, N;
                        write_diagnostics=true,
                        write_diagnostics_freq=1,
                        dump_diagnostics_on_screen=true,
                        write_solution=false,
                        write_solution_freq=N/10,
                        output_dir="nswe_eq_ncells_$(num_cells(model))_order_$(order)_imex")

  # Forward integration of the shallow water equations
  Ω = Triangulation(model)
  dΩ = Measure(Ω, degree)
  dω = Measure(Ω, degree, ReferenceDomain())

  # Setup the trial and test spaces
  R, S, U, V, P, Q = setup_mixed_spaces(model, order)

  # assemble the mass matrices (RTMM not actually needed in assembled form)
  H1MM, _, L2MM, H1MMchol, RTMMchol, L2MMchol =
      setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)

  # Project the initial conditions onto the trial spaces
  b₁(q)   = ∫(q*h₀)dΩ
  rhs1    = assemble_vector(b₁, Q)
  hn      = FEFunction(Q, copy(rhs1))
  ldiv!(L2MMchol, get_free_dof_values(hn))

  b₂(v)   = ∫(v⋅u₀)dΩ
  rhs2    = assemble_vector(b₂, V)
  un      = FEFunction(V, copy(rhs2))
  ldiv!(RTMMchol, get_free_dof_values(un))

  b₃(s)   = ∫(s*f₀)*dΩ
  rhs3    = assemble_vector(b₃, S)
  f       = FEFunction(S, copy(rhs3))
  ldiv!(H1MMchol, get_free_dof_values(f))

  # work arrays
  u_tmp = copy(get_free_dof_values(un))
  h_tmp = copy(get_free_dof_values(hn))
  w_tmp = copy(get_free_dof_values(f))

  # build the potential vorticity lhs operator once just to initialise
  bmm(a,b) = ∫(a*hn*b)dΩ
  H1h      = assemble_matrix(bmm, R, S)
  H1hchol  = lu(H1h)

  A        = setup_subassembled_A(P,V,un,dΩ)
  RTMM     = setup_subassembled_RTMM(U,V,dΩ)
  invL2MMD = setup_subassembled_invL2MMD(U,P,Q,dΩ,dω)

  B        = assemble_implicit_compound_operator(P,U,V,dt,dΩ,A,RTMM,invL2MMD)
  Bchol    = lu(B)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"nswe_diagnostics.csv")

    hm1    = clone_fe_function(Q,hn)
    ϕ      = clone_fe_function(Q,hn)

    um1    = clone_fe_function(V,un)
    um2    = clone_fe_function(V,un)
    up     = clone_fe_function(V,un)
    F      = clone_fe_function(V,un)

    wn     = clone_fe_function(S,f)
    q1     = clone_fe_function(S,f)
    q2     = clone_fe_function(S,f)

    # first step, no leap frog integration
    @time shallow_water_imex_time_step!(hn, un, up, ϕ, F, q1, q2,
				                          H1h, H1hchol, h_tmp, u_tmp, A, B, Bchol,
                                  model, dΩ, dω, U, V, P, Q, R, S, f, g, hm1, um1, um2,
                                  RTMMchol, L2MMchol, RTMM, invL2MMD, dt, τ, false)

    if (write_diagnostics)
      initialize_csv(diagnostics_file,"time", "mass", "vorticity", "kinetic", "potential", "power")
    end

    if (write_diagnostics && write_diagnostics_freq==1)
      compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
      dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                      model, dΩ, dω, S, L2MM, H1MM,
                                      hn, un, wn, ϕ, F, g, 1, dt,
                                      diagnostics_file,
                                      dump_diagnostics_on_screen)
    end

    # subsequent steps, do leap frog integration
    # (now that we have the state at two previous time levels)
    for istep in 2:N
      h_aux = hm1
      hm1   = hn
      hn    = h_aux
      u_aux = um2
      um2   = um1
      um1   = un
      un    = u_aux

      @time shallow_water_imex_time_step!(hn, un, up, ϕ, F, q1, q2,
				                            H1h, H1hchol, h_tmp, u_tmp, A, B, Bchol,
                                    model, dΩ, dω, U, V, P, Q, R, S, f, g, hm1, um1, um2,
                                    RTMMchol, L2MMchol, RTMM, invL2MMD, dt, τ, true)

      if (write_diagnostics && write_diagnostics_freq>0 && mod(istep, write_diagnostics_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
        dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                        model, dΩ, dω, S, L2MM, H1MM,
                                        hn, un, wn, ϕ, F, g, istep, dt,
                                        diagnostics_file,
                                        dump_diagnostics_on_screen)
      end
      if (write_solution && write_solution_freq>0 && mod(istep, write_solution_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, get_normal_vector(model))
        pvd[Float64(istep)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(istep)"),hn,un,wn)
      end
    end
    hn, un
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,"nswe_eq_ncells_$(num_cells(model))_order_$(order)_imex")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
