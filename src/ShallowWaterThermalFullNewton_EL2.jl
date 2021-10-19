# Generate initial monolothic solution
function uhEF₀(u₀,h₀,E₀,F₀,X,Y,dΩ)
  a((u,p,p2,u2),(v,q,q2,v2))=∫(v⋅u+q*p+q2*p2+v2⋅u2)dΩ
  b((v,q,p2,v2))=∫( v⋅u₀+ q*h₀ + q2*E₀ + v2⋅F₀ )dΩ
  solve(AffineFEOperator(a,b,X,Y))
end

"""
  Solves the nonlinear rotating shallow water equations
  T : [0,T] simulation interval
  N : number of time subintervals
  θ : Theta-method parameter [0,1)
  τ : APVM method stabilization parameter (dt/2 is typically a reasonable value)
"""
function shallow_water_theta_method_full_newton_time_stepper(
      model, order, degree, h₀, u₀, E₀, f₀, topography, g, θ, T, N, τ;
      nlrtol=1.0e-08, # Newton solver relative residual tolerance
      linear_solver::Gridap.Algebra.LinearSolver=Gridap.Algebra.BackslashSolver(),
      sparse_matrix_type::Type{<:AbstractSparseMatrix}=SparseMatrixCSC{Float64,Int},
      write_diagnostics=true,
      write_diagnostics_freq=1,
      dump_diagnostics_on_screen=true,
      write_solution=false,
      write_solution_freq=N/10,
      output_dir="nswe_ncells_$(num_cells(model))_order_$(order)_theta_method_full_newton")

  Ω  = Triangulation(model)
  n  = get_normal_vector(model)
  dΩ = Measure(Ω,degree)
  dω = Measure(Ω,degree,ReferenceDomain())

  # Setup the trial and test spaces
  R, S, U, V, P, Q = setup_mixed_spaces(model, order)

  if (write_diagnostics)
    # assemble the mass matrices
    H1MM, _, L2MM, H1MMchol, RTMMchol, L2MMchol = setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)
  end

  Y = MultiFieldFESpace([V,Q,Q,V])
  X = MultiFieldFESpace([U,P,P,U])

  a1(u,v)=∫(v⋅u)dΩ
  l1(v)=∫(v⋅u₀)dΩ
  un=solve(AffineFEOperator(a1,l1,U,V))

  a2(u,v)=∫(v*u)dΩ
  l2(v)=∫(v*h₀)dΩ
  hn=solve(AffineFEOperator(a2,l2,P,Q))

  a2(u,v)=∫(v*u)dΩ
  l2(v)=∫(v*E₀)dΩ
  En=solve(AffineFEOperator(a2,l2,P,Q))

  a3(u,v)=∫(v*u)dΩ
  l3(v)=∫(v*f₀)dΩ
  fn=solve(AffineFEOperator(a3,l3,R,S))

  unv,hnv,Env,fnv=get_free_dof_values(un,hn,En,fn)

  b = interpolate_everywhere(topography,P)

  # Compute:
  #     - Initial potential vorticity (q₀)
  #     - Initial volume flux (F₀)
  #     - Initial buoyancy (e₀)
  #     - Initial full solution
  qn=clone_fe_function(R,fn)
  Fn=clone_fe_function(V,un)
  en=clone_fe_function(R,fn)
  compute_potential_vorticity!(qn,H1MM,H1MMchol,dΩ,R,S,hn,un,fn,n)
  compute_buoyancy!(en,dΩ,S,H1MMchol,En)
  compute_mass_flux!(Fn,dΩ,V,RTMMchol,un*hn)
  uhEF=uhEF₀(un,hn,En,Fn,X,Y,dΩ)
  u,h,q,F = uhEF

  h_tmp = copy(hnv)
  w_tmp = copy(fnv)
  qₖ=clone_fe_function(R,fn)
  eₖ=clone_fe_function(R,fn)
  Fₖ=clone_fe_function(V,un)
  ϕₖ=clone_fe_function(Q,hn)
  eF=clone_fe_function(V,un)
  dT=clone_fe_function(V,un)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"tswe_diagnostics_full_newton_EL2.csv")

    dt  = T/N
    τ   = dt/2 # APVM stabilization parameter
    hc  = CellField(h₀,Ω)
    uc  = CellField(u₀,Ω)

    if (write_diagnostics)
      wn = clone_fe_function(S,fn)
      initialize_csv(diagnostics_file, "time", "mass", "vorticity", "buoyancy", "kinetic", "potential", "power_k2p", "power_k2i")
    end

    for step=1:N
       compute_potential_vorticity!(qn,H1MM,H1MMchol,dΩ,R,S,hn,un,fn,n)
       compute_buoyancy!(en,dΩ,S,H1MMchol,En)

       function residual((u,h,E,F),(v,q,q2,v2))
         compute_mass_flux!(Fₖ,dΩ,V,RTMMchol,un*(2.0*hn + h)/6.0+u*(hn + 2.0*h)/6.0)
         compute_bernoulli_potential!(ϕₖ,dΩ,Q,L2MMchol,(un⋅un + un⋅u + u⋅u)/3.0,0.5*(En + E),0.5)
         compute_potential_vorticity!(qₖ,H1MM,H1MMchol,dΩ,R,S,h,u,fn,n)
         compute_buoyancy!(eₖ,dΩ,S,H1MMchol,E)
	 compute_buoyancy_flux!(eF,dΩ,V,RTMMchol,0.5*(en+eₖ),Fₖ)
         compute_temperature_gradient!(dT,dω,V,RTMMchol,0.25*(hn+h))

         ∫((1.0/dt)*v⋅(u-un) + 
	   0.5*(qn+q-τ*(un⋅∇(qn))-τ*(u⋅∇(q)))*(v⋅⟂(Fₖ,n)) +
	   0.5*(en+eₖ-τ*(un⋅∇(en))-τ*(u⋅∇(eₖ)))*v⋅dT)dΩ - 
           ∫(DIV(v)*ϕₖ)dω +                                     # eq1
         ∫((1.0/dt)*q*(h-hn))dΩ + ∫(q*(DIV(Fₖ)))dω +            # eq2
         ∫((1.0/dt)*q*(E-En))dΩ + ∫(q*(DIV(eF)))dω +            # eq3
         ∫(v2⋅(F-un*(2.0*hn + h)/6.0+u*(hn + 2.0*h)/6.0))dΩ     # eq4
       end
       function jacobian((u,h,E,F),(du,dh,dE,dF),(v,q,q2,v2))
         ∫((1.0/dt)*v⋅du +  (dq    - τ*(uiΔu⋅∇(dq)+uidu⋅∇(qvort)))*(v⋅⟂(F ,n))
                         +  (qvort - τ*(           uiΔu⋅∇(qvort)))*(v⋅⟂(dF,n))
                         -  (∇⋅(v))*(g*hidh +uiΔu⋅uidu)    +  # eq1
           (1.0/dt)*q*dh)dΩ + ∫(q*(DIV(dF)))dω             +  # eq2
           ∫(s*(qvort*hidh+dq*hiΔh) + ⟂(∇(s),n)⋅uidu       +  # eq3
             v2⋅(dF-hiΔh*uidu-hidh*uiΔu))dΩ                   # eq4
       end

       # Solve fully-coupled monolithic nonlinear problem
       # Use previous time-step solution, ΔuΔhqF, as initial guess
       # Overwrite solution into uhqF

       # Adjust absolute tolerance ftol s.t. it actually becomes relative
       dY = get_fe_basis(Y)
       residualuhEF=residual(uhEF,dY)
       r=assemble_vector(residualuhEF,Y)
       assem = SparseMatrixAssembler(sparse_matrix_type,Vector{Float64},X,Y)
       op=FEOperator(residual,jacobian,X,Y,assem)
       nls=NLSolver(linear_solver;
                    show_trace=true,
                    method=:newton,
                    ftol=nlrtol*norm(r,Inf),
                    xtol=1.0e-02)
       solver=FESolver(nls)
       solve!(uhqF,solver,op)

       # Update current solution
       unv .= get_free_dof_values(u)
       hnv .= get_free_dof_values(h)

       if (write_diagnostics && write_diagnostics_freq>0 && mod(step, write_diagnostics_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, n)
        dump_diagnostics_thermal_shallow_water!(h_tmp, w_tmp,
                                              model, dΩ, dω, S, L2MM, H1MM,
                                              hn, un, En, wn, ϕ, F, eF, step, dt,
                                              diagnostics_file,
                                              dump_diagnostics_on_screen)
      end
      if (write_solution && write_solution_freq>0 && mod(step, write_solution_freq) == 0)
        if (!write_diagnostics || write_diagnostics_freq != write_solution_freq)
          compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, n)
        end
        pvd[dt*Float64(step)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(step)"),hn,un,wn)
      end
    end
    hn, un
  end
  if (write_diagnostics || write_solution)
    rm(output_dir,force=true,recursive=true)
    mkdir(output_dir)
  end
  if (write_solution)
    pvdfile=joinpath(output_dir,
       "nswe_ncells_$(num_cells(model))_order_$(order)_theta_method_full_newton")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end
