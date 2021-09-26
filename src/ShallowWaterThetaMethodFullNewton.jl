# Topography
function topography(xyz)
  0.0
end

# Compute initial volume flux (williamsom2)
function F₀(u₀,h₀,U,V,dΩ)
  a(u,v) = ∫(v⋅u)dΩ
  b(v)   = ∫((v⋅(h₀*u₀)))dΩ
  solve(AffineFEOperator(a,b,U,V))
end

# Compute initial potential vorticity (williamsom2)
function q₀(u₀,h₀,f,R,S,n,dΩ)
  a(r,s) = ∫( s*(r*h₀) )dΩ
  b(s)   = ∫( s*f - ⟂(∇(s),n)⋅u₀ )dΩ
  solve(AffineFEOperator(a,b,R,S))
end

# Generate initial monolothic solution (williamsom2)
function uhqF₀(u₀,h₀,q₀,F₀,X,Y,dΩ)
  a((u,p,r,u2),(v,q,s,v2))=∫(v⋅u+q*p+s*r+v2⋅u2)dΩ
  b((v,q,s,v2))=∫( v⋅u₀+ q*h₀ + s*q₀ + v2⋅F₀ )dΩ
  solve(AffineFEOperator(a,b,X,Y))
end

"""
  Solves the nonlinear rotating shallow water equations
  T : [0,T] simulation interval
  N : number of time subintervals
  θ : Theta-method parameter [0,1)
"""
# **IMPORTANT NOTE**: At present, this function ONLY implements
# Backward Euler (i.e.  θ=0) no matter the value of the θ parameter value.
# Below there is an approach (commented out) for a general implementation
# of residual and jacobian for arbitrary values of θ. However, I
# was not able to get it working (Newton-Raphson diverges). There should
# be some BUG, in the definition of the Jacobian, I guess.
function shallow_water_theta_method_full_newton_time_stepper(
      model, order, degree, h₀, u₀, f₀, g, θ, T, N;
      nlrtol=1.0e-08, # Newton solver relative residual tolerance
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
    H1MM, _, L2MM, H1MMchol, _, L2MMchol = setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)
  end


  Y = MultiFieldFESpace([V,Q,S,V])
  X = MultiFieldFESpace([U,P,R,U])

  a1(u,v)=∫(v⋅u)dΩ
  l1(v)=∫(v⋅u₀)dΩ
  un=solve(AffineFEOperator(a1,l1,U,V))

  a2(u,v)=∫(v*u)dΩ
  l2(v)=∫(v*h₀)dΩ
  hn=solve(AffineFEOperator(a2,l2,P,Q))

  a3(u,v)=∫(v*u)dΩ
  l3(v)=∫(v*f₀)dΩ
  fn=solve(AffineFEOperator(a3,l3,R,S))

  unv,hnv,fnv=get_free_dof_values(un,hn,fn)

  b = interpolate_everywhere(topography,P)

  # Compute:
  #     - Initial potential vorticity (q₀)
  #     - Initial volume flux (F₀)
  #     - Initial full solution
  ΔuΔhqF=uhqF₀(un,hn,q₀(un,hn,fn,R,S,n,dΩ),F₀(un,hn,U,V,dΩ),X,Y,dΩ)
  Δu,Δh,q,F = ΔuΔhqF

  h_tmp = copy(hnv)
  w_tmp = copy(fnv)

  function run_simulation(pvd=nothing)
    diagnostics_file = joinpath(output_dir,"nswe_diagnostics_theta_method_full_newton.csv")

    dt  = T/N
    τ   = dt/2 # APVM stabilization parameter
    hc  = CellField(h₀,Ω)
    uc  = CellField(u₀,Ω)

    if (write_diagnostics)
      ϕ = clone_fe_function(Q,hn)
      wn = clone_fe_function(S,fn)
      initialize_csv(diagnostics_file, "time", "mass", "vorticity", "kinetic", "potential", "power")
    end

    if (write_diagnostics && write_diagnostics_freq==1)
      compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, n)
      compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,un⋅un,hn,g)
      dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                      model, dΩ, dω, S, L2MM, H1MM,
                                      hn, un, wn, ϕ, F, g, 1, dt,
                                      diagnostics_file,
                                      dump_diagnostics_on_screen)
    end

    for step=1:N
       # Williamsom2
       e = hn-h₀;err_h = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(hc⋅hc)*dΩ))
       e = un-u₀;err_u = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(uc⋅uc)*dΩ))

       println("step=", step, ",\terr_u: ", err_u, ",\terr_h: ", err_h,
               " ", norm(get_free_dof_values(Δu)), " ", norm(get_free_dof_values(Δh)))

       function residual((u,h,qvort,F),(v,q,s,v2))
         uiΔu  = u
         hiΔh  = h
         hbiΔh = h
         ∫((1.0/dt)*v⋅(u-un))dΩ-∫((DIV(v))*(g*hbiΔh + 0.5*uiΔu⋅uiΔu))dω+
         ∫((qvort-τ*(uiΔu⋅∇(qvort)))*(v⋅⟂(F,n)) +   # eq1
          (1.0/dt)*q*(h-hn))dΩ + ∫(q*(DIV(F)))dω +  # eq2
         ∫(s*qvort*hiΔh + ⟂(∇(s),n)⋅uiΔu - s*fn +   # eq3
           v2⋅(F-hiΔh*uiΔu))dΩ                      # eq4
       end

       function jacobian((u,h,qvort,F),(du,dh,dq,dF),(v,q,s,v2))
          uiΔu  = u
          uidu  = du
          hiΔh  = h
          hidh  = dh
          hbidh = dh
          ∫((1.0/dt)*v⋅du +  (dq    - τ*(uiΔu⋅∇(dq)+uidu⋅∇(qvort)))*(v⋅⟂(F ,n))
                          +  (qvort - τ*(           uiΔu⋅∇(qvort)))*(v⋅⟂(dF,n)))dΩ -
                          ∫((DIV(v))*(g*hbidh +uiΔu⋅uidu))dω  + # eq1
          ∫((1.0/dt)*q*dh)dΩ + ∫(q*(DIV(dF)))dω           +  # eq2
          ∫(s*(qvort*hidh+dq*hiΔh) + ⟂(∇(s),n)⋅uidu       +  # eq3
                    v2⋅(dF-hiΔh*uidu-hidh*uiΔu))dΩ           # eq4
        end

      #  # Not working (Newton diverges)
      #  function residual((Δu,Δh,qvort,F),(v,q,s,v2))
      #    one_m_θ = (1-θ)
      #    uiΔu    = un     + one_m_θ*Δu
      #    hiΔh    = hn     + one_m_θ*Δh
      #    hbiΔh   = hn + b + one_m_θ*Δh
      #    ∫((1.0/dt)*v⋅(Δu)-(∇⋅(v))*(g*hbiΔh + 0.5*uiΔu⋅uiΔu)+
      #        (qvort-τ*(uiΔu⋅∇(qvort)))*(v⋅⟂(F,n)) +   # eq1
      #      (1.0/dt)*q*(Δh))dΩ + ∫(q*(DIV(F)))dω +  # eq2
      #    ∫(s*qvort*hiΔh + ⟂(∇(s),n)⋅uiΔu - s*fn +   # eq3
      #        v2⋅(F-hiΔh*uiΔu))dΩ                      # eq4
      #  end
      #  # Not working (Newton diverges)
      #  function jacobian((Δu,Δh,qvort,F),(du,dh,dq,dF),(v,q,s,v2))
      #    one_m_θ = (1-θ)
      #    uiΔu  = un     + one_m_θ*Δu
      #    uidu  = un     + one_m_θ*du
      #    hiΔh  = hn     + one_m_θ*Δh
      #    hidh  = hn     + one_m_θ*dh
      #    hbidh = hn + b + one_m_θ*dh
      #    ∫((1.0/dt)*v⋅du +  (dq    - τ*(uiΔu⋅∇(dq)+uidu⋅∇(qvort)))*(v⋅⟂(F ,n))
      #                    +  (qvort - τ*(           uiΔu⋅∇(qvort)))*(v⋅⟂(dF,n))
      #                    -  (∇⋅(v))*(g*hbidh +uiΔu⋅uidu)   +  # eq1
      #      (1.0/dt)*q*dh)dΩ + ∫(q*(DIV(dF)))dω             +  # eq2
      #      ∫(s*(qvort*hidh+dq*hiΔh) + ⟂(∇(s),n)⋅uidu       +  # eq3
      #        v2⋅(dF-hiΔh*uidu-hidh*uiΔu))dΩ                   # eq4
      #  end

       # Solve fully-coupled monolithic nonlinear problem
       # Use previous time-step solution, ΔuΔhqF, as initial guess
       # Overwrite solution into ΔuΔhqF

       # Adjust absolute tolerance ftol s.t. it actually becomes relative
       dY = get_fe_basis(Y)
       dX = get_trial_fe_basis(X)
       residualΔuΔhqF=residual(ΔuΔhqF,dY)
       r=assemble_vector(residualΔuΔhqF,Y)
       op=FEOperator(residual,jacobian,X,Y)
       nls=NLSolver(show_trace=true, method=:newton, ftol=nlrtol*norm(r,Inf), xtol=1.0e-02)
       solver=FESolver(nls)

       solve!(ΔuΔhqF,solver,op)

       # Update current solution
       unv .= get_free_dof_values(Δu)
       hnv .= get_free_dof_values(Δh)
       #unv .= unv .+ get_free_dof_values(Δu)
       #hnv .= unv .+ get_free_dof_values(Δh)

       if (write_diagnostics && write_diagnostics_freq>0 && mod(step, write_diagnostics_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, n)
        compute_bernoulli_potential!(ϕ,dΩ,Q,L2MMchol,un⋅un,hn,g)
        dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                        model, dΩ, dω, S, L2MM, H1MM,
                                        hn, un, wn, ϕ, F, g, step, dt,
                                        diagnostics_file,
                                        dump_diagnostics_on_screen)
      end
      if (write_solution && write_solution_freq>0 && mod(step, write_solution_freq) == 0)
        compute_diagnostic_vorticity!(wn, dΩ, S, H1MMchol, un, n)
        pvd[Float64(step)] = new_vtk_step(Ω,joinpath(output_dir,"n=$(step)"),hn,un,wn)
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
