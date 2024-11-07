module RefinedPatchDarcyCubedSphereTestsMPI
  using PartitionedArrays
  using Test
  using FillArrays
  using Gridap
  using GridapPETSc
  using GridapGeosciences
  using GridapDistributed
  using GridapP4est

  function petsc_gamg_options()
    """
      -ksp_type gmres -ksp_rtol 1.0e-06 -ksp_atol 0.0
      -ksp_monitor -pc_type gamg -pc_gamg_type agg
      -mg_levels_esteig_ksp_type gmres -mg_coarse_sub_pc_type lu
      -mg_coarse_sub_pc_factor_mat_ordering_type nd -pc_gamg_process_eq_limit 50
      -pc_gamg_square_graph 9 pc_gamg_agg_nsmooths 1
    """
  end
  function petsc_mumps_options()
    """
      -ksp_type preonly -ksp_error_if_not_converged true
      -pc_type lu -pc_factor_mat_solver_type mumps
    """
  end

  p_ex(x) = -x[1]x[2]x[3]
  u_ex(x) = VectorValue(
    x[2]x[3] - 3x[1]x[1]x[2]x[3] / (x[1]x[1] + x[2]x[2] + x[3]x[3]),
    x[1]x[3] - 3x[1]x[2]x[2]x[3] / (x[1]x[1] + x[2]x[2] + x[3]x[3]),
    x[1]x[2] - 3x[1]x[2]x[3]x[3] / (x[1]x[1] + x[2]x[2] + x[3]x[3])
    )
  f_ex(x) = -12x[1]x[2]x[3]

  # function uθϕ_ex(θϕ)
  #   u_ex(θϕ2xyz(θϕ))
  # end
  # function pθϕ_ex(θϕ)
  #   p_ex(θϕ2xyz(θϕ))
  # end
  # function Gridap.Fields.gradient(::typeof(p_ex))
  #    gradient_unit_sphere(pθϕ_ex)∘xyz2θϕ
  # end
  # function Gridap.Fields.divergence(::typeof(u_ex))
  #   divergence_unit_sphere(uθϕ_ex)∘xyz2θϕ
  # end

  # using Distributions
  # θϕ=Point(rand(Uniform(0,2*pi)),rand(Uniform(-pi/2,pi/2)))
  # θϕ2xyz(θϕ)

  function assemble_darcy_problem(model, order)
    rt_reffe = ReferenceFE(raviart_thomas, Float64, order)
    lg_reffe = ReferenceFE(lagrangian, Float64, order)
    V = FESpace(model, rt_reffe, conformity=:Hdiv)
    U = TrialFESpace(V)
    Q = FESpace(model, lg_reffe; conformity=:L2)
    P = TrialFESpace(Q)
    X = MultiFieldFESpace([U, P])
    Y = MultiFieldFESpace([V, Q])
    Ω = Triangulation(model)
    degree = 10 # 2*order
    dΩ = Measure(Ω, degree)
    a((u, p), (v, q)) = ∫(v ⋅ u - (∇ ⋅ v)p + (∇ ⋅ u)q + p*q)dΩ
    l((v, q)) = ∫(q*f_ex + q*p_ex)dΩ
    AffineFEOperator(a, l, X, Y)
  end

  function solve_darcy_problem(op)
    solve(op)
  end 

  function compute_darcy_errors(model, order, xh)
    uh,ph=xh 
    eph = ph-p_ex
    euh = uh-u_ex
    degree=10 #2*order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    err_p = sum(∫(eph*eph)dΩ)
    err_u_l2  = sum(∫(euh⋅euh)dΩ)
    err_u_div = sum(∫(euh⋅euh + (∇⋅(euh))*(∇⋅(euh)))dΩ)
    err_p, err_u_l2, err_u_div
  end
  
  function adapt_model(ranks,model)
    cell_partition=get_cell_gids(model)
    ref_coarse_flags=map(ranks,partition(cell_partition)) do rank,indices
        flags=zeros(Cint,length(indices))
        flags.=refine_flag        
    end
    GridapP4est.adapt(model,ref_coarse_flags)
  end 

  function main(distribute,parts)
    ranks = distribute(LinearIndices((prod(parts),)))

    # Change directory to the location of the script,
    # where the mesh data files are located 
    cd(@__DIR__)

    coarse_model, cell_panels, coarse_cell_wise_vertex_coordinates=
          parse_cubed_sphere_coarse_model("connectivity-gridapgeo.txt","geometry-gridapgeo.txt")


    #println(coarse_cell_wise_vertex_coordinates_data)
    #println(coarse_cell_wise_vertex_coordinates_ptrs)
    # model,_=adapt_model(ranks,model)
    # writevtk(model,"model_adapted_1")
    # model,_=adapt_model(ranks,model)
    # writevtk(model,"model_adapted_2")
    GridapPETSc.with(args=split(petsc_mumps_options())) do
      order=0
      num_uniform_refinements=0
      model=CubedSphereDiscreteModel(
        ranks,
        coarse_model,
        coarse_cell_wise_vertex_coordinates,
        cell_panels,
        num_uniform_refinements;
        radius=1.0,
        adaptive=true,
        order=1)
      op=assemble_darcy_problem(model, order)      
      xh=solve_darcy_problem(op)
      for num_uniform_refinements=1:3
        model,_=adapt_model(ranks,model)
        op=assemble_darcy_problem(model, order) 
        xh=solve_darcy_problem(op)
        println(compute_darcy_errors(model, order, xh))
      end 
    end
  end
  with_mpi() do distribute 
    main(distribute,1)
  end
end #module
