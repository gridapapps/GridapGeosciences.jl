# module DarcyCubedSphereTestsMPI
  using PartitionedArrays
  using Test
  using FillArrays
  using Gridap
  using GridapPETSc
  using GridapGeosciences

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
  function uθϕ_ex(θϕ)
    u_ex(θϕ2xyz(θϕ))
  end
  function pθϕ_ex(θϕ)
    p_ex(θϕ2xyz(θϕ))
  end
  function Gridap.Fields.gradient(::typeof(p_ex))
     gradient_unit_sphere(pθϕ_ex)∘xyz2θϕ
  end
  function Gridap.Fields.divergence(::typeof(u_ex))
    divergence_unit_sphere(uθϕ_ex)∘xyz2θϕ
  end

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

    a11(u,v)=∫(v⋅u)dΩ
    B11=assemble_matrix(a11,U,V)

    a12(p,v)=∫(∇⋅v*p)dΩ
    B12=assemble_matrix(a12,P,V)

    a21(u,q)=∫(∇⋅u*q)dΩ
    B21=assemble_matrix(a12,P,V)
 
    a22(p,q)=∫(p*q)dΩ
    B22=assemble_matrix(a22,P,Q)  

    B11, B12, B21, B22

    # a((u, p), (v, q)) = ∫(v ⋅ u - (∇ ⋅ v)p + (∇ ⋅ u)q + p*q)dΩ
    # l((v, q)) = ∫(q*f + q*p_ex)dΩ
    # AffineFEOperator(a, l, X, Y)
  end

  function solve_darcy_problem(op)
    solve(op)
  end 

  function compute_darcy_errors(model, order, xh, op)
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
    # U,P=op.trial 
    # uh=interpolate(u_ex,U)
    # ph=interpolate(p_ex,P)
    # eph = ph-p_ex
    # euh = uh-u_ex
    # err_p = sum(∫(eph*eph)dΩ)
    # err_u_l2  = sum(∫(euh⋅euh)dΩ)
    # err_u_div = sum(∫(euh⋅euh + (∇⋅(euh))*(∇⋅(euh)))dΩ)
    # err_p, err_u_l2, err_u_div
  end 
  
  function main(distribute,parts)
    ranks = distribute(LinearIndices((prod(parts),)))
    GridapPETSc.with(args=split(petsc_mumps_options())) do
      order=0
      num_uniform_refinements=0
      model=CubedSphereDiscreteModel(
             ranks,
             num_uniform_refinements;
             radius=1.0,
             adaptive=false,
             order=-1)
      B211,B212,B221,B222=assemble_darcy_problem(model, order);

      for geom_order in 1:8
        model = CubedSphereDiscreteModel(ranks, 
                                         num_uniform_refinements;
                                         radius=1.0, 
                                         order=geom_order,
                                         adaptive=true)
        B111,B112,B121,B122=assemble_darcy_problem(model, order);
        println(norm(B111.matrix_partition.item_ref[]-B211.matrix_partition.item_ref[])/norm(B211.matrix_partition.item_ref[]), " ", 
                norm(B112.matrix_partition.item_ref[]-B212.matrix_partition.item_ref[])/norm(B212.matrix_partition.item_ref[]), " ", 
                norm(B121.matrix_partition.item_ref[]-B221.matrix_partition.item_ref[])/norm(B221.matrix_partition.item_ref[]), " ", 
                norm(B122.matrix_partition.item_ref[]-B222.matrix_partition.item_ref[])/norm(B222.matrix_partition.item_ref[]))
        # println(B1)
      end

      # for num_uniform_refinements=0:0
      #   order=0
      #   model=CubedSphereDiscreteModel(
      #       ranks,
      #       num_uniform_refinements;
      #       radius=1.0,
      #       adaptive=true,
      #       order=2)
      #   op=assemble_darcy_problem(model, order) 
      #   xh=solve_darcy_problem(op)
      #   println(Array(op.op.matrix.matrix_partition.item_ref[]))
      #   println(compute_darcy_errors(model, order, xh, op))
      #   # println(norm(op.op.matrix.item_ref[]))
      #   # for num_uniform_refinements=1:6
      #   #   order=0
      #   #   op=assemble_darcy_problem(model, order)  
      #   #   xh=solve_darcy_problem(op)
      #   #   println(compute_darcy_errors(model, order, xh))
      #   # end 
      # end 
    end
  end
  with_mpi() do distribute 
    main(distribute,1)
  end; 
  
end #module
