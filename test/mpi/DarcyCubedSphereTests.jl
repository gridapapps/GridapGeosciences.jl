module DarcyCubedSphereTestsMPI
  using PartitionedArrays
  using Test
  using FillArrays
  using Gridap
  using GridapPETSc
  using GridapGeosciences

  include("../DarcyCubedSphereTests.jl")
  include("../ConvergenceAnalysisTools.jl")

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
  function main(distribute,parts)
    ranks = distribute(LinearIndices((prod(parts),)))
    GridapPETSc.with(args=split(petsc_gamg_options())) do
       num_refs=[2,3,4,5]
       hs=[2.0/2^n for n in num_refs]
       model_args_series=zip(Fill(ranks,length(num_refs)),num_refs)
       t = PartitionedArrays.PTimer(ranks,verbose=true)
       PartitionedArrays.tic!(t)
       hs1,k1errors,s1=convergence_study(solve_darcy,hs,model_args_series,1,4,PETScLinearSolver())
       PartitionedArrays.toc!(t,"DarcyCubedSphereConvergenceStudy")
       display(t)
       println(hs1)
       println(k1errors)
       println(s1)
    end
  end
  with_mpi() do distribute 
    main(distribute,1)
  end 
end #module
