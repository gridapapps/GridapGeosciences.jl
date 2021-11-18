module LaplaceBeltramiCubedSphereTestsMPI
  using PartitionedArrays
  using Test
  using FillArrays
  using Gridap
  using GridapPETSc
  using GridapGeosciences

  include("../ConvergenceAnalysisTools.jl")
  include("../LaplaceBeltramiCubedSphereTests.jl")

  function petsc_gamg_options()
    """
      -ksp_type cg -ksp_rtol 1.0e-06 -ksp_atol 0.0
      -ksp_monitor -pc_type gamg -pc_gamg_type agg
      -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cg
      -mg_coarse_sub_pc_factor_mat_ordering_type nd -pc_gamg_process_eq_limit 50
      -pc_gamg_square_graph 9 pc_gamg_agg_nsmooths 1
    """
  end
  function petsc_mumps_options()
    """
      -ksp_type preonly -ksp_error_if_not_converged true
      -pc_type cholesky -pc_factor_mat_solver_type mumps
    """
  end
  function main(parts)
    GridapPETSc.with(args=split(petsc_gamg_options())) do
       num_refs=[2,3,4,5]
       hs=[2.0/2^n for n in num_refs]
       model_args_series=zip(Fill(parts,length(num_refs)),num_refs)
       t = PartitionedArrays.PTimer(parts,verbose=true)
       PartitionedArrays.tic!(t)
       hs1,k1errors,s1=convergence_study(solve_laplace_beltrami,
                                         hs,
                                         model_args_series,1,8,PETScLinearSolver())
       PartitionedArrays.toc!(t,"LaplaceBeltramiCubedSphereConvergenceStudy")
       display(t)
       println(hs1)
       println(k1errors)
       println(s1)
    end
  end
  prun(main,mpi,4)


end
