module WeakDivPerpTestsMPI
   using PartitionedArrays
   using Test
   using FillArrays
   using Gridap
   using GridapGeosciences
   using GridapPETSc

   include("../WeakDivPerpTests.jl")
   include("../ConvergenceAnalysisTools.jl")

   function petsc_gamg_options()
    """
      -ksp_type cg -ksp_rtol 1.0e-06 -ksp_atol 0.0
      -ksp_monitor -pc_type gamg -pc_gamg_type agg
      -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky
      -mg_coarse_sub_pc_factor_mat_ordering_type nd -pc_gamg_process_eq_limit 50
      -pc_gamg_square_graph 9 pc_gamg_agg_nsmooths 1
    """
   end
   function main(parts)
     GridapPETSc.with(args=split(petsc_gamg_options())) do
       num_refs=[1,2,3,4,5]
       hs=[2.0/2^n for n in num_refs]
       model_args_series=zip(Fill(parts,length(num_refs)),num_refs)
       t = PartitionedArrays.PTimer(parts,verbose=true)
       PartitionedArrays.tic!(t)
       a,b,s=convergence_study(compute_error_weak_div_perp,
                             hs,model_args_series,0,4,PETScLinearSolver())
       PartitionedArrays.toc!(t,"WeakDivPerpTest")
       display(t)
       @test round(s,digits=2) ≈ 2.12
     end
   end
   prun(main,mpi,4)
end
