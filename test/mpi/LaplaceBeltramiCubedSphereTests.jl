module LaplaceBeltramiCubedSphereTestsMPI
  using PartitionedArrays
  using Test
  using FillArrays
  using Gridap
  using GridapPETSc
  using GridapGeosciences

  include("../ConvergenceAnalysisTools.jl")
  include("../LaplaceBeltramiCubedSphereTests.jl")

  function petsc_options()
    """
    -ksp_type cg -ksp_rtol 1.0e-06 -ksp_atol 0.0
    -ksp_monitor -pc_type asm -sub_ksp_type preonly
    -sub_pc_type lu
    """
  end

  function main(distribute,parts)
    ranks = distribute(LinearIndices((prod(parts),)))
    GridapPETSc.with(args=split(petsc_options())) do
       num_refs=[2,3,4,5]
       hs=[2.0/2^n for n in num_refs]
       model_args_series=zip(Fill(ranks,length(num_refs)),num_refs)
       t = PartitionedArrays.PTimer(ranks,verbose=true)
       PartitionedArrays.tic!(t)
       hs1,k1errors,s1=convergence_study(solve_laplace_beltrami,
                                         hs,
                                         model_args_series,1,8,PETScLinearSolver())
       PartitionedArrays.toc!(t,"LaplaceBeltramiCubedSphereConvergenceStudy")
       display(t)
       @test round(s1,digits=1) ≈ 1.0

       # Do garbage collection of all PETSc objects
       # set up during convergence_study
       GridapPETSc.gridap_petsc_gc()

       PartitionedArrays.tic!(t)
       hs2,k2errors,s2=convergence_study(solve_laplace_beltrami,
                                         hs,
                                         model_args_series,2,12,PETScLinearSolver())
       PartitionedArrays.toc!(t,"LaplaceBeltramiCubedSphereConvergenceStudy")
       display(t)
       @test round(s2,digits=1) ≈ 2.0
    end
  end
  with_mpi() do distribute 
    main(distribute,4)
  end 
end
