module LaplaceBeltramiCubedSphereTestsSeq

  using Test
  using Gridap
  using GridapGeosciences
  using Plots

  include("../ConvergenceAnalysisTools.jl")
  include("../LaplaceBeltramiCubedSphereTests.jl")

  n_values=generate_n_values(2)
  hs=[2.0/n for n in n_values]
  model_args_series=[(n,) for n in n_values]

  @time ahs1,ak1errors,as1=convergence_study(solve_laplace_beltrami,hs,model_args_series,1,8)
  @test round(as1,digits=1) ≈ 1.0

  @time ahs2,ak2errors,as2=convergence_study(solve_laplace_beltrami,hs,model_args_series,2,12)
  @test round(as2,digits=1) ≈ 2.0


end
