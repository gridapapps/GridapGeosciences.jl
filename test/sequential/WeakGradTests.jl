module WeakGradTestsSeq
   using Gridap
   using GridapGeosciences
   import Gridap.Fields: ∇, divergence
   using Plots
   using Test

   include("../ConvergenceAnalysisTools.jl")
   include("../WeakGradTests.jl")

   n_values=generate_n_values(2)
   hs=[2.0/n for n in n_values]
   model_args_series=[(n,) for n in n_values]

   @time ahs0,ak0errors,as0=convergence_study(compute_error_weak_grad,hs,model_args_series,0,4)
   @test as0 ≈ 0.8346320885900106

   n_values=generate_n_values(2,n_max=50)
   hs=[2.0/n for n in n_values]
   model_args_series=[(n,) for n in n_values]

   @time ahs1,ak1errors,as1=convergence_study(compute_error_weak_grad,hs,model_args_series,1,8)
   @test as1 ≈ 1.1034326200306834

end
