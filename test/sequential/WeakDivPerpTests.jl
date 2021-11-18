module WeakDivPerpTestsSeq
   using Gridap
   using GridapGeosciences
   using Plots
   using Test
   using FillArrays

   include("../ConvergenceAnalysisTools.jl")
   include("../WeakDivPerpTests.jl")

   n_values=generate_n_values(2)
   hs=[2.0/n for n in n_values]
   model_args_series=[(n,) for n in n_values]
   @time ahs0,ak0errors,as0=convergence_study(compute_error_weak_div_perp,hs,model_args_series,0,4)
   @test round(as0,digits=3) ≈ 2.084


   n_values=generate_n_values(2)
   hs=[2.0/n for n in n_values]
   model_args_series=zip(n_values,Fill(1,length(n_values)))
   @time bihs0,bik0errors,bis0=convergence_study(compute_error_weak_div_perp,hs,model_args_series,0,4)
   @test round(bis0,digits=3) ≈ 0.947

   n_values=generate_n_values(2)
   hs=[2.0/n for n in n_values]
   model_args_series=zip(n_values,Fill(2,length(n_values)))
   @time biqhs0,biqk0errors,biqs0=convergence_study(compute_error_weak_div_perp,hs,model_args_series,0,4)
   @test round(biqs0,digits=3) ≈ 2.083


  #  plotd=plot([ahs0,bihs0,biqhs0],[ak0errors,bik0errors,biqk0errors],
  #  xaxis=:log, yaxis=:log,
  #  label=["k=0 analytical map" "k=0 bilinear map" "k=0 biquadratic map"],
  #  shape=:auto,
  #  xlabel="h",ylabel="L2 error norm", legend=:bottomright)

  #  savefig(plotd,"L2_error_weak_div_perp.png")

end
