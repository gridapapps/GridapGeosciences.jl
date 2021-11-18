module WeakDivPerpTestsMPI
   using PartitionedArrays
   using Test
   using FillArrays
   using Gridap
   using GridapGeosciences

   include("../WeakDivPerpTests.jl")
   include("../ConvergenceAnalysisTools.jl")

   function main(parts)
     num_refs=[2,3,4,5]
     hs=[2.0/2^n for n in num_refs]
     model_args_series=zip(Fill(parts,length(num_refs)),num_refs)
     a,b,s=convergence_study(compute_error_weak_div_perp,hs,model_args_series,4)
     println(a)
     println(b)
     println(s)
     # @test round(s,digits=1) ≈ 5.6
   end
   prun(main,mpi,4)


  #  plotd=plot([ahs0,bihs0,biqhs0],[ak0errors,bik0errors,biqk0errors],
  #  xaxis=:log, yaxis=:log,
  #  label=["k=0 analytical map" "k=0 bilinear map" "k=0 biquadratic map"],
  #  shape=:auto,
  #  xlabel="h",ylabel="L2 error norm", legend=:bottomright)

  #  savefig(plotd,"L2_error_weak_div_perp.png")

end
