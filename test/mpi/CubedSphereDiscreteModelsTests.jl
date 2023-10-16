module CubedSphereDiscreteModelsTestsMPI
  using PartitionedArrays
  using Test
  using FillArrays
  using Gridap
  using GridapGeosciences

  include("../CubedSphereDiscreteModelsTests.jl")
  include("../ConvergenceAnalysisTools.jl")
  function main(distribute,parts)
    ranks = distribute(LinearIndices((prod(parts),)))
    num_refs=[2,3,4,5]
    hs=[2.0/2^n for n in num_refs]
    model_args_series=zip(Fill(ranks,length(num_refs)),num_refs)
    a,b,s=convergence_discrete_cubed_sphere_surface(hs,model_args_series,4)
    @test round(s,digits=1) ≈ 5.6
  end
  with_mpi() do distribute 
    main(distribute,4)
  end 
end #module
