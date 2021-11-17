module CubedSphereDiscreteModelsTestsMPI
  using PartitionedArrays
  using Test
  using FillArrays
  include("../CubedSphereDiscreteModelsTests.jl")
  function main(parts)
    num_refs=[2,3,4,5]
    hs=[2.0/2^n for n in num_refs]
    model_args_series=zip(Fill(parts,length(num_refs)),num_refs)
    a,b,s=CubedSphereDiscreteModelsTests.convergence_discrete_cubed_sphere_surface(hs,model_args_series,4)
    @test round(s,digits=1) ≈ 5.6
  end
  prun(main,mpi,4)
end #module
