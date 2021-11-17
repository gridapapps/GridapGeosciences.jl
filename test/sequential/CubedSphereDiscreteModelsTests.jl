module CubedSphereDiscreteModelsTestsSeq
  using Gridap
  using GridapGeosciences
  using FillArrays
  using Test

  include("../CubedSphereDiscreteModelsTests.jl")

  n_values=CubedSphereDiscreteModelsTests.generate_n_values(2)
  hs=[2.0/n for n in n_values]

  model_args_series=zip(n_values,Fill(1,length(n_values)))
  _,_,s=CubedSphereDiscreteModelsTests.convergence_discrete_cubed_sphere_surface(hs,model_args_series,0)
  @test round(s,digits=1) ≈ 2.0

  model_args_series=zip(n_values,Fill(2,length(n_values)))
  _,_,s=CubedSphereDiscreteModelsTests.convergence_discrete_cubed_sphere_surface(hs,model_args_series,4)
  @test round(s,digits=1) ≈ 4.0

  model_args_series=[(n,) for n in n_values]
  _,_,s=CubedSphereDiscreteModelsTests.convergence_discrete_cubed_sphere_surface(hs,model_args_series,4)
  @test round(s,digits=1) ≈ 5.7

end
