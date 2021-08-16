module GridapGeosciences
  using Gridap
  using FillArrays
  include("CubedSphereTriangulations.jl")
  include("CubedSphereDiscreteModels.jl")
  export CubedSphereDiscreteModel
end # module
