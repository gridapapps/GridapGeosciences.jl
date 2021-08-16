module GridapGeosciences

  using Gridap
  using FillArrays

  include("CubedSphereTriangulations.jl")
  include("CubedSphereDiscreteModels.jl")
  include("SmarterCubedSphereDiscreteModels.jl")


  export CubedSphereTriangulation
  export CubedSphereDiscreteModel
  export SmarterCubedSphereDiscreteModel

end # module
