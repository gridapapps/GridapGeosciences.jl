module GridapGeosciences
  using Gridap
  using FillArrays
  include("CubedSphereTriangulations.jl")
  include("CubedSphereDiscreteModels.jl")
  include("Operators.jl")
  include("CoordinateTransformations.jl")
  export CubedSphereDiscreteModel
  export perp
  export xyz2θϕr
  export spherical_to_cartesian_matrix
  export cartesian_to_spherical_matrix
end # module
