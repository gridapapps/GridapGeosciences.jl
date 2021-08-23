module GridapGeosciences
  using Gridap
  using FillArrays
  include("CubedSphereTriangulations.jl")
  include("CubedSphereDiscreteModels.jl")
  include("Operators.jl")
  include("CoordinateTransformations.jl")
  export CubedSphereDiscreteModel
  export perp,⟂
  export divergence_unit_sphere
  export laplacian_unit_sphere
  export normal_unit_sphere
  export gradient_unit_sphere
  export xyz2θϕr
  export xyz2θϕ
  export θϕ2xyz
  export spherical_to_cartesian_matrix
  export cartesian_to_spherical_matrix
end # module
