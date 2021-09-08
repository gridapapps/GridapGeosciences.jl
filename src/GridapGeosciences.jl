module GridapGeosciences
  using Gridap
  using FillArrays
  using LinearAlgebra
  using CSV
  using DataFrames
  include("GeoConstantsParameters.jl")
  include("CubedSphereTriangulations.jl")
  include("CubedSphereDiscreteModels.jl")
  include("Operators.jl")
  include("CoordinateTransformations.jl")
  include("DiagnosticTools.jl")
  include("ShallowWaterExplicit.jl")
  export rₑ, Ωₑ, g, f
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
  export Eₖ, Eₚ, Eₜ
  export compute_kin_to_pot!, compute_pot_to_kin!, compute_total_mass!
  export compute_diagnostics_shallow_water!
  export shallow_water_explicit_time_step!
  export shallow_water_time_stepper
  export write_to_csv, get_scalar_field_from_csv, append_to_csv, initialize_csv
end # module
