module GridapGeosciences
  using Gridap
  using GridapHybrid
  using FillArrays
  using SparseArrays
  using LinearAlgebra
  using CSV
  using DataFrames
  using WriteVTK
  using Printf
  include("GeoConstantsParameters.jl")
  include("CubedSphereTriangulations.jl")
  include("CubedSphereDiscreteModels.jl")
  include("Operators.jl")
  include("CoordinateTransformations.jl")
  include("DiagnosticTools.jl")
  include("ShallowWaterExplicit.jl")
  include("ShallowWaterRosenbrock.jl")
  include("ShallowWaterIMEX.jl")
  include("ThermalShallowWaterExplicit.jl")
  include("ThermalShallowWaterExplicit_MatAdv.jl")
  include("ShallowWaterThetaMethodFullNewton.jl")
  include("Helpers.jl")
  include("AdvectionHDG.jl")
  include("WaveEqnHDG.jl")
  include("ShallowWaterHDG.jl")
  include("QuasiGeostrophicHDG.jl")
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
  export project_shallow_water_initial_conditions
  export dump_diagnostics_shallow_water!
  export shallow_water_explicit_time_stepper
  export shallow_water_rosenbrock_time_stepper
  export shallow_water_imex_time_stepper
  export thermal_shallow_water_explicit_time_stepper
  export thermal_shallow_water_mat_adv_explicit_time_stepper
  export shallow_water_theta_method_full_newton_time_stepper
  export write_to_csv, get_scalar_field_from_csv, append_to_csv, initialize_csv
  export clone_fe_function, setup_mixed_spaces, setup_and_factorize_mass_matrices, new_vtk_step
  export advection_hdg, project_initial_conditions_hdg, wave_eqn_hdg, shallow_water_hdg, quasi_geostrophic_hdg

  # MPI-parallel part
  using PartitionedArrays
  using GridapDistributed
  using GridapP4est
  using GridapPETSc
  using P4est_wrapper
  include("mpi/CubedSphereDistributedDiscreteModels.jl")
end # module
