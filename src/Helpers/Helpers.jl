module  Helpers

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
import Gridap.TensorValues: MultiValue
using LinearAlgebra

using GridapGeosciences.Fields

include("Overloads.jl")
include("Operators.jl")
include("CoordinateMappings.jl")
include("SphereSurfaceFunctions.jl")

export pinvJ
export sqrtg,  detg
export metric, inv_metric
export surflap, surfdiv, sgrad, skew_surfdiv, skew_surfgrad
export sphere_surface_normal_vec, sphere_tangent_vec_component    
export xyz2θϕr
export J

end
