module  Helpers

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces, Gridap.Helpers, Gridap.Arrays
import Gridap.TensorValues: MultiValue
using LinearAlgebra
using FillArrays

using GridapGeosciences.Fields

include("Overloads.jl")
include("Operators.jl")
include("CoordinateMappings.jl")

export forward_jacobian, covariant_basis, forward_pinv_jacobian

export pinvJ
export sqrtg,  detg
export metric, inv_metric
export surflap, surfdiv, sgrad, skew_surfdiv, skew_surfgrad

export xyz2θϕr
export J

end
