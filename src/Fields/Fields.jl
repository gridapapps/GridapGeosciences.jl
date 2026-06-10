module Fields

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
using LinearAlgebra
using FillArrays

include("CubedSphereForwardMap.jl")
include("CubedSphereInverseMap.jl")
include("MatMultField.jl")
include("AffineField.jl")
include("Cartesian2SphericalMap.jl")

export CubedSphereForwardMap,  ForwardMap2DGenerator, ForwardMap3DGenerator
export CubedSphereInverseMap, InverseMap2DGenerator, InverseMap3DGenerator
export _evaluate_forward_jacobian_2d, _evaluate_forward_jacobian_3d
export MatMultField, MyAffineField
export Cartesian2SphericalMap
export J, normal_vec

end
