module Fields

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
using LinearAlgebra
using FillArrays

import Gridap.TensorValues: symmetric_part, SymTensorValue, ThirdOrderTensorValue, contracted_product

include("Cartesian2SphericalMap.jl")
export Cartesian2SphericalMap

include("CylinderMap.jl")
include("CylinderMetric.jl")
include("CylinderInvMetric.jl")
export CylinderMap, CylinderMetric, CylinderInvMetric

include("MobiusMap.jl")
include("MobiusMetric.jl")
include("MobiusInvMetric.jl")
export MobiusMap, MobiusMetric, MobiusInvMetric

include("CubedSphereMap.jl")
include("CubedSphereWithThicknessMap.jl")
include("CubedSphereInvMap.jl")
include("CubedSphereWithThicknessInvMap.jl")
export CubedSphereMap, CubedSphereWithThicknessMap
export CubedSphereInvMap, CubedSphereWithThicknessInvMap

include("CubedSphereMetric.jl")
include("CubedSphereInvMetric.jl")
include("CubedSphereWithThicknessMetric.jl")
include("CubedSphereWithThicknessInvMetric.jl")
export CubedSphereMetric, CubedSphereInvMetric
export CubedSphereWithThicknessMetric, CubedSphereWithThicknessInvMetric

inverse_metric_field(f::Field) = Operation(inv)(f)
export inverse_metric_field

end
