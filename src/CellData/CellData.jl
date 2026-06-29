module CellData

using Gridap
using Gridap.Geometry, Gridap.Adaptivity, Gridap.TensorValues, Gridap.Fields
using Gridap.Helpers

using GridapGeosciences.Geometry
using GridapGeosciences.Fields

import GridapGeosciences.Geometry: BFTATDM, BFTATDMIM, BFTATDMEM
import GridapGeosciences.Fields: CubedSphereMap
import Gridap.TensorValues: SymTensorValue

include("CellFields.jl")
include("SurfaceDiffOps.jl")

export MetricCellField
export InvMetricCellField
export MeasureCellField
export AmbientMapCellField
export LatLonMapCellField

export Δs
export vecΔs
export curls
export ∇s
export divs
export skew_∇s
export skew_divs
export dagger

end
