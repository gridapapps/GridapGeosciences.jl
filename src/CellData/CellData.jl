module CellData

using Gridap
using Gridap.Geometry, Gridap.Adaptivity, Gridap.TensorValues, Gridap.Fields, Gridap.CellData
using Gridap.Arrays
using Gridap.Helpers
using Gridap.ReferenceFEs

using GridapGeosciences.Geometry
using GridapGeosciences.Fields

import GridapGeosciences.Geometry: BFTATDM, BFTATDMIM, BFTATDMEM
import GridapGeosciences.Fields: CubedSphereMap
import Gridap.TensorValues: SymTensorValue
import GridapGeosciences.Helpers: sqrtg, J, metric, pinvJ, perp
import Gridap.Geometry: FaceCompressedVector, push_normal


include("CellFields.jl")
include("SurfaceDiffOps.jl")
include("NormalsAreaForms.jl")

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

export pushforward_normal
export pushforward_reference_normal
export pushforward_parametric_normal
export pullback_area_form

end
