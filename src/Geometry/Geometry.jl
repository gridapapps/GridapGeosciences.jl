module Geometry
using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
using LinearAlgebra
using FillArrays

## AtlasDiscreteModels new machinery (will eventually replace enterily what we have above)
## Some definitions in the below included julia source files must be moved to other
## GridapGeosciences modules for consistency, e.g., to GridapGeosciences.Fields
using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.ReferenceFEs, Gridap.Helpers
using Gridap.Adaptivity, Gridap.Visualization
import Gridap.TensorValues: symmetric_part, SymTensorValue, ThirdOrderTensorValue, contracted_product
import Gridap.Geometry: Grid
import GridapGeosciences.Helpers: J

include("CoarseMeshes.jl")
include("AtlasGrids.jl")
include("AtlasDiscreteModels.jl")

export AtlasGrid, AtlasDiscreteModel
export IntrinsicAtlasDiscreteModel, ExtrinsicAtlasDiscreteModel
export IntrinsicManifold, ExtrinsicManifold
export MetricCellField
export InvMetricCellField
export MeasureCellField
export AmbientMapCellField, LatLonMapCellField
export Δs
export vecΔs
export curls
export ∇s
export divs
export skew_∇s
export skew_divs

export CylinderMesh, CylinderChartMap, CylinderMetricField, CylinderInvMetricField
export MobiusStripMesh, MobiusChartMap, MobiusMetricField, MobiusInvMetricField
export CubedSphereMesh, CubedSphereMap, CubedSphereInvMap, CubedSphereMetricField, CubedSphereInvMetricField
export CubedSphereWithThicknessMesh, CubedSphereWithThicknessMap, CubedSphereWithThicknessInvMap, CubedSphereWithThicknessMetricField, CubedSphereWithThicknessInvMetricField
export CubedSphereWithThicknessMetricField, CubedSphereWithThicknessInvMetricField
export ExtrudedCubedSphereWithThicknessMesh
export get_atlas_grid
export get_cell_ambient_maps, get_cell_metric, get_cell_inv_metric
export get_coarse_mesh
export JtJ
export generate_refined_models


import Gridap.Geometry: TriangulationView
import Gridap.Geometry: FaceToCellGlue, FaceCompressedVector, push_normal

using GridapGeosciences.Fields

using GridapGeosciences.Helpers
import GridapGeosciences.Helpers: inv_metric, forward_jacobian, forward_pinv_jacobian
import GridapGeosciences.Helpers: perp

include("CubeSurface.jl")
include("PanelMatrices.jl")
include("BoundaryTriangulations.jl")
include("SkeletonTriangulations.jl")
include("AdaptedTriangulations.jl")
include("TriangulationView.jl")

export pullback_area_form
export pushforward_normal, pushforward_reference_normal, pushforward_parametric_normal
export BoundaryTriangulation, SkeletonTriangulation
export generate_ptr, coarse_cube_model

export R1p, A_cube2panel, A_panel2cube, b_panel2cube
export NPANELS, CUBE_HALF_EDGE
export get_nodes_from_coords

export get_radius, get_thickness
export normal_vec, tangent_vec
export get_surface_normal

end
