module Geometry
using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
using LinearAlgebra
using FillArrays


import Gridap.Geometry: TriangulationView
import Gridap.Geometry: FaceToCellGlue, FaceCompressedVector, push_normal

using GridapGeosciences.Fields
import GridapGeosciences.Fields: MatMultField
import GridapGeosciences.Fields: CubedSphereInverseMap

using GridapGeosciences.Helpers
import GridapGeosciences.Helpers: inv_metric, forward_jacobian

include("PanelIds.jl")
include("CubeSurface.jl")
include("PanelMatrices.jl")
include("CubedSphereParametricDiscreteModel.jl")
include("CubedSphereAmbientDiscreteModel.jl")
include("BoundaryTriangulations.jl")
include("SkeletonTriangulations.jl")
include("AdaptedTriangulations.jl")
include("ParametricCellField.jl")
include("AmbientCellField.jl")
include("TriangulationView.jl")
include("TriangulationPanelIds.jl")

export get_panel_ids, get_forward_map_generator, geo_map_func, latlon_geo_map_func
export pullback_area_form
export pushforward_normal, get_facet_normal, get_mapped_facet_normal
export BoundaryTriangulation, SkeletonTriangulation
export pushforward_trian
export generate_ptr, coarse_cube_model

export coarse_parametric_model
export R1p, A_cube2panel, A_panel2cube, b_panel2cube
export CubedSphereParametricDiscreteModel, CubedSphereAmbientDiscreteModel
export CubedSphere2DParametricDiscreteModel, CubedSphere3DParametricDiscreteModel
export NPANELS, CUBE_HALF_EDGE
export get_nodes_from_coords

export ParametricCellField, AmbientCellField
export _pushforward_normal
export _pullback_area_form

export get_radius, get_thickness
export get_refined_models, get_ambient_refined_models
export get_inverse_map_generator
export get_parametric_model
export get_surface_normal
export dagger, perp

export AmbientModels


## AtlasDiscreteModels new machinery (will eventually replace enterily what we have above)
## Some definitions in the below included julia source files must be moved to other
## GridapGeosciences modules for consistency, e.g., to GridapGeosciences.Fields
using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.ReferenceFEs, Gridap.Helpers
using Gridap.Adaptivity, Gridap.Visualization
import Gridap.TensorValues: symmetric_part, SymTensorValue
include("CoarseMeshes.jl")
include("AtlasGrids.jl") 
include("AtlasDiscreteModels.jl")

export AtlasGrid, AtlasDiscreteModel
export IntrinsicManifold, ExtrinsicManifold
export MetricCellField

export CylinderMesh, CylinderChartMap, CylinderMetricField, CylinderInvMetricField 
export MobiusStripMesh, MobiusChartMap, MobiusMetricField, MobiusInvMetricField
export CubedSphereMesh, CubedSphereMap, CubedSphereMetricField, CubedSphereInvMetricField
export get_atlas_grid
export get_cell_ambient_maps, get_cell_metric, get_cell_inv_metric
export get_coarse_mesh
export JtJ

end
