module Geometry
using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
using LinearAlgebra
using FillArrays

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.ReferenceFEs, Gridap.Helpers
using Gridap.Adaptivity, Gridap.Visualization
import Gridap.TensorValues: symmetric_part, SymTensorValue, ThirdOrderTensorValue, contracted_product
import Gridap.Geometry: Grid, GridView
import GridapGeosciences.Helpers: J
import GridapGeosciences.Fields: CubedSphereMap, CubedSphereWithThicknessMap

include("CoarseMeshes.jl")
include("CylinderMesh.jl")
include("MobiusStripMesh.jl")
include("CubedSphereMesh.jl")
include("CubedSphereWithThicknessMesh.jl")
include("ExtrudedCubedSphereWithThicknessMesh.jl")

include("AtlasGrids.jl")
include("AtlasDiscreteModels.jl")

export AtlasGrid, AtlasDiscreteModel
export IntrinsicAtlasDiscreteModel, ExtrinsicAtlasDiscreteModel
export IntrinsicManifold, ExtrinsicManifold

export CylinderMesh
export MobiusStripMesh
export CubedSphereMesh
export CubedSphereWithThicknessMesh
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

include("BoundaryTriangulations.jl")
include("AdaptedTriangulations.jl")

export pullback_area_form
export pushforward_normal, pushforward_reference_normal, pushforward_parametric_normal

export NPANELS, CUBE_HALF_EDGE
export get_radius, get_thickness
export normal_vec, tangent_vec
export get_surface_normal

end
