module Distributed

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.CellData, Gridap.ReferenceFEs
using Gridap.Adaptivity, Gridap.Helpers, Gridap.Visualization
using Gridap.Algebra, Gridap.FESpaces
using LinearAlgebra
using FillArrays

using GridapDistributed
using P4est_wrapper
using GridapP4est
using PartitionedArrays

import PartitionedArrays: getany

import GridapDistributed: DistributedCellField, DistributedTriangulation
import GridapDistributed: DistributedFaceLabeling
import GridapDistributed: DistributedDiscreteModel, GenericDistributedDiscreteModel
import GridapDistributed: BoundaryTriangulation
import GridapDistributed: CellField

using GridapGeosciences.Geometry
import GridapGeosciences.Geometry: get_radius, get_thickness
import GridapGeosciences.CellData: pullback_area_form, pushforward_normal, 
                                   pushforward_reference_normal, pushforward_parametric_normal
import GridapGeosciences.Geometry: NPANELS, CUBE_HALF_EDGE
import GridapGeosciences.Geometry: get_surface_normal

using GridapGeosciences.Fields
using GridapGeosciences.Visualisation
import GridapGeosciences.Visualisation: writevtk_with_cell_geomap, write_vtk_file_with_cell_geomap
import GridapGeosciences.Visualisation: createvtk_with_cell_geomap, create_vtk_file_with_cell_geomap

include("Vtk.jl")
include("Triangulations.jl")


## AtlasDiscreteModels-related stuff
import GridapGeosciences.Geometry: ManifoldStyle
import GridapGeosciences.Geometry: CoarseMeshInfo
import GridapGeosciences.Geometry: CoarseMesh
import GridapGeosciences.Geometry: AtlasDiscreteModel
import GridapGeosciences.Geometry: AtlasGrid
import GridapGeosciences.Geometry: get_atlas_grid
import GridapGeosciences.Geometry: BFTATDM

import GridapGeosciences.CellData: AmbientMapCellField, LatLonMapCellField
import GridapGeosciences.CellData: MetricCellField
import GridapGeosciences.CellData: InvMetricCellField
import GridapGeosciences.CellData: MeasureCellField
import GridapGeosciences.CellData: Δs
import GridapGeosciences.CellData: ∇s

import GridapGeosciences.Geometry: BFTATDMIM, 
                                   IntrinsicAtlasDiscreteModel, 
                                   ExtrinsicAtlasDiscreteModel

include("AtlasOctreeDistributedDiscreteModels.jl")
export AtlasOctreeDistributedDiscreteModel, get_atlas_model

include("ExtrudedAtlasOctreeDistributedDiscreteModels.jl")
export ExtrudedAtlasOctreeDistributedDiscreteModel

include("AtlasDistributedDiscreteModels.jl")
include("DistributedCellFields.jl")
include("SurfaceDiffOps.jl")

export IntrinsicAtlasDistributedDiscreteModel
export ExtrinsicAtlasDistributedDiscreteModel
export AtlasDistributedDiscreteModel
export AdaptedAtlasDistributedDiscreteModel
export AdaptedIntrinsicAtlasDistributedDiscreteModel
export AdaptedExtrinsicAtlasDistributedDiscreteModel

export generate_distributed_refined_models
export generate_octree_distributed_refined_models
export generate_extruded_octree_distributed_refined_models

import Gridap.FESpaces: FESpace, compute_cell_bases_changes
import GridapDistributed: generate_gids, _find_vector_type, _add_distributed_constraint, DistributedSingleFieldFESpace
import GridapDistributed: add_ghost_cells
import GridapGeosciences.FESpaces: _generate_face_to_master_cell_id, _generate_change_of_basis_matrices, _get_value_type
include("GradConformingFESpaces.jl")

export get_radius, get_thickness
export writevtk_with_cell_geomap, write_vtk_file_with_cell_geomap
export createvtk_with_cell_geomap, create_vtk_file_with_cell_geomap, create_pvtk_file_with_cell_geomap

export DistributedAdaptivityGlue
export get_distributed_refined_models
# export BoundaryTriangulation
export pullback_area_form
export pushforward_reference_normal, pushforward_parametric_normal, get_surface_normal

end
