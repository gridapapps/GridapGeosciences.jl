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
import GridapGeosciences.Geometry: _CCAM_panel_wise_node_ids
import GridapGeosciences.Geometry: _CCAM_cube_nodes_3d
import GridapGeosciences.Geometry: setup_panel_cmaps
import GridapGeosciences.Geometry: ParametricCellField, geo_map_func, latlon_geo_map_func
import GridapGeosciences.Geometry: AmbientCellField
import GridapGeosciences.Geometry: get_panel_ids, get_forward_map_generator, get_radius, get_thickness
import GridapGeosciences.Geometry: pullback_area_form, pushforward_normal, 
                                   pushforward_reference_normal, pushforward_parametric_normal
import GridapGeosciences.Geometry: NPANELS, CUBE_HALF_EDGE
import GridapGeosciences.Geometry: ParametricModels, get_parametric_model
import GridapGeosciences.Geometry: get_surface_normal

using GridapGeosciences.Fields
import GridapGeosciences.Fields: CubedSphereForwardMap, Cartesian2SphericalMap

using GridapGeosciences.Visualisation
import GridapGeosciences.Visualisation: writevtk_with_cell_geomap, write_vtk_file_with_cell_geomap
import GridapGeosciences.Visualisation: createvtk_with_cell_geomap, create_vtk_file_with_cell_geomap


include("CubedSphere2DParametricOctreeDistributedDiscreteModel.jl")
include("CubedSphere3DParametricOctreeDistributedDiscreteModel.jl")
include("CubedSphere2DParametricDistributedDiscreteModel.jl")
include("CubedSphereAmbientDistributedDiscreteModel.jl")
include("CubedSphereAmbientOctreeDistributedDiscreteModel.jl")
include("ParametricCellField.jl")
include("AmbientCellField.jl")
include("CellFields.jl")
include("PanelIds.jl")
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
import GridapGeosciences.Geometry: AmbientMapCellField
import GridapGeosciences.Geometry: MetricCellField
import GridapGeosciences.Geometry: InvMetricCellField
import GridapGeosciences.Geometry: MeasureCellField
import GridapGeosciences.Geometry: Δs
import GridapGeosciences.Geometry: ∇s
import GridapGeosciences.Geometry: BFTATDMIM, 
                                   IntrinsicAtlasDiscreteModel, 
                                   ExtrinsicAtlasDiscreteModel
include("AtlasOctreeDistributedDiscreteModels.jl")
export AtlasOctreeDistributedDiscreteModel
include("AtlasDistributedDiscreteModels.jl")
export AtlasDiscreteModel
export IntrinsicAtlasDistributedDiscreteModel
export ExtrinsicAtlasDistributedDiscreteModel
export AtlasDistributedDiscreteModel
export get_distributed_refined_models
export get_distributed_cubed_sphere_refined_models
export get_distributed_intrinsic_cubed_sphere_refined_models
export get_distributed_extrinsic_cubed_sphere_refined_models

import Gridap.FESpaces: FESpace, compute_cell_bases_changes
import GridapDistributed: generate_gids, _find_vector_type, _add_distributed_constraint, DistributedSingleFieldFESpace
import GridapDistributed: add_ghost_cells
import GridapGeosciences.FESpaces: _generate_face_to_master_cell_id, _generate_change_of_basis_matrices, _get_value_type
include("GradConformingFESpaces.jl")

export CubedSphere2DParametricOctreeDistributedDiscreteModel
export CubedSphere3DParametricOctreeDistributedDiscreteModel
export CubedSphereParametricDistributedDiscreteModel
export CubedSphere2DParametricDistributedDiscreteModel, CubedSphere3DParametricDistributedDiscreteModel
export CubedSphereAmbientDistributedDiscreteModel
export CubedSphereAmbientOctreeDistributedDiscreteModel
export CubedSphere2DAmbientOctreeDistributedDiscreteModel, CubedSphere3DAmbientOctreeDistributedDiscreteModel
export ParametricCellField, geo_map_func, get_panel_ids, latlon_geo_map_func
export AmbientCellField
export get_forward_map_generator, get_radius, get_thickness
export get_parametric_model
export writevtk_with_cell_geomap, write_vtk_file_with_cell_geomap
export createvtk_with_cell_geomap, create_vtk_file_with_cell_geomap, create_pvtk_file_with_cell_geomap

export distributed_panel_ids
export DistributedAdaptivityGlue
export get_distributed_extrinsic_cubed_sphere_refined_models
export get_panel_ids, get_owned_panel_ids, get_skel_panel_ids
# export BoundaryTriangulation
export pullback_area_form
export pushforward_reference_normal, pushforward_parametric_normal, get_surface_normal
export get_octree_refined_models, get_3D_octree_refined_models
export get_octree_ambient_refined_models, get_3D_octree_ambient_refined_models

export CellField




end
