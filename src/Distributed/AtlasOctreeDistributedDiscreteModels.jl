# DistributedAtlasDiscreteModels.jl
#
# Distributed atlas discrete model built on p4est / GridapP4est.
# Extends AtlasGrid/AtlasDiscreteModel from AtlasGrids.jl/AtlasDiscreteModels.jl
# to the distributed setting via OctreeDistributedDiscreteModel.
#
# Key design change vs. old SBAtlasModels.jl:
#   - AtlasGrid now stores LOCAL (α,β) reference coords (not 3D ambient coords).
#   - Ambient Da-dimensional coords are computed only in visualization_data
#     (inherited from AtlasDiscreteModels.jl via _local_to_ambient).
#   - The p4est infrastructure provides (α,β) coords per fine cell directly;
#     these are stored as-is into cell_chart_coords.
#
# Run (e.g. 4 MPI processes):
#   mpiexec -n 4 julia --project=. <script>

include("AtlasDiscreteModels.jl")

using GridapGeosciences
using GridapDistributed
using GridapDistributed: GenericDistributedDiscreteModel, get_cell_gids
using GridapP4est
using GridapP4est: OctreeDistributedDiscreteModel
using GridapDistributed: local_views

import GridapGeosciences.Distributed: _generate_octree_dmodel_alpha_beta_coordinates_and_panels

# ============================================================
# AtlasOctreeDistributedDiscreteModel
# ============================================================

"""
    AtlasOctreeDistributedDiscreteModel{A,B,M}

Distributed discrete model for an atlas-based 2D manifold mesh built on p4est.

# Fields
- `octree_dmodel`  — underlying `OctreeDistributedDiscreteModel{2,2}` owning the
  p4est forest, MPI topology, and adaptive refinement support.
- `dmodel`         — `GenericDistributedDiscreteModel{2,2}` wrapping per-rank
  `AtlasDiscreteModel` instances, each carrying an `AtlasGrid{2,Da}` with local
  (α,β) reference coords and the per-chart ambient maps.
- (M)              — `ManifoldStyle` type parameter, same meaning as in `AtlasGrid`.

The `DistributedDiscreteModel` interface delegates to `dmodel`.
Ambient Da-dimensional coords are computed on demand in `visualization_data`
via `_local_to_ambient` (defined in AtlasDiscreteModels.jl).
"""
struct AtlasOctreeDistributedDiscreteModel{
  A <: OctreeDistributedDiscreteModel{2,2},
  B <: GenericDistributedDiscreteModel{2,2},
  M <: ManifoldStyle,
} <: GridapDistributed.DistributedDiscreteModel{2,2}
  octree_dmodel :: A
  dmodel        :: B
end

# ----------------------------------------------------------
# DistributedDiscreteModel interface (delegate to dmodel)
# ----------------------------------------------------------

GridapDistributed.local_views(m::AtlasOctreeDistributedDiscreteModel) =
  local_views(m.dmodel)

GridapDistributed.get_cell_gids(m::AtlasOctreeDistributedDiscreteModel) =
  get_cell_gids(m.dmodel)

# ----------------------------------------------------------
# Custom API
# ----------------------------------------------------------

get_octree_dmodel(m::AtlasOctreeDistributedDiscreteModel) = m.octree_dmodel
get_dmodel(m::AtlasOctreeDistributedDiscreteModel)        = m.dmodel
ManifoldStyle(::Type{<:AtlasOctreeDistributedDiscreteModel{A,B,M}}) where {A,B,M} = M()
ManifoldStyle(m::AtlasOctreeDistributedDiscreteModel) = ManifoldStyle(typeof(m))

get_cell_metric(m::AtlasOctreeDistributedDiscreteModel) =
  map(get_cell_metric, local_views(m.dmodel))

# ============================================================
# Constructors
# ============================================================

"""
    AtlasOctreeDistributedDiscreteModel(ranks, info::CoarseMeshInfo;
                                        num_initial_uniform_refinements=0,
                                        manifold_style=ExtrinsicManifold())

Build a distributed `AtlasOctreeDistributedDiscreteModel` from a `CoarseMeshInfo`.
The coarse `DiscreteModel` stored in `info.model` is passed to p4est for forest
construction; `info.cell_chart_coords`, `info.ambient_maps`, and `info.metric_fields`
are forwarded to each per-rank `AtlasGrid`.

The per-rank `AtlasGrid` stores local reference coords from `info.cell_chart_coords`
(not ambient 3D coords). Ambient coords are computed lazily only during
VTK visualization via `_local_to_ambient`.
"""
function AtlasOctreeDistributedDiscreteModel(
    ranks,
    info  :: CoarseMeshInfo;
    num_initial_uniform_refinements :: Int = 0,
    manifold_style = ExtrinsicManifold(),
)
  ambient_maps       = info.ambient_maps
  metric_fields      = info.metric_fields
  coarse_cell_panels = collect(1:Gridap.Geometry.num_cells(info.model))

  octree_dmodel, cell_wise_chart_coords, cell_panels =
    _generate_octree_dmodel_alpha_beta_coordinates_and_panels(
      ranks,
      info.model,
      num_initial_uniform_refinements,
      info.cell_chart_coords,
      coarse_cell_panels,
    )

  atlas_models = map(
    local_views(octree_dmodel.dmodel),
    cell_wise_chart_coords,
    cell_panels,
  ) do omodel, cell_chart_coords, cell_to_chart_local

    param_grid        = Gridap.Geometry.get_grid(omodel)
    grid_topology     = Gridap.Geometry.get_grid_topology(omodel)
    face_labeling     = Gridap.Geometry.get_face_labeling(omodel)
    orientation_style = Gridap.Geometry.OrientationStyle(param_grid)

    cell_ambient_maps = lazy_map(Reindex(ambient_maps), cell_to_chart_local)
    cell_metric       = lazy_map(Reindex(metric_fields),  cell_to_chart_local)

    atlas_grid = AtlasGrid(
      param_grid,
      cell_chart_coords,
      cell_ambient_maps,
      cell_metric,
      orientation_style,
      manifold_style,
    )

    AtlasDiscreteModel(atlas_grid, grid_topology, face_labeling)
  end

  dmodel = GenericDistributedDiscreteModel(
    atlas_models, get_cell_gids(octree_dmodel.dmodel))

  M = typeof(manifold_style)
  AtlasOctreeDistributedDiscreteModel{typeof(octree_dmodel),typeof(dmodel),M}(octree_dmodel, dmodel)
end

"""
    AtlasOctreeDistributedDiscreteModel(ranks, mesh::CoarseMesh;
                                        num_initial_uniform_refinements=0,
                                        manifold_style=ExtrinsicManifold())

Build a distributed `AtlasOctreeDistributedDiscreteModel` from a mesh descriptor
(e.g. `CubedSphereMesh(1.0)`). Calls `get_coarse_mesh(mesh)` and delegates to the
`CoarseMeshInfo` constructor.
"""
function AtlasOctreeDistributedDiscreteModel(
    ranks,
    mesh  :: CoarseMesh;
    num_initial_uniform_refinements :: Int = 0,
    manifold_style = ExtrinsicManifold(),
)
  AtlasOctreeDistributedDiscreteModel(ranks, get_coarse_mesh(mesh);
    num_initial_uniform_refinements, manifold_style)
end

"""
    AtlasOctreeDistributedDiscreteModel(ranks, radius::Real;
                                        num_initial_uniform_refinements=0,
                                        manifold_style=ExtrinsicManifold())

Backward-compatible constructor: builds the standard cubed sphere of the given
`radius`. Equivalent to `AtlasOctreeDistributedDiscreteModel(ranks, CubedSphereMesh(radius); ...)`.
"""
function AtlasOctreeDistributedDiscreteModel(
    ranks,
    radius :: Real;
    num_initial_uniform_refinements :: Int = 0,
    manifold_style = ExtrinsicManifold(),
)
  AtlasOctreeDistributedDiscreteModel(ranks, CubedSphereMesh(Float64(radius));
    num_initial_uniform_refinements, manifold_style)
end
