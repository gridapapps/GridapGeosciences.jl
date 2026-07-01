
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


"""
Bilinear map from p4est integer quadrant coordinates (x, y) to physical space,
given the four corner vertices of the coarse tree in p4est ordering:
  v0: (xi=0, yi=0),  v1: (xi=1, yi=0)
  v2: (xi=0, yi=1),  v3: (xi=1, yi=1)

Equivalent to `p4est_qcoord_to_vertex` (2D, P4_TO_P8=false) but takes
tree vertices directly instead of a connectivity pointer + treeid.
"""
function quad_coord_to_vertex_coord(coarse_vertices,
                                    x::p4est_qcoord_t,
                                    y::p4est_qcoord_t)
  P4EST_ROOT_LEN = p4est_qcoord_t(1) << P4est_wrapper.P4EST_MAXLEVEL

  tx = Float64(x) / Float64(P4EST_ROOT_LEN)
  ty = Float64(y) / Float64(P4EST_ROOT_LEN)

  wx = (1.0 - tx, tx)
  wy = (1.0 - ty, ty)

  vx = 0.0
  vy = 0.0
  k = 1
  for yi in 1:2
    for xi in 1:2
      w = wy[yi] * wx[xi]
      vx += w * coarse_vertices[k][1]
      vy += w * coarse_vertices[k][2]
      k += 1
    end
  end
  vx, vy
end

"""
Trilinear map from p8est integer octant coordinates (x, y, z) to physical space,
given the eight corner vertices of the coarse tree in p8est ordering:
  v0: (xi=0, yi=0, zi=0),  v1: (xi=1, yi=0, zi=0)
  v2: (xi=0, yi=1, zi=0),  v3: (xi=1, yi=1, zi=0)
  v4: (xi=0, yi=0, zi=1),  v5: (xi=1, yi=0, zi=1)
  v6: (xi=0, yi=1, zi=1),  v7: (xi=1, yi=1, zi=1)

Equivalent to `p8est_qcoord_to_vertex` (3D) but takes tree vertices directly
instead of a connectivity pointer + treeid.
"""
function oct_coord_to_vertex_coord(coarse_vertices,
                                   x::p4est_qcoord_t,
                                   y::p4est_qcoord_t,
                                   z::p4est_qcoord_t)
  P8EST_ROOT_LEN = p4est_qcoord_t(1) << P4est_wrapper.P8EST_MAXLEVEL

  tx = Float64(x) / Float64(P8EST_ROOT_LEN)
  ty = Float64(y) / Float64(P8EST_ROOT_LEN)
  tz = Float64(z) / Float64(P8EST_ROOT_LEN)

  wx = (1.0 - tx, tx)
  wy = (1.0 - ty, ty)
  wz = (1.0 - tz, tz)

  vx = 0.0
  vy = 0.0
  vz = 0.0
  k = 1
  for zi in 1:2
    for yi in 1:2
      for xi in 1:2
        w = wz[zi] * wy[yi] * wx[xi]
        vx += w * coarse_vertices[k][1]
        vy += w * coarse_vertices[k][2]
        vz += w * coarse_vertices[k][3]
        k += 1
      end
    end
  end
  vx, vy, vz
end

function get_quad_vertex_coord(coarse_vertices,
                               x::p4est_qcoord_t,
                               y::p4est_qcoord_t,
                               level::Int8,
                               corner::Cint)

  myself = Ref{p4est_quadrant_t}(
    p4est_quadrant_t(x,y,level,Int8(0),Int16(0),P4est_wrapper.quadrant_data(Clong(0)))
  )
  neighbour = Ref{p4est_quadrant_t}(myself[])
  if corner == 1
      p4est_quadrant_face_neighbor(myself,corner,neighbour)
  elseif corner == 2
      p4est_quadrant_face_neighbor(myself,corner+1,neighbour)
  elseif corner == 3
      p4est_quadrant_corner_neighbor(myself,corner,neighbour)
  end
  # Extract numerical coordinates of lower_left
  # corner of my corner neighbour
  quad_coord_to_vertex_coord(coarse_vertices,
                             neighbour[].x,
                             neighbour[].y)
end

function  p8est_get_quadrant_vertex_coordinates(coarse_vertices,
                                                x::p4est_qcoord_t,
                                                y::p4est_qcoord_t,
                                                z::p4est_qcoord_t,
                                                level::Int8,
                                                corner::Cint)

  myself = Ref{p8est_quadrant_t}(
    p8est_quadrant_t(x,y,z,level,Int8(0),Int16(0),P4est_wrapper.quadrant_data(Clong(0)))
  )
  neighbour = Ref{p8est_quadrant_t}(myself[])

  if ( corner == 1 )
    p8est_quadrant_face_neighbor(myself,Cint(1),neighbour)
  elseif ( corner == 2 )
    p8est_quadrant_face_neighbor(myself,Cint(3),neighbour)
  elseif ( corner == 3 )
    p8est_quadrant_edge_neighbor(myself,Cint(11),neighbour)
  elseif ( corner == 4 )
    p8est_quadrant_face_neighbor(myself,Cint(5),neighbour)
  elseif ( corner == 5 )
    p8est_quadrant_edge_neighbor(myself,Cint(7),neighbour)
  elseif ( corner == 6 )
    p8est_quadrant_edge_neighbor(myself,Cint(3),neighbour)
  elseif ( corner == 7 )
    p8est_quadrant_corner_neighbor(myself,Cint(7),neighbour)
  end

  # Extract numerical coordinates of lower_left corner of my corner neighbour
  oct_coord_to_vertex_coord(coarse_vertices,
                         neighbour[].x,
                         neighbour[].y,
                         neighbour[].z)
                         
end

_ghost_quadrant_ptr(::GridapP4est.P4estType, pXest_ghost) =
  Ptr{P4est_wrapper.p4est_quadrant_t}(pXest_ghost.ghosts.array)

_ghost_quadrant_ptr(::GridapP4est.P8estType, pXest_ghost) =
  Ptr{P4est_wrapper.p8est_quadrant_t}(pXest_ghost.ghosts.array)

_get_cell_vertex_coord(coarse_vertices, quadrant::P4est_wrapper.p4est_quadrant_t, vertex::Cint) =
  get_quad_vertex_coord(coarse_vertices, quadrant.x, quadrant.y, quadrant.level, vertex)

_get_cell_vertex_coord(coarse_vertices, quadrant::P4est_wrapper.p8est_quadrant_t, vertex::Cint) =
  p8est_get_quadrant_vertex_coordinates(coarse_vertices, quadrant.x, quadrant.y, quadrant.z, quadrant.level, vertex)

function generate_cell_coordinates(ranks,
                                   coarse_cell_wise_vertex_coordinates,
                                   ptr_pXest,
                                   ptr_pXest_ghost,
                                   pXest_type::GridapP4est.PXestType)
  Dc = GridapP4est.num_cell_dims(pXest_type)
  PXEST_CORNERS = 2^Dc
  pXest_ghost = ptr_pXest_ghost[]
  pXest = ptr_pXest[]

  ptr_ghost_quadrants = _ghost_quadrant_ptr(pXest_type, pXest_ghost)

  tree_offsets = unsafe_wrap(Array, pXest_ghost.tree_offsets, pXest_ghost.num_trees+1)
  map(ranks) do _
    ncells = pXest.local_num_quadrants + pXest_ghost.ghosts.elem_count
    data = Vector{Point{Dc,Float64}}(undef, ncells * PXEST_CORNERS)
    ptr  = [1:PXEST_CORNERS:(ncells*PXEST_CORNERS); ncells*PXEST_CORNERS+1]
    current = 1
    for itree = 1:pXest_ghost.num_trees
      tree = GridapP4est.pXest_tree_array_index(pXest_type, pXest, itree-1)[]
      for cell = 1:tree.quadrants.elem_count
        quadrant = GridapP4est.pXest_quadrant_array_index(pXest_type, tree, cell-1)[]
        for vertex = 1:PXEST_CORNERS
          vcoords = _get_cell_vertex_coord(coarse_cell_wise_vertex_coordinates[itree],
                                           quadrant, Cint(vertex-1))
          data[current] = Point{Dc,Float64}(vcoords...)
          current += 1
        end
      end
    end
    # Go over ghost cells
    for i = 1:pXest_ghost.num_trees
      for j = tree_offsets[i]:tree_offsets[i+1]-1
        quadrant = ptr_ghost_quadrants[j+1]
        for vertex = 1:PXEST_CORNERS
          vcoords = _get_cell_vertex_coord(coarse_cell_wise_vertex_coordinates[i],
                                           quadrant, Cint(vertex-1))
          data[current] = Point{Dc,Float64}(vcoords...)
          current += 1
        end
      end
    end
    Gridap.Arrays.Table(data, ptr)
  end
end

function generate_cell_to_chart_id(ranks,
                                   coarse_discrete_model,
                                   ptr_pXest_connectivity,
                                   ptr_pXest,
                                   ptr_pXest_ghost)

  pXest_ghost = ptr_pXest_ghost[]
  pXest = ptr_pXest[]

  coarse_cell_to_chart_id = collect(1:num_cells(coarse_discrete_model))

  tree_offsets = unsafe_wrap(Array, pXest_ghost.tree_offsets, pXest_ghost.num_trees+1)
  cell_to_chart_id=map(ranks) do part
     ncells=pXest.local_num_quadrants+pXest_ghost.ghosts.elem_count
     cell_to_chart_id = Vector{Int}(undef,ncells)
     current_cell=1
     for itree=1:pXest_ghost.num_trees
       tree = GridapP4est.p4est_tree_array_index(pXest.trees, itree-1)[]
       for cell=1:tree.quadrants.elem_count
          cell_to_chart_id[current_cell]=coarse_cell_to_chart_id[itree]
          current_cell=current_cell+1
       end
     end
     # Go over ghost cells
     for i=1:pXest_ghost.num_trees
      for j=tree_offsets[i]:tree_offsets[i+1]-1
          cell_to_chart_id[current_cell]=i
          current_cell=current_cell+1
       end
     end
     cell_to_chart_id
  end
  cell_to_chart_id
end

function _dummy_grid_and_topology_function(pXest_type::GridapP4est.P4P8estType,
                                           cell_corner_lids,
                                           ptr_pXest_connectivity,
                                           ptr_pXest,
                                           ptr_pXest_ghost)
  grid,topology=_generate_topology_grid_and_topology(pXest_type, cell_corner_lids)
end

function _generate_topology_grid_and_topology(pXest_type::GridapP4est.PXestType,
                                              cell_corner_lids)

  Dc = GridapP4est.num_cell_dims(pXest_type)
  map(cell_corner_lids) do cell_corner_lids
    n_corners = maximum(cell_corner_lids.data;init=0)
    T=Point{Dc,Float64}
    corner_coords = Vector{T}(undef,n_corners)
    corner_coords .= zero(T)

    poly  = (Dc==2) ? QUAD : HEX
    reffe = Gridap.ReferenceFEs.ReferenceFE(poly,lagrangian,Float64,1)
    cell_types = fill(1,length(cell_corner_lids))

    grid = Gridap.Geometry.UnstructuredGrid(
      corner_coords,cell_corner_lids,[reffe],cell_types,Gridap.Geometry.NonOriented()
    )
    topology = Gridap.Geometry.UnstructuredGridTopology(
      corner_coords,cell_corner_lids,cell_types,[poly],Gridap.Geometry.NonOriented()
    )
    return grid, topology
  end |> tuple_of_arrays
end

function generate_octree_dmodel_cell_chart_coords_and_cell_to_chart_id(ranks,
                                                                       coarse_model::DiscreteModel{Dc,Dc}, 
                                                                       coarse_cell_chart_coords,
                                                                       num_uniform_refinements) where Dc

   comm = ranks.comm
   pXest_type = GridapP4est._dim_to_pXest_type(Dc)
   pXest_refinement_rule_type = GridapP4est.PXestUniformRefinementRuleType()

   ptr_pXest_connectivity=GridapP4est.setup_pXest_connectivity(coarse_model)
   
   # Create a new forest
   ptr_pXest = GridapP4est.setup_pXest(pXest_type,comm,ptr_pXest_connectivity,num_uniform_refinements)
   
   # Build the ghost layer
   ptr_pXest_ghost=GridapP4est.setup_pXest_ghost(pXest_type,ptr_pXest)

   ptr_pXest_lnodes=GridapP4est.setup_pXest_lnodes_nonconforming(pXest_type, ptr_pXest, ptr_pXest_ghost)

   dmodel, non_conforming_glue  = GridapP4est.setup_non_conforming_distributed_discrete_model(pXest_type,
                                                    pXest_refinement_rule_type,
                                                    ranks,
                                                    coarse_model,
                                                    ptr_pXest_connectivity,
                                                    ptr_pXest,
                                                    ptr_pXest_ghost,
                                                    ptr_pXest_lnodes;
                                                    grid_and_topology_function=_dummy_grid_and_topology_function)


   GridapP4est.pXest_lnodes_destroy(pXest_type,ptr_pXest_lnodes)

   connectivity_ref = GridapP4est.PXestConnectivityRef(pXest_type, ptr_pXest_connectivity)

   omodel=OctreeDistributedDiscreteModel(Dc,
                                  Dc,
                                  ranks,
                                  dmodel,
                                  non_conforming_glue,
                                  coarse_model,
                                  ptr_pXest_connectivity,
                                  ptr_pXest,
                                  pXest_type,
                                  pXest_refinement_rule_type,
                                  connectivity_ref)

  cell_to_chart_id = generate_cell_to_chart_id(ranks,
                                               coarse_model,
                                               omodel.ptr_pXest_connectivity,
                                               omodel.ptr_pXest,
                                               ptr_pXest_ghost)

  cell_wise_chart_coords = generate_cell_coordinates(ranks,
                                   coarse_cell_chart_coords,
                                   omodel.ptr_pXest,
                                   ptr_pXest_ghost,
                                   pXest_type)

  GridapP4est.pXest_ghost_destroy(pXest_type,ptr_pXest_ghost)
                                 
  omodel, cell_wise_chart_coords, cell_to_chart_id
end


# ============================================================
# AtlasOctreeDistributedDiscreteModel
# ============================================================

"""
    AtlasOctreeDistributedDiscreteModel{A,B,M}

Distributed discrete model for an atlas-based 2D manifold mesh built on p4est.

# Fields
- `octree_dmodel`  — underlying `OctreeDistributedDiscreteModel{Dc,Dc}` owning the
  p4est forest, MPI topology, and adaptive refinement support.
- `atlas_dmodel`   — `GenericDistributedDiscreteModel{Dc,Dp}` wrapping per-rank
  `AtlasDiscreteModel` instances, each carrying an `AtlasGrid{Dc,Da}` with local
  (α,β) reference coords and the per-chart ambient maps.
- (M)              — `ManifoldStyle` type parameter, same meaning as in `AtlasGrid`.

The `DistributedDiscreteModel` interface delegates to `atlas_dmodel`.
Ambient Da-dimensional coords are computed on demand in `visualization_data`
via `_local_to_ambient` (defined in AtlasDiscreteModels.jl).
"""
struct AtlasOctreeDistributedDiscreteModel{
  Dc, Dp,
  A <: OctreeDistributedDiscreteModel{Dc,Dc},
  B <: GenericDistributedDiscreteModel{Dc,Dp},
  M <: ManifoldStyle,
} <: GridapDistributed.DistributedDiscreteModel{Dc,Dp}
  octree_dmodel :: A
  atlas_dmodel  :: B
  coarse_info   :: CoarseMeshInfo
end

# ----------------------------------------------------------
# DistributedDiscreteModel interface (delegate to dmodel)
# ----------------------------------------------------------

GridapDistributed.local_views(m::AtlasOctreeDistributedDiscreteModel) =
  local_views(m.atlas_dmodel)

GridapDistributed.get_cell_gids(m::AtlasOctreeDistributedDiscreteModel) =
  get_cell_gids(m.atlas_dmodel)

GridapDistributed.get_face_gids(m::AtlasOctreeDistributedDiscreteModel, dim::Integer) =
  get_face_gids(m.atlas_dmodel, dim)

# ----------------------------------------------------------
# Custom API
# ----------------------------------------------------------

get_octree_dmodel(m::AtlasOctreeDistributedDiscreteModel) = m.octree_dmodel
get_atlas_model(m::AtlasOctreeDistributedDiscreteModel)  = m.atlas_dmodel
ManifoldStyle(::Type{<:AtlasOctreeDistributedDiscreteModel{Dc,Dp,A,B,M}}) where {Dc,Dp,A,B,M} = M()
ManifoldStyle(m::AtlasOctreeDistributedDiscreteModel) = ManifoldStyle(typeof(m))

function Base.getproperty(m::AtlasOctreeDistributedDiscreteModel, sym::Symbol)
  if sym === :face_gids
    return getfield(getfield(m, :atlas_dmodel), :face_gids)
  end
  return getfield(m, sym)
end

Base.propertynames(m::AtlasOctreeDistributedDiscreteModel) =
  (fieldnames(typeof(m))..., :face_gids)

get_cell_metric(m::AtlasOctreeDistributedDiscreteModel) =
  map(get_cell_metric, local_views(m.atlas_dmodel))

# ============================================================
# Constructors
# ============================================================

"""
    AtlasOctreeDistributedDiscreteModel(ranks, info::CoarseMeshInfo,
                                        num_initial_uniform_refinements;
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
    info  :: CoarseMeshInfo,
    num_refinements;
    manifold_style = ExtrinsicManifold(),
)
  ambient_maps       = info.ambient_maps
  metric_fields      = info.metric_fields
  coarse_model       = info.model
  coarse_cell_panels = collect(1:Gridap.Geometry.num_cells(coarse_model))
  coarse_cell_chart_coords = info.cell_chart_coords

  octree_dmodel,cell_wise_chart_coords,cell_to_chart_id = 
    generate_octree_dmodel_cell_chart_coords_and_cell_to_chart_id(
       ranks,
       coarse_model,
       coarse_cell_chart_coords,
       num_refinements
  )                                           

  atlas_models = map(
    local_views(octree_dmodel.dmodel),
    cell_wise_chart_coords,
    cell_to_chart_id,
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

  atlas_dmodel = GenericDistributedDiscreteModel(
    atlas_models, get_cell_gids(octree_dmodel.dmodel))

  Dc = num_cell_dims(atlas_dmodel)
  Dp = num_point_dims(atlas_dmodel)
  M  = typeof(manifold_style)
  AtlasOctreeDistributedDiscreteModel{Dc,Dp,typeof(octree_dmodel),typeof(atlas_dmodel),M}(octree_dmodel, atlas_dmodel, info)
end

"""
    AtlasOctreeDistributedDiscreteModel(ranks, mesh::CoarseMesh,
                                        num_initial_uniform_refinements;
                                        manifold_style=ExtrinsicManifold())

Build a distributed `AtlasOctreeDistributedDiscreteModel` from a mesh descriptor
(e.g. `CubedSphereMesh(1.0)`). Calls `get_coarse_mesh(mesh)` and delegates to the
`CoarseMeshInfo` constructor.
"""
function AtlasOctreeDistributedDiscreteModel(
    ranks,
    mesh  :: CoarseMesh,
    num_initial_uniform_refinements;
    manifold_style = ExtrinsicManifold(),
)
  AtlasOctreeDistributedDiscreteModel(ranks,
                                      get_coarse_mesh(mesh),
                                      num_initial_uniform_refinements;
                                      manifold_style)
end

# ============================================================
# Adaptivity
# ============================================================

 function _adapt_atlas_octree_dmodel(
    octree_dmodel::OctreeDistributedDiscreteModel,
    coarse_cell_chart_coords,
    refinement_and_coarsening_flags::MPIArray{<:Vector},
)
  pXest_type              = octree_dmodel.pXest_type
  Dc                      = GridapP4est.num_cell_dims(pXest_type)
  pXest_refinement_rule_type = octree_dmodel.pXest_refinement_rule_type
  ranks                   = octree_dmodel.parts

  ptr_new_pXest = GridapP4est._refine_coarsen_balance!(octree_dmodel, refinement_and_coarsening_flags)

  ptr_pXest_ghost  = GridapP4est.setup_pXest_ghost(pXest_type, ptr_new_pXest)
  ptr_pXest_lnodes = GridapP4est.setup_pXest_lnodes_nonconforming(pXest_type, ptr_new_pXest, ptr_pXest_ghost)
  ptr_pXest_connectivity = octree_dmodel.ptr_pXest_connectivity
  coarse_model    = octree_dmodel.coarse_model

  new_dmodel, non_conforming_glue = GridapP4est.setup_non_conforming_distributed_discrete_model(
    pXest_type,
    pXest_refinement_rule_type,
    ranks,
    coarse_model,
    ptr_pXest_connectivity,
    ptr_new_pXest,
    ptr_pXest_ghost,
    ptr_pXest_lnodes;
    grid_and_topology_function=_dummy_grid_and_topology_function,
  )

  cell_wise_chart_coords = generate_cell_coordinates(
    ranks,
    coarse_cell_chart_coords,
    ptr_new_pXest,
    ptr_pXest_ghost,
    pXest_type,
  )

  cell_to_chart_id = generate_cell_to_chart_id(
    ranks,
    coarse_model,
    ptr_pXest_connectivity,
    ptr_new_pXest,
    ptr_pXest_ghost,
  )

  GridapP4est.pXest_ghost_destroy(pXest_type, ptr_pXest_ghost)
  GridapP4est.pXest_lnodes_destroy(pXest_type, ptr_pXest_lnodes)

  stride = GridapP4est.pXest_stride_among_children(
    pXest_type,
    pXest_refinement_rule_type,
    octree_dmodel.ptr_pXest,
  )

  adaptivity_glue = GridapP4est._compute_fine_to_coarse_model_glue(
    pXest_type,
    pXest_refinement_rule_type,
    ranks,
    octree_dmodel.dmodel,
    new_dmodel,
    refinement_and_coarsening_flags,
    stride,
  )

  adapted_models = map(
    local_views(octree_dmodel.dmodel),
    local_views(new_dmodel),
    adaptivity_glue,
  ) do parent_model, new_model, glue
    parent = isa(parent_model, AdaptedDiscreteModel) ? parent_model.model : parent_model
    Gridap.Adaptivity.AdaptedDiscreteModel(new_model, parent, glue)
  end

  adapted_dmodel = GridapDistributed.DistributedDiscreteModel(
    adapted_models, get_cell_gids(new_dmodel)
  )

  adapted_omodel = OctreeDistributedDiscreteModel(
    Dc, Dc,
    ranks,
    adapted_dmodel,
    non_conforming_glue,
    coarse_model,
    octree_dmodel.ptr_pXest_connectivity,
    ptr_new_pXest,
    octree_dmodel.pXest_type,
    GridapP4est.PXestUniformRefinementRuleType(),
    octree_dmodel.connectivity_ref)

  adapted_omodel, cell_wise_chart_coords, cell_to_chart_id, adaptivity_glue
end

function Gridap.Adaptivity.refine(model::AtlasOctreeDistributedDiscreteModel)
  octree_dmodel = model.octree_dmodel
  flags = map(partition(get_cell_gids(octree_dmodel))) do indices
    f = Vector{Cint}(undef, length(indices))
    f .= GridapP4est.refine_flag
    f
  end
  Gridap.Adaptivity.adapt(model, flags)
end

function Gridap.Adaptivity.adapt(
    model::AtlasOctreeDistributedDiscreteModel,
    refinement_and_coarsening_flags::MPIArray{<:Vector},
)
  info           = model.coarse_info
  manifold_style = ManifoldStyle(model)
  ambient_maps   = info.ambient_maps
  metric_fields  = info.metric_fields

  adapted_octree_dmodel, cell_wise_chart_coords, cell_to_chart_id, adaptivity_glue =
    _adapt_atlas_octree_dmodel(
      model.octree_dmodel,
      info.cell_chart_coords,
      refinement_and_coarsening_flags,
    )

  atlas_models = map(
    local_views(adapted_octree_dmodel.dmodel),
    cell_wise_chart_coords,
    cell_to_chart_id,
  ) do omodel, cell_chart_coords, cell_to_chart_local
    param_grid        = Gridap.Geometry.get_grid(omodel)
    grid_topology     = Gridap.Geometry.get_grid_topology(omodel)
    face_labeling     = Gridap.Geometry.get_face_labeling(omodel)
    orientation_style = Gridap.Geometry.OrientationStyle(param_grid)

    cell_ambient_maps = lazy_map(Reindex(ambient_maps), cell_to_chart_local)
    cell_metric       = lazy_map(Reindex(metric_fields), cell_to_chart_local)

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

  adaptive_models = map(
    atlas_models,
    local_views(model.atlas_dmodel),
    local_views(adapted_octree_dmodel.dmodel),
  ) do atlas_model, atlas_model_parent, octree_adapted_model
    parent = isa(atlas_model_parent, AdaptedDiscreteModel) ? atlas_model_parent.model : atlas_model_parent
    Gridap.Adaptivity.AdaptedDiscreteModel(
      atlas_model,
      parent,
      get_adaptivity_glue(octree_adapted_model),
    )
  end

  atlas_dmodel = GenericDistributedDiscreteModel(
    adaptive_models, get_cell_gids(adapted_octree_dmodel.dmodel)
  )

  Dc = num_cell_dims(atlas_dmodel)
  Dp = num_point_dims(atlas_dmodel)
  M  = typeof(manifold_style)
  AtlasOctreeDistributedDiscreteModel{Dc,Dp,typeof(adapted_octree_dmodel),typeof(atlas_dmodel),M}(
    adapted_octree_dmodel, atlas_dmodel, info
  ), adaptivity_glue
end

function generate_octree_distributed_refined_models(ranks,
                                               coarse_mesh,
                                               n_ref_lvls,
                                               manifold_style,
                                               coarse_model=false)

  models = Vector{AtlasOctreeDistributedDiscreteModel}(undef,n_ref_lvls)
  cmodel = AtlasOctreeDistributedDiscreteModel(ranks, coarse_mesh, 0; manifold_style=manifold_style)
  model = cmodel
  for n in n_ref_lvls:-1:1
    model, _ = Gridap.Adaptivity.refine(model)
    models[n] = model
  end
  if coarse_model
    push!(models,cmodel)
  end
  models
end 

