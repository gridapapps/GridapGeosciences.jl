
struct ExtrudedAtlasOctreeDistributedDiscreteModel{
  A <: OctreeDistributedDiscreteModel{3,3},
  B <: GenericDistributedDiscreteModel{3,3},
  M <: ManifoldStyle,
} <: GridapDistributed.DistributedDiscreteModel{3,3}
  octree_dmodel :: A
  atlas_dmodel  :: B
  coarse_info   :: CoarseMeshInfo{2}
end

# ----------------------------------------------------------
# DistributedDiscreteModel interface (delegate to dmodel)
# ----------------------------------------------------------

GridapDistributed.local_views(m::ExtrudedAtlasOctreeDistributedDiscreteModel) =
  local_views(m.atlas_dmodel)

GridapDistributed.get_cell_gids(m::ExtrudedAtlasOctreeDistributedDiscreteModel) =
  get_cell_gids(m.atlas_dmodel)

# ----------------------------------------------------------
# Custom API
# ----------------------------------------------------------

get_octree_dmodel(m::ExtrudedAtlasOctreeDistributedDiscreteModel) = m.octree_dmodel
get_atlas_model(m::ExtrudedAtlasOctreeDistributedDiscreteModel)  = m.atlas_dmodel
ManifoldStyle(::Type{<:ExtrudedAtlasOctreeDistributedDiscreteModel{A,B,M}}) where {A,B,M} = M()
ManifoldStyle(m::ExtrudedAtlasOctreeDistributedDiscreteModel) = ManifoldStyle(typeof(m))

get_cell_metric(m::ExtrudedAtlasOctreeDistributedDiscreteModel) =
  map(get_cell_metric, local_views(m.atlas_dmodel))

# ============================================================
# p6est helper functions (also used by CubedSphere3DParametricOctreeDistributedDiscreteModel)
# ============================================================

function _dummy_grid_and_topology_function(pXest_type::GridapP4est.P6estType,
                                          non_conforming_glue,
                                          cell_vertices,
                                          ptr_pXest_connectivity,
                                          ptr_pXest,
                                          ptr_pXest_ghost)
  function JaggedToTable(x::MPIArray{<:JaggedArray})
      map(x) do x
        Gridap.Arrays.Table(x.data,x.ptrs)
      end
  end
  grid,topology=_generate_topology_grid_and_topology(pXest_type,JaggedToTable(cell_vertices))
end

function generate_cell_alpha_beta_gamma_coordinates_and_panels(parts,
                                   coarse_discrete_model,
                                   coarse_coarse_cell_wise_vertex_alpha_beta_coordinates,
                                   coarse_cell_panel,
                                   ptr_pXest_connectivity,
                                   ptr_pXest,
                                   ptr_pXest_ghost)

  Dc=3
  PXEST_CORNERS=2^Dc
  pXest_ghost = ptr_pXest_ghost[]
  pXest = ptr_pXest[]
  pXest_type = GridapP4est.P6estType()

  dcell_coordinates_and_panels=map(parts) do part
     panels = Int[]
     data = Point{Dc,Float64}[]
     ncells = 0
     vxy=Vector{Cdouble}(undef,Dc)
     pvxy=pointer(vxy,1)
     for itree=1:pXest_ghost.num_trees
       tree = GridapP4est.pXest_tree_array_index(pXest_type, pXest, itree-1)[]
       if tree.quadrants.elem_count > 0
          set_coarse_cell_vertices_coordinates!( ptr_pXest_connectivity[].conn4,
                                                 coarse_discrete_model,
                                                 itree,
                                                 coarse_coarse_cell_wise_vertex_alpha_beta_coordinates[itree])
       end
       for cell=1:tree.quadrants.elem_count
          quadrant=GridapP4est.pXest_quadrant_array_index(pXest_type,tree,cell-1)[]

          # Loop over layers in the current column
          for l=1:GridapP4est.pXest_num_quadrant_layers(pXest_type,quadrant)
            layer=GridapP4est.pXest_get_layer(pXest_type, quadrant, pXest, l-1)
            coords=GridapP4est.pXest_cell_coords(pXest_type,quadrant,layer)
            levels=GridapP4est.pXest_get_quadrant_and_layer_levels(pXest_type,quadrant,layer)
            push!(panels, coarse_cell_panel[itree])
            for vertex=1:PXEST_CORNERS
              GridapP4est.pXest_get_quadrant_vertex_coordinates(pXest_type,
                                                               ptr_pXest_connectivity,
                                                               p4est_topidx_t(itree-1),
                                                               coords,
                                                               levels,
                                                               Cint(vertex-1),
                                                               pvxy)
              push!(data, Point{Dc,Float64}(vxy[3],vxy[1],vxy[2]))
            end
            ncells = ncells+1
          end
       end
     end

     function sc_array_p4est_locidx_t_index(sc_array_object::sc_array_t, it)
      @assert sc_array_object.elem_size == sizeof(p4est_locidx_t)
      @assert it in 0:sc_array_object.elem_count
      ptr=Ptr{p4est_locidx_t}(sc_array_object.array + sc_array_object.elem_size*it)
      return unsafe_wrap(Array, ptr, 1)[]
     end

     column_ghost = pXest_ghost.column_ghost[]
     ptr_p2est_ghost_quadrants = GridapP4est._unwrap_ghost_quadrants(pXest_type, pXest_ghost)
     ptr_p4est_ghost_quadrants = GridapP4est._unwrap_ghost_quadrants(GridapP4est.P4estType(), column_ghost)

     tree_offsets = unsafe_wrap(Array, column_ghost.tree_offsets, pXest_ghost.num_trees+1)

     current_ghost_column=0

     # Go over ghost cells
     for i=1:pXest_ghost.num_trees
       if tree_offsets[i+1]-tree_offsets[i] > 0
          set_coarse_cell_vertices_coordinates!( ptr_pXest_connectivity[].conn4,
                                                 coarse_discrete_model,
                                                 i,
                                                 coarse_coarse_cell_wise_vertex_alpha_beta_coordinates[i])
       end

       for j=tree_offsets[i]:tree_offsets[i+1]-1
          p4est_quadrant = ptr_p4est_ghost_quadrants[j+1]
          k = sc_array_p4est_locidx_t_index(pXest_ghost.column_layer_offsets[],current_ghost_column)
          l = sc_array_p4est_locidx_t_index(pXest_ghost.column_layer_offsets[],current_ghost_column+1)
          for m=k:l-1
            p2est_quadrant = ptr_p2est_ghost_quadrants[m+1]
            coords=GridapP4est.pXest_cell_coords(pXest_type,p4est_quadrant,p2est_quadrant)
            levels=GridapP4est.pXest_get_quadrant_and_layer_levels(pXest_type,p4est_quadrant,p2est_quadrant)
            push!(panels, coarse_cell_panel[i])
            for vertex=1:PXEST_CORNERS
                GridapP4est.pXest_get_quadrant_vertex_coordinates(pXest_type,
                                                      ptr_pXest_connectivity,
                                                      p4est_topidx_t(i-1),
                                                      coords,
                                                      levels,
                                                      Cint(vertex-1),
                                                      pvxy)
                push!(data, Point{Dc,Float64}(vxy[3],vxy[1],vxy[2]))
            end
            ncells=ncells+1
          end
          current_ghost_column=current_ghost_column+1
       end
    end
    ptr=generate_ptr(Dc,ncells)
    Gridap.Arrays.Table(data,ptr), panels
  end |> tuple_of_arrays
end

# ============================================================
# Internal p6est setup helper
# ============================================================

"""
    _generate_extruded_octree_cell_chart_coords_and_chart_id(
        ranks, coarse_model, coarse_cell_chart_coords, coarse_cell_panels,
        num_horizontal_refinements, num_vertical_refinements)

Build a p6est-based `OctreeDistributedDiscreteModel{3,3}` from a 2D coarse model and
return:
- the octree distributed discrete model
- per-rank `Table{Point{3,Float64}}` of 3D chart coordinates `(γ,α,β)` per fine cell
  (owned + ghost), with γ ∈ [0,1] the normalised vertical coordinate and (α,β) the
  horizontal chart coordinates interpolated from `coarse_cell_chart_coords`
- per-rank `Vector{Int}` mapping each fine cell to the coarse panel/chart index

The 2D coarse chart coordinate corners in `coarse_cell_chart_coords[k]` (one
`Vector{Point{2}}` per coarse cell k) are embedded into the p4est connectivity so that
`generate_cell_alpha_beta_gamma_coordinates_and_panels` can perform bilinear interpolation
without requiring cubed-sphere-specific logic.
"""
function _generate_extruded_octree_cell_chart_coords_and_chart_id(
    ranks,
    coarse_model  :: DiscreteModel{2,2},
    coarse_cell_chart_coords,
    coarse_cell_panels,
    num_horizontal_refinements,
    num_vertical_refinements,
)
  comm = ranks.comm
  Dc   = 3
  pXest_type                 = GridapP4est.P6estType()
  pXest_refinement_rule_type = GridapP4est.PXestHorizontalRefinementRuleType()

  extrusion_vector = Vector{Float64}([0.0, 0.0, 1.0])

  ptr_pXest_connectivity = GridapP4est.setup_pXest_connectivity(pXest_type,
                                                               coarse_model,
                                                               extrusion_vector)

  ptr_pXest = P4est_wrapper.p6est_new_ext(comm,
                 ptr_pXest_connectivity,
                 Cint(0),
                 Cint(num_horizontal_refinements), # min_level  (horizontal)
                 Cint(num_vertical_refinements),   # min_zlevel (vertical)
                 Cint(1),                          # num_zroot
                 Cint(1),                          # fill_uniform
                 Cint(1),                          # data_size
                 C_NULL,                           # init_fn
                 C_NULL)                           # user_pointer

  ptr_pXest_ghost  = GridapP4est.setup_pXest_ghost(pXest_type, ptr_pXest)
  ptr_pXest_lnodes = GridapP4est.setup_pXest_lnodes_nonconforming(pXest_type,
                                                                   ptr_pXest,
                                                                   ptr_pXest_ghost)

  dmodel, non_conforming_glue = GridapP4est.setup_non_conforming_distributed_discrete_model(
    pXest_type,
    pXest_refinement_rule_type,
    ranks,
    coarse_model,
    ptr_pXest_connectivity,
    ptr_pXest,
    ptr_pXest_ghost,
    ptr_pXest_lnodes;
    grid_and_topology_function=_dummy_grid_and_topology_function,
    grid_and_topology_bottom_function=_dummy_grid_and_topology_function,
  )

  # Generate 3D chart coords (γ,α,β) and chart (panel) IDs per fine cell.
  # `generate_cell_alpha_beta_gamma_coordinates_and_panels` encodes the 2D
  # coarse_cell_chart_coords into the p4est connectivity vertex positions and then
  # uses bilinear interpolation (horizontal) + p6est layer coordinates (vertical γ)
  # to compute the 3D reference-frame corners of every fine hexahedral cell.
  cell_chart_coords_3d, cell_to_chart_id =
    generate_cell_alpha_beta_gamma_coordinates_and_panels(
      ranks,
      coarse_model,
      coarse_cell_chart_coords,
      coarse_cell_panels,
      ptr_pXest_connectivity,
      ptr_pXest,
      ptr_pXest_ghost,
    )

  GridapP4est.pXest_lnodes_destroy(pXest_type, ptr_pXest_lnodes)
  GridapP4est.pXest_ghost_destroy(pXest_type, ptr_pXest_ghost)

  omodel = OctreeDistributedDiscreteModel(
    Dc, Dc,
    ranks,
    dmodel,
    non_conforming_glue,
    coarse_model,
    ptr_pXest_connectivity,
    ptr_pXest,
    pXest_type,
    pXest_refinement_rule_type,
    true,
    nothing,
  )

  omodel, cell_chart_coords_3d, cell_to_chart_id
end

# ============================================================
# Constructors
# ============================================================

"""
    ExtrudedAtlasOctreeDistributedDiscreteModel(
        ranks, info::CoarseMeshInfo{2},
        num_vertical_refinements, num_horizontal_refinements;
        manifold_style=ExtrinsicManifold())

Build a distributed 3D `ExtrudedAtlasOctreeDistributedDiscreteModel` from a 2D
`CoarseMeshInfo`.

The 2D coarse `DiscreteModel` in `info.model` defines the horizontal topology and is
handed to p6est for forest construction; the extrusion direction is `[0,0,1]` (unit
interval in γ).  Per-fine-cell 3D chart coordinates `(γ,α,β)` are computed directly
from the p6est structure without modifying the permanent connectivity: the horizontal
(α,β) part comes from bilinear interpolation of `info.cell_chart_coords`, and the
vertical γ part is extracted from the p6est layer coordinates.

`info.ambient_maps` must accept 3D chart-space `Point{3}` inputs (e.g.
`CubedSphereWithThicknessMap`). `info.metric_fields` must likewise be 3D fields.

Use `ExtrudedCubedSphereWithThicknessMesh(radius, thickness)` or similar extruded
`CoarseMesh` subtypes to build a suitable `CoarseMeshInfo{2}` via `get_coarse_mesh`.
"""
function ExtrudedAtlasOctreeDistributedDiscreteModel(
    ranks,
    info  :: CoarseMeshInfo{2},
    num_vertical_refinements,
    num_horizontal_refinements;
    manifold_style = ExtrinsicManifold(),
)
  ambient_maps             = info.ambient_maps
  metric_fields            = info.metric_fields
  coarse_model             = info.model
  coarse_cell_panels       = collect(1:Gridap.Geometry.num_cells(coarse_model))
  coarse_cell_chart_coords = info.cell_chart_coords

  octree_dmodel, cell_wise_chart_coords_3d, cell_to_chart_id =
    _generate_extruded_octree_cell_chart_coords_and_chart_id(
      ranks,
      coarse_model,
      coarse_cell_chart_coords,
      coarse_cell_panels,
      num_horizontal_refinements,
      num_vertical_refinements,
    )

  atlas_models = map(
    local_views(octree_dmodel.dmodel),
    cell_wise_chart_coords_3d,
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

  atlas_dmodel = GenericDistributedDiscreteModel(
    atlas_models, get_cell_gids(octree_dmodel.dmodel))

  M = typeof(manifold_style)
  ExtrudedAtlasOctreeDistributedDiscreteModel{
    typeof(octree_dmodel), typeof(atlas_dmodel), M}(
    octree_dmodel, atlas_dmodel, info)
end

"""
    ExtrudedAtlasOctreeDistributedDiscreteModel(
        ranks, mesh::CoarseMesh, num_initial_uniform_refinements;
        manifold_style=ExtrinsicManifold())

Convenience constructor: calls `get_coarse_mesh(mesh)` and delegates to the
`CoarseMeshInfo{2}` constructor, applying `num_initial_uniform_refinements` to
both the horizontal and vertical directions.

Example:
```julia
model = ExtrudedAtlasOctreeDistributedDiscreteModel(
    ranks,
    ExtrudedCubedSphereWithThicknessMesh(1.0, 0.1),
    2,          # num_horizontal_refinements = num_vertical_refinements = 2
)
```
"""
function ExtrudedAtlasOctreeDistributedDiscreteModel(
    ranks,
    mesh  :: CoarseMesh,
    num_vertical_refinements,
    num_horizontal_refinements;
    manifold_style = ExtrinsicManifold(),
)
  ExtrudedAtlasOctreeDistributedDiscreteModel(
    ranks,
    get_coarse_mesh(mesh),
    num_vertical_refinements,
    num_horizontal_refinements;
    manifold_style,
  )
end
