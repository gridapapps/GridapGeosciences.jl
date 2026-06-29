# AtlasDiscreteModels.jl
#
# Defines AtlasDiscreteModel{Dc,Da}: wraps an AtlasGrid with topology and face labeling.
# Ambient Da-dimensional coordinates are computed HERE (visualization_data only).

# ============================================================
# AtlasDiscreteModel
# ============================================================

"""
    AtlasDiscreteModel{Dc,Da,G,A,P,O,T,L} <: Gridap.Geometry.DiscreteModel{Dc,Da}

Combines an `AtlasGrid{Dc,Da}` (local Dc-dim geometry) with a `GridTopology{Dc,Dc}`
and a `FaceLabeling`.  Ambient Da-dimensional coordinates are never stored; they are
computed on the fly inside `visualization_data`.

Face labels from the coarse `CoarseMeshInfo` model (e.g. "bottom", "top" for the
cylinder) are propagated to the fine mesh by Gridap's refinement machinery and are
accessible via `Gridap.Geometry.get_face_labeling(model)`.
"""
struct AtlasDiscreteModel{Dc, Da,
                           G, A, P, C, O, M,
                           T <: Gridap.Geometry.GridTopology{Dc,Dc},
                           L <: Gridap.Geometry.FaceLabeling
                           } <: Gridap.Geometry.DiscreteModel{Dc,Da}
  atlas_grid    :: AtlasGrid{Dc,Da,Dc,G,A,P,C,O,M}
  grid_topology :: T
  face_labeling :: L

  function AtlasDiscreteModel(
    atlas_grid    :: AtlasGrid{Dc,Da,Dc,G,A,P,C,O,M},
    grid_topology :: Gridap.Geometry.GridTopology{Dc,Dc},
    face_labeling :: Gridap.Geometry.FaceLabeling,
  ) where {Dc,Da,G,A,P,C,O,M}
    T = typeof(grid_topology)
    L = typeof(face_labeling)
    new{Dc,Da,G,A,P,C,O,M,T,L}(atlas_grid, grid_topology, face_labeling)
  end
end

# ----------------------------------------------------------
# Outer constructor
# ----------------------------------------------------------

"""
    AtlasDiscreteModel(coarse_model, coarse_chart_coords, ambient_maps, num_refinements=1;
                       metric_fields=nothing, orientation_style=nothing,
                       manifold_style=ExtrinsicManifold())

Refine `coarse_model` `num_refinements` times, build an `AtlasGrid` with local coords,
and wrap it with the fine model's topology and face labeling.

When `metric_fields` is not provided it is computed lazily from `ambient_maps` via
`_pullback_metrics(ambient_maps)` for each chart.
"""
function AtlasDiscreteModel(
    coarse_model    :: Gridap.Geometry.DiscreteModel{Dc,Dc},
    coarse_chart_coords,
    ambient_maps,
    num_refinements :: Int;
    metric_fields     = nothing,
    orientation_style = nothing,
    manifold_style    = ExtrinsicManifold(),
) where Dc
  _metric = isnothing(metric_fields) ?
    _pullback_metrics(ambient_maps) : metric_fields
  atlas_grid, fine_model = _build_atlas_grid(
    coarse_model, coarse_chart_coords, ambient_maps, _metric,
    num_refinements, orientation_style, manifold_style)

  AtlasDiscreteModel(
    atlas_grid,
    Gridap.Geometry.get_grid_topology(fine_model),
    Gridap.Geometry.get_face_labeling(fine_model),
  )
end

# ----------------------------------------------------------
# DiscreteModel{Dc,Dc} interface
# ----------------------------------------------------------

Gridap.Geometry.get_grid(m::AtlasDiscreteModel)          = m.atlas_grid
Gridap.Geometry.get_cell_map(m::AtlasDiscreteModel)      = get_cell_map(m.atlas_grid)
Gridap.Geometry.get_cell_map(m::AdaptedDiscreteModel{Dc,Dp,<:AtlasDiscreteModel}) where {Dc,Dp} = get_cell_map(m.model)
Gridap.Geometry.get_grid_topology(m::AtlasDiscreteModel) = m.grid_topology
Gridap.Geometry.get_face_labeling(m::AtlasDiscreteModel) = m.face_labeling
Gridap.Geometry.num_point_dims(m::AtlasDiscreteModel)    = num_point_dims(get_grid(m))
Gridap.Geometry.num_point_dims(m::AdaptedDiscreteModel{Dc,Dp,<:AtlasDiscreteModel}) where {Dc,Dp} = num_point_dims(m.model)


# ----------------------------------------------------------
# Custom API
# ----------------------------------------------------------

get_atlas_grid(m::AtlasDiscreteModel)              = m.atlas_grid
get_ambient_dim(m::AtlasDiscreteModel{Dc,Da}) where {Dc,Da} = Da
get_cell_ambient_maps(m::AtlasDiscreteModel)       = get_cell_ambient_maps(m.atlas_grid)
get_cell_metric(m::AtlasDiscreteModel)             = get_cell_metric(m.atlas_grid)
get_cell_inv_metric(m::AtlasDiscreteModel)         = get_cell_inv_metric(m.atlas_grid)
ManifoldStyle(::Type{<:AtlasDiscreteModel{Dc,Da,G,A,P,C,O,M}}) where {Dc,Da,G,A,P,C,O,M} = M()
ManifoldStyle(m::AtlasDiscreteModel) = ManifoldStyle(typeof(m))

# ============================================================
# Visualization: ambient coords computed here only
# ============================================================

"""
    _local_to_ambient(cell_chart_coords, cell_ambient_maps)

Return a lazy array whose `i`-th entry is the `Da`-dimensional ambient corners of
cell `i`, obtained by applying `cell_ambient_maps[i]` pointwise to
`cell_chart_coords[i]`.

`cell_ambient_maps[i]` is a `Point{Dc} → Point{Da}` map (e.g. `CubedSphereMap`).
`Broadcasting(cell_ambient_maps[i])` lifts it to `Vector{Point} → Vector{Point}`;
`lazy_map(evaluate, …)` chains the two lazy arrays with zero allocation until accessed.
"""
function _local_to_ambient(cell_chart_coords, cell_ambient_maps)
  cell_maps = lazy_map(Broadcasting, cell_ambient_maps)
  lazy_map(evaluate, cell_maps, cell_chart_coords)
end

function Gridap.Visualization.visualization_data(
    model   :: AtlasDiscreteModel{Dc,Da},
    filebase :: AbstractString;
    labels  :: Gridap.Geometry.FaceLabeling = Gridap.Geometry.get_face_labeling(model),
) where {Dc,Da}
  g         = model.atlas_grid
  phys_lazy = _local_to_ambient(g.cell_chart_coords, g.cell_ambient_maps)
  ncells    = Gridap.Geometry.num_cells(g)
  n_corners = length(g.cell_chart_coords[1])
  dg_node_ids = Gridap.Arrays.Table(
    Int32.(1:ncells*n_corners),
    Int32[1 + i*n_corners for i in 0:ncells],
  )
  phys_viz_grid = UnstructuredGrid(
    collect(Iterators.flatten(phys_lazy)),
    dg_node_ids,
    get_reffes(g),
    get_cell_type(g),
    NonOriented(),
  )

  # GridapDistributed expects Dc+1 VisualizationData items (one per dim 0..Dc).
  # For d < Dc build grids from parametric coords (the stored param_grid);
  # for d == Dc use the ambient-coordinate DG grid built above.
  param_model = Gridap.Geometry.UnstructuredDiscreteModel(
    Gridap.Geometry.UnstructuredGrid(g.param_grid),
    Gridap.Geometry.UnstructuredGridTopology(model.grid_topology),
    labels,
  )
  map(0:Dc) do d
    if d < Dc
      sub_grid = Gridap.Geometry.Grid(Gridap.ReferenceFEs.ReferenceFE{d}, param_model)
      cdata    = Gridap.Visualization._prepare_cdata(labels, d)
      Gridap.Visualization.VisualizationData(sub_grid, "$(filebase)_$(d)"; celldata=cdata)
    else
      cdata = Gridap.Visualization._prepare_cdata(labels, d)
      Gridap.Visualization.VisualizationData(phys_viz_grid, "$(filebase)_$(Dc)"; celldata=cdata)
    end
  end
end

function Gridap.Visualization.visualization_data(
    model    :: AdaptedDiscreteModel{Dc,Da,<:AtlasDiscreteModel},
    filebase :: AbstractString;
    labels   :: Gridap.Geometry.FaceLabeling = Gridap.Geometry.get_face_labeling(model.model),
) where {Dc,Da}
  Gridap.Visualization.visualization_data(model.model, filebase; labels=labels)
end

# ============================================================
# Convenience constructors via CoarseMeshInfo / CoarseMesh
# ============================================================

# ----------------------------------------------------------
# AtlasDiscreteModel
# ----------------------------------------------------------

"""
    AtlasDiscreteModel(info::CoarseMeshInfo, num_refinements;
                       orientation_style=nothing, manifold_style=ExtrinsicManifold())

Build an `AtlasDiscreteModel` from a `CoarseMeshInfo`, using the analytic metric fields
stored in `info.metric_fields`.
"""
function AtlasDiscreteModel(
    info            :: CoarseMeshInfo,
    num_refinements :: Int;
    orientation_style = nothing,
    manifold_style    = ExtrinsicManifold(),
)
  AtlasDiscreteModel(info.model, info.cell_chart_coords, info.ambient_maps, num_refinements;
                     metric_fields=info.metric_fields, orientation_style, manifold_style)
end

"""
    AtlasDiscreteModel(info::CoarseMeshInfo, ambient_maps, num_refinements;
                       orientation_style=nothing, manifold_style=ExtrinsicManifold())

Build an `AtlasDiscreteModel` from a `CoarseMeshInfo`, overriding the default ambient maps.
Metric fields are recomputed from `ambient_maps` via `JtJ`.
"""
function AtlasDiscreteModel(
    info            :: CoarseMeshInfo,
    ambient_maps,
    num_refinements :: Int;
    orientation_style = nothing,
    manifold_style    = ExtrinsicManifold(),
)
  custom_metrics = _pullback_metrics(ambient_maps)
  AtlasDiscreteModel(info.model, info.cell_chart_coords, ambient_maps, num_refinements;
                     metric_fields=custom_metrics, orientation_style, manifold_style)
end

"""
    AtlasDiscreteModel(mesh::CoarseMesh, num_refinements;
                       orientation_style=nothing, manifold_style=ExtrinsicManifold())

Build an `AtlasDiscreteModel` directly from a mesh descriptor (e.g. `CubedSphereMesh(1.0)`).
"""
function AtlasDiscreteModel(
    mesh            :: CoarseMesh,
    num_refinements :: Int;
    orientation_style = nothing,
    manifold_style    = ExtrinsicManifold(),
)
  AtlasDiscreteModel(get_coarse_mesh(mesh), num_refinements; orientation_style, manifold_style)
end

const IntrinsicAtlasDiscreteModel{Dc,Da,G,A,P,C,O} = AtlasDiscreteModel{Dc,Da,G,A,P,C,O,<:IntrinsicManifold}
const ExtrinsicAtlasDiscreteModel{Dc,Da,G,A,P,C,O} = AtlasDiscreteModel{Dc,Da,G,A,P,C,O,<:ExtrinsicManifold}

const BFTATDM{Dc,Dp} = Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel}
const BFTATDMIM{Dct,Dcm,Da,G,A,P,C,O} =
    Gridap.Geometry.BodyFittedTriangulation{Dct,Da,<:IntrinsicAtlasDiscreteModel{Dcm,Da,G,A,P,C,O}}
const BFTATDMEM{Dct,Dcm,Da,G,A,P,C,O} =
    Gridap.Geometry.BodyFittedTriangulation{Dct,Da,<:ExtrinsicAtlasDiscreteModel{Dcm,Da,G,A,P,C,O}}


function get_radius(model::AtlasDiscreteModel{Dc,Dp, G, A, <:AbstractVector{<:CubedSphereMap}}) where {Dc,Dp,G,A}
   model.atlas_grid.cell_ambient_maps.values[1].radius
end

function get_radius(model::AdaptedDiscreteModel{Dc,Dp,<:AtlasDiscreteModel}) where {Dc,Dp}
  get_radius(model.model)
end

function get_thickness(model::AtlasDiscreteModel{Dc,Dp, G, A, <:AbstractVector{<:CubedSphereWithThicknessMap}}) where {Dc,Dp,G,A}
   model.atlas_grid.cell_ambient_maps.values[1].thickness
end

function get_thickness(model::AdaptedDiscreteModel{Dc,Dp,<:AtlasDiscreteModel}) where {Dc,Dp}
  get_thickness(model.model)
end


################################################################################
########## 3D ########
################################################################################
# unit normal
normal_vec(XYZ) = 1.0/sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2] + XYZ[3]*XYZ[3])*VectorValue(XYZ[1],XYZ[2],XYZ[3])

# tangent component of aribitary 3D vector vecX
tangent_vec(vecX::Function) = XYZ -> vecX(XYZ) - (vecX(XYZ)⋅normal_vec(XYZ))⋅normal_vec(XYZ)

function get_surface_normal(trian::BFTATDM{Dc,3}) where {Dc}
  ns = CellField(normal_vec,trian)
  ## This cellfield is, by default, on the physical domain
  ## Change to the reference domain. Recall the ambient model has junk nodes
  ## So being on the reference domain means the evaluatation at pts is via ref points
  change_domain(ns,DomainStyle(ns),ReferenceDomain())
end

function generate_refined_models(n_ref_lvls,
                            coarse_mesh,
                            manifold_style,
                            coarse_model=false)
  model = AtlasDiscreteModel(coarse_mesh, 0; manifold_style=manifold_style)
  models = Vector{DiscreteModel}(undef,n_ref_lvls)
  for n in n_ref_lvls:-1:1
    model = Gridap.Adaptivity.refine(model)
    models[n] = model
  end
  if coarse_model
    push!(models,AtlasDiscreteModel(coarse_mesh,0; manifold_style=manifold_style))
  end
  models
end

function get_cubed_sphere_refined_models(n_ref_lvls::Int,
                                         radius::Real,
                                         manifold_style,
                                         coarse_model=false)
  coarse_mesh = CubedSphereMesh(radius)
  generate_refined_models(n_ref_lvls, coarse_mesh, manifold_style, coarse_model)
end




"""
    Gridap.Adaptivity.refine(model::AtlasDiscreteModel) -> AdaptedDiscreteModel

Uniformly refine `model` once and return an `AdaptedDiscreteModel` wrapping the
refined `AtlasDiscreteModel` together with its parent and the refinement glue.

The refined atlas grid inherits ambient maps and metric from each parent cell.
New per-cell chart coordinates are obtained by composing the parent-cell's
ref-to-chart map with the sub-cell reference positions from the refinement rule,
exactly as in `_build_atlas_grid`.
"""
function Gridap.Adaptivity.refine(model::AtlasDiscreteModel{Dc}, args...; kwargs...) where Dc
  g = model.atlas_grid
  n_old_cells = Gridap.Geometry.num_cells(g)

  # Build an UnstructuredDiscreteModel from the current atlas components so that
  # Gridap's refinement machinery (uniformly_refine + blocked_refinement_glue)
  # operates on plain connectivity — same pattern as visualization_data.
  fine_unstr = Gridap.Geometry.UnstructuredDiscreteModel(
    Gridap.Geometry.UnstructuredGrid(g.param_grid),
    Gridap.Geometry.UnstructuredGridTopology(model.grid_topology),
    model.face_labeling,
  )

  # Refine the underlying model once (n_factor = 2) to get glue + refined topo/labeling
  adapted          = Gridap.Adaptivity.refine(fine_unstr, 2)
  glue             = adapted.glue
  ref_model        = adapted.model
  cell_to_old_cell = Vector{Int}(glue.n2o_faces_map[Dc + 1])

  # Sub-cell reference coordinates from the (uniform) refinement rule
  rrule      = glue.refinement_rules[1]
  ref_cgrid  = Gridap.Adaptivity.get_ref_grid(rrule)
  ref_coords = Gridap.Geometry.get_cell_coordinates(Gridap.Geometry.get_grid(ref_cgrid))
  n_per_cell = Gridap.Geometry.num_cells(Gridap.Geometry.get_grid(ref_cgrid))

  # Per-old-cell maps: ref-element → chart coordinates
  reffe     = Gridap.Geometry.get_reffes(g)[1]
  shapefuns = Gridap.ReferenceFEs.get_shapefuns(reffe)
  Ψ_maps    = map(corners -> linear_combination(corners, shapefuns), g.cell_chart_coords)

  # Tile ref_coords and apply the parent-cell Ψ map lazily.
  # blocked_refinement_glue guarantees parent-major ordering, so child_ids
  # cycles 1..n_per_cell once for every parent cell.
  child_ids             = repeat(1:n_per_cell, n_old_cells)
  ref_per_new_cell      = lazy_map(Reindex(ref_coords), child_ids)
  cell_Ψ                = lazy_map(Reindex(Ψ_maps), cell_to_old_cell)
  new_cell_chart_coords = lazy_map(evaluate, cell_Ψ, ref_per_new_cell)

  # Inherit ambient maps and metric from parent cells
  new_cell_ambient_maps = lazy_map(Reindex(g.cell_ambient_maps), cell_to_old_cell)
  new_cell_metric       = lazy_map(Reindex(g.cell_metric),       cell_to_old_cell)

  new_atlas_grid = AtlasGrid(
    Gridap.Geometry.get_grid(ref_model),
    new_cell_chart_coords,
    new_cell_ambient_maps,
    new_cell_metric,
    Gridap.Geometry.OrientationStyle(g),
    ManifoldStyle(g),
  )

  ref_atlas_model = AtlasDiscreteModel(
    new_atlas_grid,
    Gridap.Geometry.get_grid_topology(ref_model),
    Gridap.Geometry.get_face_labeling(ref_model),
  )

  AdaptedDiscreteModel(ref_atlas_model, model, glue)
end

function Gridap.Geometry.Grid(::Type{ReferenceFE{Dcg}},
                              model::AtlasDiscreteModel{Dcm},
                              face_to_bgface,
                              face_to_lcell) where {Dcg,Dcm}
   @check Dcg < Dcm

   cell_grid = get_grid(model)
   cell_param_grid = cell_grid.param_grid

   face_param_grid_node_coordinates = collect1d(get_node_coordinates(cell_param_grid))
   face_param_grid_cell_to_nodes = Table(get_face_nodes(model,Dcg))
   face_param_reffes = get_reffaces(ReferenceFE{Dcg},model)
   face_param_grid_cell_to_type = collect1d(get_face_type(model,Dcg))

   face_param_grid =  view(UnstructuredGrid(face_param_grid_node_coordinates,
                                      face_param_grid_cell_to_nodes,
                                      face_param_reffes,
                                      face_param_grid_cell_to_type,
                                      Gridap.Geometry.OrientationStyle(cell_param_grid)),
                            face_to_bgface)

    topology = get_grid_topology(model)
    nfaces = num_faces(topology,Dcg)
    glue = FaceToCellGlue(
               topology,
               cell_param_grid,
               face_param_grid,
               face_to_bgface,
               face_to_lcell)

   face_to_cell_reference_map =
      Gridap.Geometry.compute_face_to_cell_reference_map(cell_grid,face_param_grid,glue)
   face_to_cell_ambient_maps =
        lazy_map(Reindex(get_cell_ambient_maps(model)), glue.face_to_cell)

   face_to_q_vertex_coords =
      Fill(get_vertex_coordinates(get_polytope(face_param_reffes[1])), nfaces)

   chart_maps = lazy_map(Reindex(_chart_maps(cell_grid)), glue.face_to_cell)
   face_chart_maps = lazy_map(∘, chart_maps, face_to_cell_reference_map)
   face_chart_coords = lazy_map(evaluate, face_chart_maps, face_to_q_vertex_coords)

   face_to_cell_metric =
      lazy_map(Reindex(get_cell_metric(model)), glue.face_to_cell)

   AtlasGrid(
     face_param_grid,
     face_chart_coords,
     face_to_cell_ambient_maps,
     face_to_cell_metric,
     Gridap.Geometry.OrientationStyle(cell_param_grid),
     ManifoldStyle(model),
   )
end
