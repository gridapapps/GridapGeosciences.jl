# AtlasDiscreteModels.jl
#
# Defines AtlasDiscreteModel{Dc,Da}: wraps an AtlasGrid with topology and face labeling.
# Ambient Da-dimensional coordinates are computed HERE (visualization_data only).

include("AtlasGrids.jl")   # also includes CoarseMeshes.jl

# ============================================================
# AtlasDiscreteModel
# ============================================================

"""
    AtlasDiscreteModel{Dc,Da,G,A,P,O,T,L} <: Gridap.Geometry.DiscreteModel{Dc,Dc}

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
                           } <: Gridap.Geometry.DiscreteModel{Dc,Dc}
  atlas_grid    :: AtlasGrid{Dc,Da,G,A,P,C,O,M}
  grid_topology :: T
  face_labeling :: L

  function AtlasDiscreteModel(
    atlas_grid    :: AtlasGrid{Dc,Da,G,A,P,C,O,M},
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
    num_refinements :: Int = 1;
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
Gridap.Geometry.get_grid_topology(m::AtlasDiscreteModel) = m.grid_topology
Gridap.Geometry.get_face_labeling(m::AtlasDiscreteModel) = m.face_labeling

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

# ============================================================
# Convenience constructors via CoarseMeshInfo / CoarseMesh
# ============================================================

# AtlasGrid equivalents live in AtlasGrids.jl (after include("CoarseMeshes.jl")).

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

# ============================================================
# ParametricCellField for AtlasDiscreteModel triangulations
# ============================================================
#
# Mirrors the existing CubedSphereParametricDiscreteModel pattern.
# f(forward_map) takes the per-cell ambient map (e.g. CylinderChartMap or CubedSphereMap) and
# returns a function αβ::Point{Dc} → T evaluated in chart-local coordinates.
# forward_map(αβ) calls the Field at αβ (via Gridap's callable-Field interface)
# and returns a Da-dimensional ambient point.
#
# Usage (e.g. for a scalar field on the cylinder):
#
#   function u_exact_factory(fwd)
#     αβ -> begin x = fwd(αβ); cos(x[3]) + x[1] end
#   end
#   u_cf = ParametricCellField(u_exact_factory, Ω)

import GridapGeosciences: ParametricCellField

function ParametricCellField(
    f     :: Function,
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel},
) where {Dc,Dp}
  model      = Gridap.Geometry.get_background_model(trian)
  cell_amaps = get_cell_ambient_maps(model)
  cell_field = lazy_map(m -> GenericField(f(m)), cell_amaps)
  Gridap.CellData.GenericCellField(cell_field, trian, Gridap.CellData.PhysicalDomain())
end

"""
    MetricCellField(trian)

Return the per-cell pullback metric `g` stored in the underlying `AtlasGrid` as a
`CellField` on `trian`.  Each cell's metric is a `SymTensorValue{Dc,Dc}` field
evaluated in chart coordinates.

Use `InvMetricCellField(Ω)` for `g⁻¹` (preferred over `Operation(inv)(MetricCellField(Ω))`
for built-in shapes — uses the explicit analytic formula).
Use `Operation(x -> sqrt(det(x)))(MetricCellField(Ω))` for `√det g`.
This is the correct intrinsic source for the metric — it uses the analytic metric
stored in `CoarseMeshInfo.metric_fields`, independent of any ambient embedding.
"""
function MetricCellField(
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_metric(model), trian, Gridap.CellData.PhysicalDomain())
end

"""
    InvMetricCellField(trian)

Return the per-cell inverse pullback metric `g⁻¹` as a `CellField` on `trian`.
For built-in shapes (`CylinderMesh`, `MobiusStripMesh`, `CubedSphereMesh`) this
uses an explicit analytic formula via `inverse_metric_field`; the generic fallback
applies `Operation(inv)` at each quadrature point.
"""
function InvMetricCellField(
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_inv_metric(model), trian, Gridap.CellData.PhysicalDomain())
end
