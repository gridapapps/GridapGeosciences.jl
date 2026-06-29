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
    trian :: BFTATDM{Dc,Dp},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_metric(model),
                                   trian,
                                   Gridap.CellData.PhysicalDomain())
end

function MetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = MetricCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function MetricCellField(
    trian ::  Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  Gridap.CellData.GenericCellField(get_cell_metric(trian.trian.grid),
                                   trian,
                                   Gridap.CellData.PhysicalDomain())
end

function MetricCellField(
    trian :: SkeletonTriangulation{Dc,Dp,<:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  m_cf_plus_bt = MetricCellField(trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = MetricCellField(trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function MetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:SkeletonTriangulation{Dc,Dp,
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}},
) where {Dc,Dp}
  m_cf_plus_bt = MetricCellField(trian.trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = MetricCellField(trian.trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function MetricCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = MetricCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function MetricCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  cf = MetricCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function MeasureCellField(trian :: BFTATDM{Dc,Dp}) where {Dc,Dp}
    sqrt∘det∘MetricCellField(trian)
end

function MeasureCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = MeasureCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function MeasureCellField(
    trian ::  Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  sqrt∘det∘MetricCellField(trian)
end

function MeasureCellField(
    trian :: SkeletonTriangulation{Dc,Dp,<:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  m_cf_plus_bt = MeasureCellField(trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = MeasureCellField(trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function MeasureCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:SkeletonTriangulation{Dc,Dp,
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}},
) where {Dc,Dp}
  m_cf_plus_bt = MeasureCellField(trian.trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = MeasureCellField(trian.trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function MeasureCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  sqrt∘det∘MetricCellField(trian)
end

function MeasureCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  sqrt∘det∘MetricCellField(trian)
end

"""
    InvMetricCellField(trian)

Return the per-cell inverse pullback metric `g⁻¹` as a `CellField` on `trian`.
For built-in shapes (`CylinderMesh`, `MobiusStripMesh`, `CubedSphereMesh`) this
uses an explicit analytic formula via `inverse_metric_field`; the generic fallback
applies `Operation(inv)` at each quadrature point.
"""
function InvMetricCellField(
    trian :: BFTATDM{Dc,Dp},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_inv_metric(model), trian, Gridap.CellData.PhysicalDomain())
end

function InvMetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = InvMetricCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function InvMetricCellField(
    trian ::  Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  Gridap.CellData.GenericCellField(get_cell_inv_metric(trian.trian.grid),
                                   trian,
                                   Gridap.CellData.PhysicalDomain())
end

function InvMetricCellField(
    trian :: SkeletonTriangulation{Dc,Dp,<:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  m_cf_plus_bt = InvMetricCellField(trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = InvMetricCellField(trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function InvMetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:SkeletonTriangulation{Dc,Dp,
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}},
) where {Dc,Dp}
  m_cf_plus_bt = InvMetricCellField(trian.trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = InvMetricCellField(trian.trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function InvMetricCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = InvMetricCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function InvMetricCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  cf = InvMetricCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function AmbientMapCellField(
    trian :: BFTATDM{Dc,Dp},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_ambient_maps(model), trian, Gridap.CellData.PhysicalDomain())
end

function AmbientMapCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = AmbientMapCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function AmbientMapCellField(
    trian :: Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  Gridap.CellData.GenericCellField(get_cell_ambient_maps(trian.trian.grid),
                                   trian,
                                   Gridap.CellData.PhysicalDomain())
end

function AmbientMapCellField(
    trian :: SkeletonTriangulation{Dc,Dp,<:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  m_cf_plus_bt = AmbientMapCellField(trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = AmbientMapCellField(trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function AmbientMapCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:SkeletonTriangulation{Dc,Dp,
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}},
) where {Dc,Dp}
  m_cf_plus_bt = AmbientMapCellField(trian.trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = AmbientMapCellField(trian.trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function AmbientMapCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = AmbientMapCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function AmbientMapCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  cf = AmbientMapCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

# Only supported for the cubed sphere mesh
# For visualisation purposes
function LatLonMapCellField(trian::Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},
                                        C,
                                        O,
                                        M}}) where {Dc,Da,G,A,C,O,M}
  Operation(Cartesian2SphericalMap())(AmbientMapCellField(trian))
end

function LatLonMapCellField(trian::AdaptedTriangulation{Dc,Da,<:Gridap.Geometry.BodyFittedTriangulation{Dc,Da,
                                        <:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},
                                        C,
                                        O,
                                        M}}}) where {Dc,Da,G,A,C,O,M}
  cf = LatLonMapCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

# Right now only supported by AtlasDiscreteModel of the sphere
function InvAmbientMapCellField(
    trian :: AdaptedTriangulation{Dc,Da,<:Gridap.Geometry.BodyFittedTriangulation{Dc,Da,
                                        <:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:CubedSphereMap},
                                        C,
                                        O,
                                        <:ExtrinsicManifold}}},
) where {Dc,Da,G,A,C,O}
  cf = InvAmbientMapCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function InvAmbientMapCellField(
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:CubedSphereMap},
                                        C,
                                        O,
                                        <:ExtrinsicManifold}}) where {Dc,Da,G,A,C,O}
  model = Gridap.Geometry.get_background_model(trian)
  cell_ambient_maps = get_cell_ambient_maps(model)
  ptrs = cell_ambient_maps.ptrs
  ambient_maps = cell_ambient_maps.values
  radius = ambient_maps[1].radius
  inv_ambient_maps = [CubedSphereInvMap(panel, radius) for panel in 1:length(ambient_maps)]
  cell_inv_ambient_maps = CompressedArray(inv_ambient_maps, ptrs)
  Gridap.CellData.GenericCellField(cell_inv_ambient_maps, trian, Gridap.CellData.PhysicalDomain())
end

# Right now only supported by AtlasDiscreteModel of the sphere
function InvAmbientMapCellField(
    trian :: AdaptedTriangulation{Dc,Da,<:Gridap.Geometry.BodyFittedTriangulation{Dc,Da,
                                        <:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:CubedSphereWithThicknessMap},
                                        C,
                                        O,
                                        <:ExtrinsicManifold}}},
) where {Dc,Da,G,A,C,O}
  cf = InvAmbientMapCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function InvAmbientMapCellField(
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:CubedSphereWithThicknessMap},
                                        C,
                                        O,
                                        <:ExtrinsicManifold}}) where {Dc,Da,G,A,C,O}
  model = Gridap.Geometry.get_background_model(trian)
  cell_ambient_maps = get_cell_ambient_maps(model)
  ptrs = cell_ambient_maps.ptrs
  ambient_maps = cell_ambient_maps.values
  radius = ambient_maps[1].radius
  inv_ambient_maps = 
     [CubedSphereWithThicknessInvMap(panel, radius, ambient_maps[panel].thickness) for panel in 1:length(ambient_maps)]
  cell_inv_ambient_maps = CompressedArray(inv_ambient_maps, ptrs)
  Gridap.CellData.GenericCellField(cell_inv_ambient_maps, trian, Gridap.CellData.PhysicalDomain())
end

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


function _fm(f, m)
   function fm(m)
     αβ -> begin
         x = m(αβ)
         f(x)
     end
   end
end

 deriv_sqrt= x -> 0.5/sqrt(x)
 function deriv_det(x::SymTensorValue{2})
   Gridap.TensorValues.SymTensorValue(x[2,2],-x[2,1],x[1,1])
 end
 function deriv_det(x::SymTensorValue{3})
   Gridap.TensorValues.SymTensorValue(
     x[2,2]*x[3,3] - x[2,3]^2,
     x[2,3]*x[1,3] - x[1,2]*x[3,3],
     x[1,2]*x[2,3] - x[2,2]*x[1,3],
     x[1,1]*x[3,3] - x[1,3]^2,
     x[1,2]*x[1,3] - x[1,1]*x[2,3],
     x[1,1]*x[2,2] - x[1,2]^2,
   )
 end

cpAB = (A,B)->contracted_product(Val(2), A, permutedims(B,(2,3,1)))
function _Δs_no_ad(f, Ω_atlas)
  # surflap(f::Function) = m -> surflap(f,m)
  # surflap(f::Function,m::Field) = αβ -> 1/sqrtg(m,αβ) * ( divergence(W(f,m))(αβ) )
  # W(f::Function,m::Field) = αβ ->  sqrtg(m,αβ)*( inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )

  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  f_cf = f∘ambient_map_cf
  metric_cf = MetricCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  gradient_f_cf = (∇(f)∘ambient_map_cf)⋅covariant_basis_cf

  ## BEGIN Machinery to compute gradient(meas_cf)
  # v_l = A_ij * B_kij
  grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*
                   Operation(cpAB)(deriv_det∘metric_cf,gradient(metric_cf))
  ## END Machinery to compute gradient(meas_cf)

  ## BEGIN Machinery to compute gradient_gradient(f_cf)
  # A_ij = v_k * B_ijk
  cpvB=(v,B)->contracted_product(Val(1), v, permutedims(B,(3,1,2)))
  gradient_gradient_cf = ∇(ambient_map_cf)⋅(∇∇(f)∘ambient_map_cf)⋅covariant_basis_cf +
                                            Operation(cpvB)(∇(f)∘ambient_map_cf,
                                            ∇∇(ambient_map_cf))
  ## END Machinery to compute gradient_gradient(f_cf)

  # w_cf = meas_cf*(inv_metric_cf⋅gradient_f_cf)
  # divergence(w_cf) =
  #   grad(meas_cf)⋅( inv_metric_cf⋅gradient_f_cf ) + (1)
  #   meas_cf*divergence(inv_metric_cf⋅gradient_f_cf) (2+3) =
  #   grad(meas_cf)⋅( inv_metric_cf⋅gradient_f_cf ) +   (1)
  #   meas_cf*divergence(inv_metric_cf)⋅gradient_f_cf + (2)
  #   meas_cf*(inv_metric_cf ⊙ gradient(gradient_f_cf)) (3)
  div_wcf_first_term = grad_meas_cf⋅(inv_metric_cf⋅gradient_f_cf)
  div_wcf_second_term = meas_cf*(divergence(inv_metric_cf)⋅gradient_f_cf)
  div_wcf_third_term = meas_cf*(inv_metric_cf ⊙ gradient_gradient_cf)
  div_wcf = div_wcf_first_term + div_wcf_second_term + div_wcf_third_term
  1.0/meas_cf * div_wcf
end

function _Δs_ad(f, Ω_atlas)
  # surflap(f::Function) = m -> surflap(f,m)
  # surflap(f::Function,m::Field) = αβ -> 1/sqrtg(m,αβ) * ( divergence(W(f,m))(αβ) )
  # W(f::Function,m::Field) = αβ ->  sqrtg(m,αβ)*( inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(surflap(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

## f is an scalar-valued ambient-space function
function Δs(f::Function,
            Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    use_automatic_differentiation ? _Δs_ad(f, Ω_atlas) : _Δs_no_ad(f, Ω_atlas)
end

function Δs(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    Δs_trian = use_automatic_differentiation ? _Δs_ad(f, Ω_atlas.trian) : _Δs_no_ad(f, Ω_atlas.trian)
    Gridap.CellData.GenericCellField(get_data(Δs_trian), Ω_atlas, Gridap.CellData.DomainStyle(Δs_trian))
end

function _compose(parametric_space_quantity, inv_ambient_map_cell_field)
    # Not able to do Δs_parametric_space ∘ InvAmbientMapCellField(Ω_atlas) with Gridap
    # I perform the composition manually with lazy_map below as a workaround.
    parametric_space_data = Gridap.CellData.get_data(parametric_space_quantity)
    inv_ambient_map_data = Gridap.CellData.get_data(inv_ambient_map_cell_field)
    composed_data = lazy_map(∘, parametric_space_data, inv_ambient_map_data)
    CellData.GenericCellField(composed_data,
                              Gridap.Geometry.get_triangulation(inv_ambient_map_cell_field),
                              Gridap.CellData.PhysicalDomain())
end

function Δs(f::Function,
            Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    Δs_parametric_space = use_automatic_differentiation ? _Δs_ad(f, Ω_atlas) : _Δs_no_ad(f, Ω_atlas)
    _compose(Δs_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function Δs(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMEM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    Δs_parametric_space = use_automatic_differentiation ? _Δs_ad(f, Ω_atlas) : _Δs_no_ad(f, Ω_atlas)
    _compose(Δs_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function _∇s_no_ad(f, Ω_atlas)
  # sgrad(f::Function) = m -> sgrad(f,m)
  # sgrad(f::Function,m::Field) = αβ -> J(m,αβ) ⋅
  #                                     (inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  gradient_f_cf = (∇(f)∘ambient_map_cf)⋅covariant_basis_cf
  covariant_basis_cf⋅(inv_metric_cf⋅gradient_f_cf)
end

function _∇s_ad(f, Ω_atlas)
  # sgrad(f::Function) = m -> sgrad(f,m)
  # sgrad(f::Function,m::Field) = αβ -> J(m,αβ) ⋅
  #                                     (inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(sgrad(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end


function ∇s(f::Function,
            Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _∇s_ad(f, Ω_atlas) : _∇s_no_ad(f, Ω_atlas)
end

function ∇s(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_trian = use_automatic_differentiation ? _∇s_ad(f, Ω_atlas.trian) : _∇s_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(∇s_trian), Ω_atlas, Gridap.CellData.DomainStyle(∇s_trian))
end

function ∇s(f::Function,
            Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _∇s_ad(f, Ω_atlas) : _∇s_no_ad(f, Ω_atlas)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function ∇s(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMEM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _∇s_ad(f, Ω_atlas) : _∇s_no_ad(f, Ω_atlas)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function _skew_∇s_no_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  J_cf = transpose∘∇(ambient_map_cf)
  grad_f_cf = (∇(f)∘ambient_map_cf)⋅J_cf
  skew_grad_parametric = J_cf⋅(perp∘grad_f_cf)*(1.0/meas_cf) 
end

function _skew_∇s_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(skew_surfgrad(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function skew_∇s(f::Function, Ω_atlas::BFTATDMIM{2,2,Da,G,A,P,C,O};
                   use_automatic_differentiation=false) where {Da, G, A, P, C, O}
   use_automatic_differentiation ? _skew_∇s_ad(f, Ω_atlas) : _skew_∇s_no_ad(f, Ω_atlas)
end

function skew_∇s(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                   use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  @notimplemented "skew_∇s is only implemented for 2D surfaces"
end

function skew_∇s(f::Function, Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
                   use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  @notimplemented "skew_∇s is only implemented for intrinsic manifolds"
end


# Contravariant components of 3D vector vecX
# The contravariatn mapping is  ̃u = J u
# so u = J^† ̃u
contra_v(vecX::Function,m::Field) = αβ -> forward_pinv_jacobian(m)(αβ)⋅vecX(m)(αβ)
contra_v(vecX::Function) = p -> contra_v(vecX,p)

function _divs_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(surfdiv(contra_v(_fm(f,m)),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _divs_no_ad(f, Ω_atlas)
    # 1/m * div( m * (J^†⋅(f∘ϕ)) ), where J^†=inv(g)⋅Jᵀ
    # grad(m)⋅(J^†⋅(f∘ϕ)) + 
    # div((J^†⋅(f∘ϕ))) = tr(grad(J^†):(f∘ϕ)) + tr(J^†⋅grad(f∘ϕ))
    # grad(J^†) = grad(inv(g)⋅Jᵀ) = grad(inv(g))⋅Jᵀ + inv(g)⊙grad(Jᵀ)
    metric_cf = MetricCellField(Ω_atlas)
    meas_cf = MeasureCellField(Ω_atlas)
    inv_metric_cf = InvMetricCellField(Ω_atlas)
    ambient_map_cf = AmbientMapCellField(Ω_atlas)
    grad_ambient_map_cf = ∇(ambient_map_cf)
    f_cf = f∘ambient_map_cf
    grad_f_cf = ∇(f)∘ambient_map_cf
    Jt_cf = ∇(ambient_map_cf)
    
    # grad(inv(g))⋅Jᵀ
    grad_inv_metric_cf = ∇(inv_metric_cf)
    trace_1=Operation(tr)((grad_inv_metric_cf⋅Jt_cf)⋅f_cf)
    
    # inv(g)⋅grad(Jᵀ)
    trace_2 = Operation(tr)((inv_metric_cf ⋅ ∇(Jt_cf))⋅f_cf)

    # tr(J^†⋅grad(f∘ϕ)) = tr((inv(g)⋅Jᵀ)⋅grad(f∘ϕ))
    trace_3=Operation(tr)((inv_metric_cf⋅grad_ambient_map_cf)⋅
                              ((grad_f_cf)⋅(transpose∘grad_ambient_map_cf)))

    grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*
                   Operation(cpAB)(deriv_det∘metric_cf,gradient(metric_cf))
   

    return (1.0/meas_cf)*(meas_cf*(trace_1+trace_2+trace_3) + 
                           grad_meas_cf⋅(inv_metric_cf⋅Jt_cf⋅f_cf))
end

function _skew_divs_no_ad(f, Ω_atlas)
    # -1/m * div( m^2 * inv(g) R(J^†⋅(f∘ϕ)) ), where J^†=inv(g)⋅Jᵀ
    # div( m^2 * inv(g) R(J^†⋅(f∘ϕ)) )
    # div( m^2 * inv(g) R(J^†⋅(f∘ϕ)) ) = 
    #   grad(m^2)⋅(inv(g) R(J^†⋅(f∘ϕ))) + m^2 * div(inv(g) R(J^†⋅(f∘ϕ)))
    # div(inv(g) R(J^†⋅(f∘ϕ))) = tr(grad(inv(g))⋅R(J^†⋅(f∘ϕ))) + tr(inv(g)⋅grad(R(J^†⋅(f∘ϕ))))
    #
    Gridap.Helpers.@notimplemented "skew_divs without automatic differentiation is not implemented yet"  
end

function _skew_divs_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(skew_surfdiv(contra_v(_fm(f,m)),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

# Surface divergence of an ambient vector-valued function which is 
# pulled back using the pseudo-inverse of the jacobian of the ambient 
# map without multiplying by the measure
function divs(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
              use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
   use_automatic_differentiation ? _divs_ad(f, Ω_atlas) : _divs_no_ad(f, Ω_atlas)
end

function divs(f::Function,
              Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
              use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  divs_trian = use_automatic_differentiation ? _divs_ad(f, Ω_atlas.trian) : _divs_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(divs_trian), Ω_atlas, Gridap.CellData.DomainStyle(divs_trian))
end

function divs(f::Function,
              Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
              use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _divs_ad(f, Ω_atlas) : _divs_no_ad(f, Ω_atlas)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function divs(f::Function,
              Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMEM{Dc,Dc,Da,G,A,P,C,O}};
              use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _divs_ad(f, Ω_atlas.trian) : _divs_no_ad(f, Ω_atlas.trian)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end


function skew_divs(f::Function, Ω_atlas::BFTATDMIM{2,2,Da,G,A,P,C,O};
                   use_automatic_differentiation=true) where {Da, G, A, P, C, O}
   use_automatic_differentiation ? _skew_divs_ad(f, Ω_atlas) : _skew_divs_no_ad(f, Ω_atlas)
end

function skew_divs(f::Function,
                   Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
                   use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  skew_divs_trian = use_automatic_differentiation ? _skew_divs_ad(f, Ω_atlas.trian) : _skew_divs_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(skew_divs_trian), Ω_atlas, Gridap.CellData.DomainStyle(skew_divs_trian))
end

function skew_divs(f::Function,
                   Ω_atlas::BFTATDMEM{2,2,Da,G,A,P,C,O};
                   use_automatic_differentiation=true) where {Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _skew_divs_ad(f, Ω_atlas) : _skew_divs_no_ad(f, Ω_atlas)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function skew_divs(f::Function,
                   Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMEM{2,2,Da,G,A,P,C,O}};
                   use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _skew_divs_ad(f, Ω_atlas.trian) : _skew_divs_no_ad(f, Ω_atlas.trian)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function skew_divs(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                   use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  @notimplemented "skew_divs is only implemented for 2D surfaces"
end

function skew_divs(f::Function, Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
                   use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  @notimplemented "skew_divs is only implemented for 2D surfaces"
end

dagger(vec::Function) = m -> dagger(vec,m)
dagger(vec::Function,m::Field) = αβ ->  J(m)(αβ)⋅(inv_metric(m,αβ)⋅perp( contra_v(vec(m))(αβ) )) * sqrtg(m,αβ)

function _dagger_ad(f::Function, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(dagger(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _dagger_no_ad(f::Function, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  f_cf = f∘ambient_map_cf
  measure_cf = MeasureCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  J_cf = transpose∘∇(ambient_map_cf)
  f_cf_parametric = (pinvJ∘J_cf)⋅f_cf
  J_cf⋅(inv_metric_cf⋅(perp∘f_cf_parametric))*measure_cf
end

function dagger(f::Function, Ω_atlas::BFTATDMIM{2,2,Da,G,A,P,C,O};
                use_automatic_differentiation=false) where {Da, G, A, P, C, O}
  use_automatic_differentiation ? _dagger_ad(f, Ω_atlas) : _dagger_no_ad(f, Ω_atlas)
end

function dagger(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  @notimplemented "dagger is only implemented for 2D surfaces"
end

function dagger(f::Function, Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  @notimplemented "dagger is only implemented for 2D surfaces"
end


Jt(m) = x -> transpose(J(m,x))
Jtu(u,m) = x -> Jt(m)(x)⋅u(m)(x)
# Returns the co-vector associated to the surface curl of a vector-valued field
curls(u,m) = x-> 1.0/sqrtg(m,x)*metric(m,x)⋅curl(Jtu(u,m))(x)
# Returns the co-vector associated to the surface curl of the surface curl of a vector-valued field
curls_curls(u, m) = x -> 1.0/sqrtg(m,x)*metric(m,x)⋅curl(curls(u,m))(x)

## surface divergence
_divs(u,m) = x -> sqrtg(m)(x)*inv(J(m,x))⋅u(m)(x)
divs(u, m) = x -> 1/sqrtg(m)(x)*(divergence(_divs(u,m))(x))

### covariant vector surfgrad(surfdiv u)
grads_divs(u, m) = x-> gradient(divs(u,m))(x)
vec_laps(u,m) = x -> grads_divs(u,m)(x) - curls_curls(u,m)(x)

function _vecΔs_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(vec_laps(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _vecΔs_no_ad(f, Ω_atlas)
  Gridap.Helpers.@notimplemented "vecΔs without automatic differentiation is not implemented yet"  
end

# Returns the co-vector components of the vector surface laplacian applied to the 
# ambient vector-valued function f
function vecΔs(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _vecΔs_ad(f, Ω_atlas) : _vecΔs_no_ad(f, Ω_atlas)
end

function vecΔs(f::Function, Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  vecΔs_trian = use_automatic_differentiation ? _vecΔs_ad(f, Ω_atlas.trian) : _vecΔs_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(vecΔs_trian), Ω_atlas, Gridap.CellData.DomainStyle(vecΔs_trian))
end

# ### Curl of covariant components of u
# ucov(u,m,x) = Jt(m)(x)⋅u(m)(x)
# ucov(u,m) = x -> ucov(u,m,x)
# curl_ucov(u,m,x) = curl(ucov(u,m))(x)
# curl_ucov(u,m) = x -> curl_ucov(u,m,x)

# ### Covariant components of surfcurl u
# _curls(u,m,x) = 1.0/sqrtg(m,x)*metric(m,x)⋅curl_ucov(u,m,x)
# _curls(u,m) = x -> _curls(u,m,x)
# curls(u,m,x) = curl(_curls(u,m))(x)
# curls(u,m) = x -> curls(u,m,x)
function _curls_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(curls(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _curls_no_ad(f, Ω_atlas)
  Gridap.Helpers.@notimplemented "curls without automatic differentiation is not implemented yet"  
end

# Returns the co-vector components of the surface curl operator applied to the 
# ambient vector-valued function f
function curls(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _curls_ad(f, Ω_atlas) : _curls_no_ad(f, Ω_atlas)
end

function curls(f::Function, Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  curls_trian = use_automatic_differentiation ? _curls_ad(f, Ω_atlas.trian) : _curls_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(curls_trian), Ω_atlas, Gridap.CellData.DomainStyle(curls_trian))
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

"""
dagger
 
computes ̃u^† = ̃k × ̃u, where ̃k is only defined for ambient models.
This function will fail if get_surface_normal fails (i.e for parametric models)
"""
function dagger(u::CellField)
  trian = get_triangulation(u)
  n = get_surface_normal(trian)
  n×u
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
