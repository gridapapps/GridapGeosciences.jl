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
Gridap.Geometry.get_grid_topology(m::AtlasDiscreteModel) = m.grid_topology
Gridap.Geometry.get_face_labeling(m::AtlasDiscreteModel) = m.face_labeling
Gridap.Geometry.num_point_dims(m::AtlasDiscreteModel)    = num_point_dims(get_grid(m))

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

function MeasureCellField(trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel}) where {Dc,Dp}
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
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_inv_metric(model), trian, Gridap.CellData.PhysicalDomain())
end

function AmbientMapCellField(
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_ambient_maps(model), trian, Gridap.CellData.PhysicalDomain())
end

# Right now only supported by AtlasDiscreteModel of the sphere
function InvAmbientMapCellField(
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc, 
                                        Da, 
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

function get_radius(model::AtlasDiscreteModel{Dc,Dp, G, A, <:AbstractVector{<:CubedSphereMap}}) where {Dc,Dp,G,A}
   model.atlas_grid.cell_ambient_maps.values[1].radius
end

function _fm(f, m)
   function fm(m)
     αβ -> begin
         x = m(αβ)
         f(x)
     end
   end
end

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
  deriv_sqrt= x -> 0.5/sqrt(x)
  function deriv_det(x)
    Gridap.TensorValues.SymTensorValue(x[2,2],-x[2,1],x[1,1])
  end
  # v_l = A_ij * B_kij
  # How to express this contraction in terms of tensor operations? 
  function contract_rank_2_rank_3(A,B)
    x1 = A[1,1]*B[1,1,1] + A[1,2]*B[1,1,2] + A[2,1]*B[1,2,1] + A[2,2]*B[1,2,2]
    x2 = A[1,1]*B[2,1,1] + A[1,2]*B[2,1,2] + A[2,1]*B[2,2,1] + A[2,2]*B[2,2,2]
   VectorValue(x1,x2)
  end
  grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*
                       Operation(contract_rank_2_rank_3)(deriv_det∘metric_cf,gradient(metric_cf))
  ## END Machinery to compute gradient(meas_cf)

  ## BEGIN Machinery to compute gradient_gradient(f_cf)
  # A_ij = v_k * B_ijk
  # How to express this contraction in terms of Gridap's tensor operations? 
  function contract_rank_1_rank_3(v,B)
    x11 = v[1]*B[1,1,1] + v[2]*B[1,1,2] + v[3]*B[1,1,3]
    x12 = v[1]*B[1,2,1] + v[2]*B[1,2,2] + v[3]*B[1,2,3]
    x21 = v[1]*B[2,1,1] + v[2]*B[2,1,2] + v[3]*B[2,1,3]
    x22 = v[1]*B[2,2,1] + v[2]*B[2,2,2] + v[3]*B[2,2,3]
    TensorValue(x11,x21,x12,x22)
  end
  gradient_gradient_cf = ∇(ambient_map_cf)⋅(∇∇(f)∘ambient_map_cf)⋅covariant_basis_cf + 
                          Operation(contract_rank_1_rank_3)(∇(f)∘ambient_map_cf,
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
            Ω_atlas::Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc, 
                                        Da, 
                                        G, 
                                        A, 
                                        P, 
                                        C, 
                                        O, 
                                        <:IntrinsicManifold}};
                                        use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    use_automatic_differentiation ? _Δs_ad(f, Ω_atlas) : _Δs_no_ad(f, Ω_atlas)
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
            Ω_atlas::Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc, 
                                        Da, 
                                        G, 
                                        A, 
                                        P, 
                                        C, 
                                        O, 
                                        <:ExtrinsicManifold}};
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
            Ω_atlas::Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc, 
                                        Da, 
                                        G, 
                                        A, 
                                        P, 
                                        C, 
                                        O, 
                                        <:IntrinsicManifold}};
                                        use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _∇s_ad(f, Ω_atlas) : _∇s_no_ad(f, Ω_atlas)
end 

function ∇s(f::Function, 
            Ω_atlas::Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc, 
                                        Da, 
                                        G, 
                                        A, 
                                        P, 
                                        C, 
                                        O, 
                                        <:ExtrinsicManifold}};
                                        use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _∇s_ad(f, Ω_atlas) : _∇s_no_ad(f, Ω_atlas)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end 

