# AtlasGrids.jl
#
# Defines AtlasGrid{Dc,Da}: a Grid{Dc,Dc} whose cells carry local reference coords
# (Dc-dimensional, one coordinate system per coarse chart).  Ambient Da-dimensional
# coords are NOT stored here; they are computed only in visualization_data.

using Gridap
using Gridap.Geometry, Gridap.Fields, Gridap.Arrays, Gridap.ReferenceFEs, Gridap.Helpers
using Gridap.Adaptivity, Gridap.Visualization
import Gridap.TensorValues: symmetric_part, SymTensorValue

# ============================================================
# IdentityField
# ============================================================

"""
    IdentityField{D}()

Zero-cost `Field` that maps every `Point{D}` to itself.
Used as the ambient map when no embedding is available (intrinsic manifold
without an explicit ambient-space representation).

`gradient(IdentityField{D}())` returns an `IdentityFieldGrad{D}()` whose value is
`one(TensorValue{D,D,Float64})` at every point — the Jacobian of the identity is the identity tensor.
"""
struct IdentityField{D} <: Field end

Gridap.Arrays.evaluate!(cache, ::IdentityField{D}, x::Point{D,T}) where {D,T} = x

function Gridap.Arrays.return_cache(::IdentityField{D}, xs::AbstractArray{<:Point{D,T}}) where {D,T}
  nothing
end
function Gridap.Arrays.evaluate!(cache, ::IdentityField{D}, xs::AbstractArray{<:Point{D}}) where D
  xs
end

struct IdentityFieldGrad{D} <: Field end

Gridap.Arrays.evaluate!(cache, ::IdentityFieldGrad{D}, x::Point{D}) where D =
  one(TensorValue{D,D,Float64})
function Gridap.Arrays.return_cache(::IdentityFieldGrad{D}, xs::AbstractArray{<:Point{D}}) where D
  CachedArray(similar(xs, TensorValue{D,D,Float64}))
end
function Gridap.Arrays.evaluate!(cache, ::IdentityFieldGrad{D}, xs::AbstractArray{<:Point{D}}) where D
  setsize!(cache, size(xs))
  fill!(cache.array, one(TensorValue{D,D,Float64}))
  cache.array
end

Gridap.Fields.gradient(::IdentityField{D}) where D = IdentityFieldGrad{D}()

# ============================================================
# ManifoldStyle trait
# ============================================================

"""
    ManifoldStyle

Trait controlling how `get_cell_map` assembles the cell map in `AtlasGrid`.

- `ExtrinsicManifold()` — compose RefFE→Chart with Chart→Ambient (default).
  Gridap integration uses the full 3D Jacobian; the metric is implicit.
- `IntrinsicManifold()` — return only the RefFE→Chart map.
  The metric must be queried explicitly via `get_cell_metric` to write PDEs.
"""
abstract type ManifoldStyle end
struct ExtrinsicManifold <: ManifoldStyle end
struct IntrinsicManifold <: ManifoldStyle end

# ============================================================
# JtJ: metric from Jacobian transpose  (g = JᵀJ, J = ∇φ)
# ============================================================

"""
    JtJ <: Map

`Map` that computes the pullback metric `g = symmetric_part(J ⋅ Jᵀ)` from the
Jacobian transpose `J = ∇φ` (a `TensorValue{Dc,Da}`).  Used with `Operation` to
build per-cell metric fields lazily:

    metric_field = Operation(JtJ())(gradient(ambient_map))
"""
struct JtJ <: Gridap.Arrays.Map end
Gridap.Arrays.evaluate!(cache, ::JtJ, J) = symmetric_part(J ⋅ transpose(J))

_pullback_metrics(ambient_maps) = [Operation(JtJ())(gradient(φ)) for φ in ambient_maps]

# ============================================================
# lazy_map(Reindex(plain_vector), indices) → CompressedArray
# ============================================================
#
# Gridap already handles Reindex{<:Fill}, Reindex{<:CompressedArray}, and
# Reindex{<:LazyArray} with specialised methods (more specific → higher priority).
# The plain AbstractArray case is missing; without it lazy_map falls through to
# LazyArray, losing the CompressedArray optimisation downstream (e.g. in
# _lazy_map_compressed when applying gradient/metric/inverse lazily over cells).
#
# Candidate for upstreaming to Gridap.Arrays.Reindex.
function Gridap.Arrays.lazy_map(
    k       :: Gridap.Arrays.Reindex{<:AbstractArray},
    ::Type{T},
    j_to_i  :: AbstractArray,
) where T
  Gridap.Arrays.CompressedArray(k.values, j_to_i)
end

# ============================================================
# AtlasGrid
# ============================================================

"""
    AtlasGrid{Dc,Da,G,A,P,C,O,M} <: Gridap.Geometry.Grid{Dc,Dc}

A DG-style grid for a `Dc`-dimensional manifold atlas.  Each fine cell stores its
corners in the **local reference frame** of its coarse chart (Dc-dimensional).
Ambient Da-dimensional coordinates are never materialised here; they are computed
on demand only during visualization.

# Type parameters
- `Dc` — cell dimension (= manifold dimension)
- `Da` — ambient space dimension
- `G`  — concrete type of `param_grid`
- `A`  — concrete type of `cell_chart_coords`
- `P`  — concrete type of `cell_ambient_maps`
- `C`  — concrete type of `cell_metric`
- `O`  — concrete `OrientationStyle` subtype
- `M`  — concrete `ManifoldStyle` subtype (drives `get_cell_map` dispatch)

# Fields
- `param_grid`          — fine `Grid{Dc,Dc}` from uniform refinement (topology/connectivity).
- `cell_chart_coords`   — per-cell local corner coords, DG-style.  May be a lazy
                          `LazyArray` (serial path via `_build_atlas_grid`) or an
                          eager `Table` (distributed path via p4est).
- `cell_ambient_maps`  — lazy per-cell map `Point{Dc} → Point{Da}`.  Always concrete:
                          `Fill(IdentityField{Dc}(), n)` when no embedding is provided,
                          giving `Da = Dc` and chart-space visualization.
- `cell_metric`         — lazy per-cell `Field{Dc → SymTensorValue{Dc}}`.  Always concrete:
                          either an analytic expression from `CoarseMeshInfo` or
                          `_pullback_metrics(ambient_maps)` computed lazily from the
                          ambient map.
- `orientation_style`   — kept explicitly because atlas orientation can differ from the
                          underlying grid topology (e.g. a Möbius strip).
- (M)                   — `ManifoldStyle` encoded as type parameter; not stored as a field.
                          Access via `ManifoldStyle(g)`.
"""
struct AtlasGrid{Dc, Da,
                 G <: Gridap.Geometry.Grid{Dc,Dc},
                 A <: AbstractVector,
                 P <: AbstractVector{<:Field},
                 C <: AbstractVector,
                 O <: Gridap.Geometry.OrientationStyle,
                 M <: ManifoldStyle} <: Gridap.Geometry.Grid{Dc,Dc}
  param_grid         :: G
  cell_chart_coords  :: A
  cell_ambient_maps :: P
  cell_metric        :: C
  orientation_style  :: O

  function AtlasGrid(
    param_grid         :: Gridap.Geometry.Grid{Dc,Dc},
    cell_chart_coords  :: AbstractVector,
    cell_ambient_maps  :: AbstractVector{<:Field},
    cell_metric        :: AbstractVector,
    orientation_style  :: Gridap.Geometry.OrientationStyle,
    manifold_style     :: ManifoldStyle,
  ) where Dc
    sample_pt = cell_chart_coords[1][1]
    fwd0      = cell_ambient_maps[1]
    Da        = Gridap.TensorValues.num_components(Gridap.Arrays.return_type(fwd0, sample_pt))
    n = Gridap.Geometry.num_cells(param_grid)
    @check length(cell_chart_coords)  == n  "cell_chart_coords has $(length(cell_chart_coords)) entries but param_grid has $n cells"
    @check length(cell_ambient_maps)  == n  "cell_ambient_maps has $(length(cell_ambient_maps)) entries but param_grid has $n cells"
    @check length(cell_metric)        == n  "cell_metric has $(length(cell_metric)) entries but param_grid has $n cells"
    G = typeof(param_grid)
    A = typeof(cell_chart_coords)
    P = typeof(cell_ambient_maps)
    C = typeof(cell_metric)
    O = typeof(orientation_style)
    M = typeof(manifold_style)
    new{Dc,Da,G,A,P,C,O,M}(param_grid, cell_chart_coords, cell_ambient_maps, cell_metric, orientation_style)
  end
end

# ----------------------------------------------------------
# ManifoldStyle accessor (mirrors OrientationStyle pattern)
# ----------------------------------------------------------

ManifoldStyle(::Type{<:AtlasGrid{Dc,Da,G,A,P,C,O,M}}) where {Dc,Da,G,A,P,C,O,M} = M()
ManifoldStyle(g::AtlasGrid) = ManifoldStyle(typeof(g))

# ----------------------------------------------------------
# Private helper: build RefFE→Chart maps (common to both styles)
# ----------------------------------------------------------

function _chart_maps(g::AtlasGrid)
  shapefuns = Gridap.ReferenceFEs.get_shapefuns(only(Gridap.Geometry.get_reffes(g)))
  lazy_map(linear_combination, g.cell_chart_coords, Gridap.Arrays.Fill(shapefuns, num_cells(g)))
end

# ----------------------------------------------------------
# _build_atlas_grid: shared private constructor
# ----------------------------------------------------------

"""
    _build_atlas_grid(coarse_model, coarse_chart_coords, ambient_maps, metric_fields,
                      num_refinements, orientation_style, manifold_style) -> (AtlasGrid, fine_model)

Private helper shared by the `AtlasGrid` and `AtlasDiscreteModel` outer constructors.
Refines `coarse_model` `num_refinements` times and returns both the `AtlasGrid` and the
`fine_model` so the caller can extract topology and face labeling without re-refining.

`num_refinements` must be ≥ 1.  The face labeling of `coarse_model` is propagated to
`fine_model` by Gridap's refinement machinery.

# Arguments
- `coarse_model`       — coarse `DiscreteModel{Dc,Dc}` with face labeling.
- `coarse_chart_coords`— `AbstractVector`; entry `k` is a vector of `Point{Dc}` giving
                         the corners of chart `k` in its local reference frame.
- `ambient_maps`       — `AbstractVector` of `Field`s; `ambient_maps[k]` maps
                         `Point{Dc} → Point{Da}` for chart `k`.
- `metric_fields`      — `AbstractVector` of `Field`s; `metric_fields[k]` maps
                         `Point{Dc} → SymTensorValue{Dc}` (the pullback metric for chart `k`).
- `num_refinements`    — number of uniform refinements (≥ 0; 0 returns the coarse mesh as-is).
- `orientation_style`  — grid orientation; `nothing` → copied from the fine param_grid.
- `manifold_style`     — `ExtrinsicManifold()` or `IntrinsicManifold()`.
"""
function _build_atlas_grid(
    coarse_model        :: Gridap.Geometry.DiscreteModel{Dc,Dc},
    coarse_chart_coords,
    ambient_maps,
    metric_fields,
    num_refinements :: Int,
    orientation_style,
    manifold_style,
) where Dc

  @check num_refinements >= 0 "num_refinements must be ≥ 0, got $num_refinements"

  # ── 1. Refine coarse_model in one shot → fine_model, cell_to_chart ───────────
  #    Convert to UnstructuredDiscreteModel unconditionally so that refine(model, n)
  #    always dispatches to uniformly_refine → blocked_refinement_glue (parent-major).
  #    CartesianDiscreteModel.refine would use _create_cartesian_f2c_maps instead,
  #    giving lexicographic ordering that would silently break the coord tiling below.
  #    See test_refinement_ordering.jl for a verification of parent-major ordering.
  coarse_unstr = Gridap.Geometry.UnstructuredDiscreteModel(coarse_model)
  ncharts      = Gridap.Geometry.num_cells(coarse_unstr)

  if num_refinements == 0
    # Zero refinements: fine mesh is the coarse mesh; each cell maps to its own chart.
    fine_model    = coarse_unstr
    cell_to_chart = collect(1:ncharts)
    local_lazy    = coarse_chart_coords
  else
    n_factor      = 2^num_refinements

    adapted       = Gridap.Adaptivity.refine(coarse_unstr, n_factor)
    glue          = adapted.glue
    fine_model    = adapted.model
    cell_to_chart = Vector{Int}(glue.n2o_faces_map[Dc+1])

    # ── 2. Get reference positions from the refinement rule's ref model ─────────
    #    compute_reference_grid(p, n_factor) produces the n_factor×n_factor ref grid
    #    directly — no further refinement needed.
    ref_model   = Gridap.Adaptivity.get_ref_grid(glue.refinement_rules[1])
    ref_coords  = Gridap.Geometry.get_cell_coordinates(Gridap.Geometry.get_grid(ref_model))
    n_per_chart = Gridap.Geometry.num_cells(Gridap.Geometry.get_grid(ref_model))

    # ── 3. Build per-chart local maps Ψ_k : ref_element → coarse_chart_coords[k] ──
    reffe     = Gridap.Geometry.get_reffes(Gridap.Geometry.get_grid(coarse_unstr))[1]
    shapefuns = Gridap.ReferenceFEs.get_shapefuns(reffe)
    Ψ_maps    = map(corners -> linear_combination(corners, shapefuns), coarse_chart_coords)

    # ── 4. Tile ref_coords and apply Ψ_k lazily ──────────────────────────────────
    #    child_ids[i] = position of fine cell i within its coarse block (1..n_per_chart).
    #    Valid because blocked_refinement_glue guarantees parent-major ordering.
    #    LinearCombinationField supports array evaluate! directly — no Broadcasting needed.
    child_ids    = repeat(1:n_per_chart, ncharts)
    ref_per_fine = lazy_map(Reindex(ref_coords), child_ids)
    chart_Ψ      = lazy_map(Reindex(Ψ_maps), cell_to_chart)
    local_lazy   = lazy_map(evaluate, chart_Ψ, ref_per_fine)
  end

  # ── 5. Resolve orientation_style ─────────────────────────────────────────
  os = if isnothing(orientation_style)
    Gridap.Geometry.OrientationStyle(Gridap.Geometry.get_grid(fine_model))
  else
    orientation_style
  end

  # ── 6. Build per-cell ambient maps and metric lazily from cell_to_chart ──
  cell_ambient_maps = lazy_map(Reindex(ambient_maps), cell_to_chart)
  cell_metric       = lazy_map(Reindex(metric_fields),  cell_to_chart)

  atlas_grid = AtlasGrid(
    Gridap.Geometry.get_grid(fine_model),
    local_lazy,
    cell_ambient_maps,
    cell_metric,
    os,
    manifold_style,
  )

  atlas_grid, fine_model
end

"""
    AtlasGrid(coarse_model, coarse_chart_coords, ambient_maps, num_refinements=1;
              metric_fields=nothing, orientation_style=nothing, manifold_style=ExtrinsicManifold())

Build an `AtlasGrid` from a coarse model.  When `metric_fields` is not provided it is
computed lazily via `_pullback_metrics(ambient_maps)` for each ambient map `φ`.
"""
function AtlasGrid(
    coarse_model        :: Gridap.Geometry.DiscreteModel{Dc,Dc},
    coarse_chart_coords,
    ambient_maps,
    num_refinements :: Int = 1;
    metric_fields     = nothing,
    orientation_style = nothing,
    manifold_style    = ExtrinsicManifold(),
) where Dc
  _metric = isnothing(metric_fields) ?
    _pullback_metrics(ambient_maps) : metric_fields
  atlas_grid, _ = _build_atlas_grid(
    coarse_model, coarse_chart_coords, ambient_maps, _metric,
    num_refinements, orientation_style, manifold_style)
  atlas_grid
end

# ----------------------------------------------------------
# Gridap.Geometry.Grid{Dc,Dc} interface
# ----------------------------------------------------------

Gridap.Geometry.OrientationStyle(
  ::Type{<:AtlasGrid{Dc,Da,G,A,P,C,O,M}}) where {Dc,Da,G,A,P,C,O,M} = O()

Gridap.Geometry.num_cells(g::AtlasGrid) = Gridap.Geometry.num_cells(g.param_grid)

Gridap.Geometry.get_reffes(g::AtlasGrid) = Gridap.Geometry.get_reffes(g.param_grid)

Gridap.Geometry.get_cell_type(g::AtlasGrid) = Gridap.Geometry.get_cell_type(g.param_grid)

Gridap.Geometry.get_cell_coordinates(g::AtlasGrid) = g.cell_chart_coords

# Delegate to param_grid: gives the correct shared-node count for FESpace/num_nodes.
# Coordinate values are junk (2D parametric), but FEM assembly uses get_cell_map (overridden
# below) and never reads these values — only the length matters.
Gridap.Geometry.get_node_coordinates(g::AtlasGrid) =
  Gridap.Geometry.get_node_coordinates(g.param_grid)

# get_cell_map dispatches on ManifoldStyle:
#   ExtrinsicManifold — RefFE→Chart composed with Chart→Ambient (current behavior)
#   IntrinsicManifold — RefFE→Chart only; metric queried via get_cell_metric
Gridap.Geometry.get_cell_map(g::AtlasGrid) = _get_cell_map(g, ManifoldStyle(g))

function _get_cell_map(g::AtlasGrid, ::ExtrinsicManifold)
  lazy_map(∘, g.cell_ambient_maps, _chart_maps(g))
end

function _get_cell_map(g::AtlasGrid, ::IntrinsicManifold)
  _chart_maps(g)
end

# Delegate to param_grid: shared-node IDs consistent with get_node_coordinates.
# visualization_data builds its own DG sequential table for VTK.
Gridap.Geometry.get_cell_node_ids(g::AtlasGrid) =
  Gridap.Geometry.get_cell_node_ids(g.param_grid)

# ----------------------------------------------------------
# Custom API
# ----------------------------------------------------------

get_param_grid(g::AtlasGrid)                       = g.param_grid
get_cell_ambient_maps(g::AtlasGrid)               = g.cell_ambient_maps
get_cell_metric(g::AtlasGrid)                      = g.cell_metric
get_cell_inv_metric(g::AtlasGrid)                  = lazy_map(inverse_metric_field, g.cell_metric)
get_ambient_dim(::AtlasGrid{Dc,Da}) where {Dc,Da}  = Da

# ============================================================
# Coarse mesh library and convenience constructors
# ============================================================

include("CoarseMeshes.jl")

"""
    AtlasGrid(info::CoarseMeshInfo, num_refinements;
              orientation_style=nothing, manifold_style=ExtrinsicManifold())

Build an `AtlasGrid` from a `CoarseMeshInfo`, using the analytic metric fields
stored in `info.metric_fields`.
"""
function AtlasGrid(
    info            :: CoarseMeshInfo,
    num_refinements :: Int;
    orientation_style = nothing,
    manifold_style    = ExtrinsicManifold(),
)
  AtlasGrid(info.model, info.cell_chart_coords, info.ambient_maps, num_refinements;
            metric_fields=info.metric_fields, orientation_style, manifold_style)
end

"""
    AtlasGrid(info::CoarseMeshInfo, ambient_maps, num_refinements;
              orientation_style=nothing, manifold_style=ExtrinsicManifold())

Build an `AtlasGrid` from a `CoarseMeshInfo` with custom ambient maps (overriding
`info.ambient_maps`).  Metric fields are recomputed from `ambient_maps` via `JtJ`.
"""
function AtlasGrid(
    info            :: CoarseMeshInfo,
    ambient_maps,
    num_refinements :: Int;
    orientation_style = nothing,
    manifold_style    = ExtrinsicManifold(),
)
  custom_metrics = _pullback_metrics(ambient_maps)
  AtlasGrid(info.model, info.cell_chart_coords, ambient_maps, num_refinements;
            metric_fields=custom_metrics, orientation_style, manifold_style)
end

"""
    AtlasGrid(mesh::CoarseMesh, num_refinements;
              orientation_style=nothing, manifold_style=ExtrinsicManifold())

Build an `AtlasGrid` directly from a mesh descriptor (e.g. `CubedSphereMesh(1.0)`).
"""
function AtlasGrid(
    mesh            :: CoarseMesh,
    num_refinements :: Int;
    orientation_style = nothing,
    manifold_style    = ExtrinsicManifold(),
)
  AtlasGrid(get_coarse_mesh(mesh), num_refinements; orientation_style, manifold_style)
end
