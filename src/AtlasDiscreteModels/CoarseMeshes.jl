# CoarseMeshes.jl
#
# Library of canonical coarse meshes for AtlasGrid / AtlasDiscreteModel.
# Each shape is represented by a concrete subtype of CoarseMesh that
# carries the geometric parameters (radius, height, …).  Calling
# get_coarse_mesh(shape) returns a CoarseMeshInfo bundling the coarse
# DiscreteModel (with face labels), per-cell local-frame corner coordinates,
# and default ambient maps.
#
# No dependency on GridapGeosciences — suitable for eventual upstreaming to Gridap.

const n_panels       = 6
const cube_half_edge = π/4

# ============================================================
# CoarseMeshInfo
# ============================================================

"""
    CoarseMeshInfo{Dc, Dm, A, M, G}

Bundles a coarse `DiscreteModel{Dc,Dc}` (with face labels) with per-cell
local-frame corner coordinates, per-chart ambient maps, and per-chart metric fields.
Returned by `get_coarse_mesh`; consumed by the `AtlasGrid` and `AtlasDiscreteModel`
convenience constructors.

- `model`         — coarse `DiscreteModel{Dc,Dc}` carrying topology and
                    `FaceLabeling` (node coordinates are junk — only connectivity
                    matters).  For meshes with physical boundaries (e.g. cylinder),
                    boundary edges/nodes are tagged by `get_coarse_mesh`.
- `cell_chart_coords`  — one entry per coarse cell; `cell_chart_coords[k]` is a vector of
                    `Point{Dc}` giving the corners of chart k in its local frame.
- `ambient_maps`  — one `Field` per chart: `Point{Dc} → Point{Da}`.
- `metric_fields` — one `Field` per chart: `Point{Dc} → SymTensorValue{Dc}`,
                    the pullback metric `g`.  For built-in shapes these are
                    concrete analytic types (e.g. `CubedSphereMetricField`);
                    user-defined shapes may use `_pullback_metrics(ambient_maps)`
                    as a generic fallback.  The explicit inverse is obtained via
                    `inverse_metric_field(metric_field)`.
"""
struct CoarseMeshInfo{Dc,
                      Dm <: Gridap.Geometry.DiscreteModel{Dc,Dc},
                      A  <: AbstractVector,
                      M,
                      G}
  model              :: Dm
  cell_chart_coords  :: A
  ambient_maps       :: M
  metric_fields      :: G

  function CoarseMeshInfo(
      model             :: Gridap.Geometry.DiscreteModel{Dc,Dc},
      cell_chart_coords :: A,
      ambient_maps      :: M,
      metric_fields     :: G,
  ) where {Dc, A <: AbstractVector, M, G}
    Dm = typeof(model)
    new{Dc,Dm,A,M,G}(model, cell_chart_coords, ambient_maps, metric_fields)
  end
end

# ============================================================
# CoarseMesh
# ============================================================

"""
    CoarseMesh

Supertype for all canonical coarse-mesh descriptors.
Subtypes carry the geometric parameters (radius, height, …) and are passed
to `get_coarse_mesh` to obtain a `CoarseMeshInfo`.
"""
abstract type CoarseMesh end

# ============================================================
# CylinderMesh
# ============================================================

"""
    CylinderMesh(radius=1.0, height=1.0)

3×3 atlas for a cylinder: 3 cells around the circumference × 3 rows along
the height = 9 cells, 12 nodes.

With 3 cells per ring each bottom/top circle edge belongs to exactly one
cell (genuine boundary edge). The 3 vertical seam edges are shared between
two adjacent cells (interior). All edge permutation indices are 1.

**Why 3 cells minimum?** Gridap's topology construction identifies edges by
their unordered vertex pair.  With 2 cells (C1=[1,2,3,4], C2=[2,1,4,3])
the seam wraps with the same orientation so C2's four edge vertex-pairs are
`{1,2},{1,3},{2,4},{3,4}` — identical to C1's.  Every edge is counted as
shared; no boundary edges appear and the incidence tables degenerate.
With 3 cells each adjacent pair shares exactly one edge, leaving the
top/bottom edges unshared and correctly tagged as boundary.
"""
struct CylinderMesh <: CoarseMesh
  radius :: Float64
  height :: Float64
  CylinderMesh(radius=1.0, height=1.0) = new(radius, height)
end

# ── CylinderChartMap ─────────────────────────────────────────────────────────
#
# Chart map (θ,z) → (r·cosθ, r·sinθ, z) and its explicit Jacobian.
# gradient(CylinderChartMap) overrides Gridap's default FieldGradient path.
# Array evaluate! methods are required: lazy_map(∘, cell_ambient_maps, chart_maps)
# decomposes into lazy_map(evaluate, cell_ambient_maps, chart_coord_arrays) during
# FE assembly (ApplyOptimizations.jl), so the array path is in the hot path.

struct CylinderChartMap <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CylinderChartMap, x::Point) =
  Point(m.radius*cos(x[1]), m.radius*sin(x[1]), x[2])
function Gridap.Arrays.return_cache(m::CylinderChartMap, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, Point{3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CylinderChartMap, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  r = m.radius
  @inbounds for i in eachindex(xs)
    x = xs[i]
    cache.array[i] = Point(r*cos(x[1]), r*sin(x[1]), x[2])
  end
  cache.array
end

struct CylinderChartMapGrad <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CylinderChartMapGrad, x::Point) =
  TensorValue{2,3,Float64}(-m.radius*sin(x[1]), 0.0, m.radius*cos(x[1]), 0.0, 0.0, 1.0)
function Gridap.Arrays.return_cache(m::CylinderChartMapGrad, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, TensorValue{2,3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CylinderChartMapGrad, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  r = m.radius
  @inbounds for i in eachindex(xs)
    x = xs[i]
    cache.array[i] = TensorValue{2,3,Float64}(-r*sin(x[1]), 0.0, r*cos(x[1]), 0.0, 0.0, 1.0)
  end
  cache.array
end

Gridap.Fields.gradient(f::CylinderChartMap) = CylinderChartMapGrad(f.radius)

# ── CylinderMetricField ───────────────────────────────────────────────────────
#
# Pullback metric g = diag(r², 1) — constant in chart coordinates.
# Inverse g⁻¹ = diag(1/r², 1).

struct CylinderMetricField <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CylinderMetricField, x::Point) =
  SymTensorValue{2,Float64,3}(m.radius^2, 0.0, 1.0)
function Gridap.Arrays.return_cache(m::CylinderMetricField, xs::AbstractArray{<:Point})
  CachedArray(fill(SymTensorValue{2,Float64,3}(m.radius^2, 0.0, 1.0), size(xs)))
end
function Gridap.Arrays.evaluate!(cache, m::CylinderMetricField, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  fill!(cache.array, SymTensorValue{2,Float64,3}(m.radius^2, 0.0, 1.0))
  cache.array
end

struct CylinderInvMetricField <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CylinderInvMetricField, x::Point) =
  SymTensorValue{2,Float64,3}(1/m.radius^2, 0.0, 1.0)
function Gridap.Arrays.return_cache(m::CylinderInvMetricField, xs::AbstractArray{<:Point})
  CachedArray(fill(SymTensorValue{2,Float64,3}(1/m.radius^2, 0.0, 1.0), size(xs)))
end
function Gridap.Arrays.evaluate!(cache, m::CylinderInvMetricField, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  fill!(cache.array, SymTensorValue{2,Float64,3}(1/m.radius^2, 0.0, 1.0))
  cache.array
end

inverse_metric_field(f::CylinderMetricField) = CylinderInvMetricField(f.radius)

# ── get_coarse_mesh(CylinderMesh) ────────────────────────────────────────────

"""
    get_coarse_mesh(m::CylinderMesh) → CoarseMeshInfo{2}

Hard-coded 3×3 QUAD coarse mesh for a cylinder (radius `m.radius`, height `m.height`).
12 nodes (4 rings × 3), 9 cells.  Gridap Z-order per cell: BL, BR, TL, TR.

  θ=0    2π/3   4π/3   (2π≡0)
  10─────11─────12────(10)
  │  C7  │  C8  │  C9  │
   7──────8──────9────(7)
  │  C4  │  C5  │  C6  │
   4──────5──────6────(4)
  │  C1  │  C2  │  C3  │
   1──────2──────3────(1)

Cells C3, C6, C9 carry the seam wrap: their BR/TR indices are 1, 4, 7 / 4, 7, 10.
Bottom boundary (entity 3): edges {1,2} {2,3} {3,1} — each in exactly one cell.
Top    boundary (entity 4): edges {10,11} {11,12} {12,10}.
"""
function get_coarse_mesh(m::CylinderMesh)
  r, h = m.radius, m.height

  # 12 nodes; coordinate values are junk — AtlasGrid replaces them.
  node_coords = Vector{Point{2,Float64}}([
    Point(0.0, 0.0), Point(1.0, 0.0), Point(2.0, 0.0),   # ring 0  (z=0)
    Point(0.0, 1.0), Point(1.0, 1.0), Point(2.0, 1.0),   # ring 1  (z=h/3)
    Point(0.0, 2.0), Point(1.0, 2.0), Point(2.0, 2.0),   # ring 2  (z=2h/3)
    Point(0.0, 3.0), Point(1.0, 3.0), Point(2.0, 3.0),   # ring 3  (z=h)
  ])

  cell_node_data = Int32[
     1, 2, 4, 5,    # C1
     2, 3, 5, 6,    # C2
     3, 1, 6, 4,    # C3  ← seam wrap
     4, 5, 7, 8,    # C4
     5, 6, 8, 9,    # C5
     6, 4, 9, 7,    # C6  ← seam wrap
     7, 8,10,11,    # C7
     8, 9,11,12,    # C8
     9, 7,12,10,    # C9  ← seam wrap
  ]
  cell_node_ptrs = Int32[1,5,9,13,17,21,25,29,33,37]
  cell_node_ids  = Gridap.Arrays.Table(cell_node_data, cell_node_ptrs)
  cell_types     = fill(Int32(1), 9)
  reffe          = Gridap.ReferenceFEs.LagrangianRefFE(Float64, QUAD, 1)
  grid = Gridap.Geometry.UnstructuredGrid(
    node_coords, cell_node_ids, [reffe], cell_types, Gridap.Geometry.Oriented())

  topo   = Gridap.Geometry.UnstructuredGridTopology(grid)
  labels = Gridap.Geometry.FaceLabeling(topo)

  # Tag bottom (both node IDs ≤ 3) and top (both > 9) edges.
  edge_to_vert = Gridap.Geometry.get_faces(topo, 1, 0)
  n_edges      = Gridap.Geometry.num_faces(topo, 1)
  for e in 1:n_edges
    vs = collect(Int, edge_to_vert[e])
    if all(v <= 3 for v in vs)
      labels.d_to_dface_to_entity[2][e] = Int32(3)
    elseif all(v > 9 for v in vs)
      labels.d_to_dface_to_entity[2][e] = Int32(4)
    end
  end
  for e in 1:n_edges
    entity = labels.d_to_dface_to_entity[2][e]
    if entity == Int32(3) || entity == Int32(4)
      for v in edge_to_vert[e]
        labels.d_to_dface_to_entity[1][v] = entity
      end
    end
  end
  Gridap.Geometry.add_tag!(labels, "bottom", [3])
  Gridap.Geometry.add_tag!(labels, "top",    [4])

  model = Gridap.Geometry.UnstructuredDiscreteModel(grid, topo, labels)

  dθ = 2π/3;  dz = h/3

  # Chart-local coordinates are the cylinder parameters (θ,z).
  # Each coarse cell k covers [θ_k, θ_{k+1}] × [z_k, z_{k+1}].
  # Corner ordering: [BL, BR, TL, TR] = [(θ_min,z_min),(θ_max,z_min),(θ_min,z_max),(θ_max,z_max)]
  # The seam cells (C3, C6, C9) have their right corners at θ = 2π (= 0 physically);
  # using θ=2π keeps the chart map smooth — topology handles the identification.
  cell_chart_coords = [
    [Point(0dθ,0dz), Point(1dθ,0dz), Point(0dθ,1dz), Point(1dθ,1dz)],  # C1
    [Point(1dθ,0dz), Point(2dθ,0dz), Point(1dθ,1dz), Point(2dθ,1dz)],  # C2
    [Point(2dθ,0dz), Point(3dθ,0dz), Point(2dθ,1dz), Point(3dθ,1dz)],  # C3 ← seam, θ_max=2π
    [Point(0dθ,1dz), Point(1dθ,1dz), Point(0dθ,2dz), Point(1dθ,2dz)],  # C4
    [Point(1dθ,1dz), Point(2dθ,1dz), Point(1dθ,2dz), Point(2dθ,2dz)],  # C5
    [Point(2dθ,1dz), Point(3dθ,1dz), Point(2dθ,2dz), Point(3dθ,2dz)],  # C6 ← seam, θ_max=2π
    [Point(0dθ,2dz), Point(1dθ,2dz), Point(0dθ,3dz), Point(1dθ,3dz)],  # C7
    [Point(1dθ,2dz), Point(2dθ,2dz), Point(1dθ,3dz), Point(2dθ,3dz)],  # C8
    [Point(2dθ,2dz), Point(3dθ,2dz), Point(2dθ,3dz), Point(3dθ,3dz)],  # C9 ← seam, θ_max=2π
  ]

  ambient_maps  = fill(CylinderChartMap(r), 9)
  metric_fields = fill(CylinderMetricField(r), 9)

  CoarseMeshInfo(model, cell_chart_coords, ambient_maps, metric_fields)
end

# ============================================================
# MobiusStripMesh
# ============================================================

"""
    MobiusStripMesh(radius=1.0, half_width=0.3)

Two-chart atlas for a Möbius strip with major radius `radius` and half-width `half_width`.
The two charts cover θ ∈ [0,π] (C1) and θ ∈ [π,2π] (C2). The half-twist is encoded
topologically: the right edge of C2 is identified with the left edge of C1 reversed.

**Why 2 cells work here (unlike the cylinder)?** The half-twist shifts the seam node
order: C2=[2,3,4,1] instead of the direct-wrap [2,1,4,3] a cylinder would need.
C2's two non-interior edges `{2,3}` and `{1,4}` are *distinct* from C1's edges, so
they become genuine boundary edges — correctly tracing the single boundary loop of
the strip.  The cylinder's direct wrap would make every edge of C2 coincide with an
edge of C1, collapsing the boundary.
"""
struct MobiusStripMesh <: CoarseMesh
  radius     :: Float64
  half_width :: Float64
  MobiusStripMesh(radius=1.0, half_width=0.3) = new(radius, half_width)
end

# ── MobiusChartMap ────────────────────────────────────────────────────────────

struct MobiusChartMap <: Field
  radius :: Float64; half_width :: Float64; theta_offset :: Float64
end
function Gridap.Arrays.evaluate!(cache, m::MobiusChartMap, x::Point)
  θ = π*(x[1] + m.theta_offset)/2
  ρ = m.radius + m.half_width*x[2]*cos(θ/2)
  Point(ρ*cos(θ), ρ*sin(θ), m.half_width*x[2]*sin(θ/2))
end
function Gridap.Arrays.return_cache(m::MobiusChartMap, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, Point{3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::MobiusChartMap, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  @inbounds for i in eachindex(xs)
    x = xs[i]
    θ = π*(x[1] + m.theta_offset)/2
    ρ = m.radius + m.half_width*x[2]*cos(θ/2)
    cache.array[i] = Point(ρ*cos(θ), ρ*sin(θ), m.half_width*x[2]*sin(θ/2))
  end
  cache.array
end

struct MobiusChartMapGrad <: Field
  radius :: Float64; half_width :: Float64; theta_offset :: Float64
end
function Gridap.Arrays.evaluate!(cache, m::MobiusChartMapGrad, x::Point)
  θ    = π*(x[1] + m.theta_offset)/2
  t    = x[2]; W = m.half_width
  ρ    = m.radius + W*t*cos(θ/2)
  dρds = -W*t*sin(θ/2)*(π/4)
  dρdt =  W*cos(θ/2)
  TensorValue{2,3,Float64}(
    dρds*cos(θ) - ρ*sin(θ)*(π/2),  dρdt*cos(θ),
    dρds*sin(θ) + ρ*cos(θ)*(π/2),  dρdt*sin(θ),
    W*t*cos(θ/2)*(π/4),             W*sin(θ/2),
  )
end
function Gridap.Arrays.return_cache(m::MobiusChartMapGrad, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, TensorValue{2,3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::MobiusChartMapGrad, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  @inbounds for i in eachindex(xs)
    x = xs[i]
    θ    = π*(x[1] + m.theta_offset)/2
    t    = x[2]; W = m.half_width
    ρ    = m.radius + W*t*cos(θ/2)
    dρds = -W*t*sin(θ/2)*(π/4)
    dρdt =  W*cos(θ/2)
    cache.array[i] = TensorValue{2,3,Float64}(
      dρds*cos(θ) - ρ*sin(θ)*(π/2),  dρdt*cos(θ),
      dρds*sin(θ) + ρ*cos(θ)*(π/2),  dρdt*sin(θ),
      W*t*cos(θ/2)*(π/4),             W*sin(θ/2),
    )
  end
  cache.array
end

Gridap.Fields.gradient(f::MobiusChartMap) =
  MobiusChartMapGrad(f.radius, f.half_width, f.theta_offset)

# ── MobiusMetricField ─────────────────────────────────────────────────────────
#
# Chart (s,t) ∈ [−1,1]², θ = π(s+offset)/2, ρ = R + W·t·cos(θ/2).
# Metric is diagonal:
#   g₁₁ = (π/4)²·W²·t² + (π/2)²·ρ²
#   g₁₂ = 0
#   g₂₂ = W²
# Derivation: |∂φ/∂s|² expands to W²t²(π/4)² + ρ²(π/2)² (cross terms cancel by
# cos² + sin² = 1 for the angular part and dρ/dt terms); ∂φ/∂s · ∂φ/∂t = 0
# (verified by direct computation); |∂φ/∂t|² = W².

struct MobiusMetricField <: Field
  radius :: Float64; half_width :: Float64; theta_offset :: Float64
end
function Gridap.Arrays.evaluate!(cache, m::MobiusMetricField, x::Point)
  θ   = π*(x[1] + m.theta_offset)/2
  ρ   = m.radius + m.half_width*x[2]*cos(θ/2)
  g11 = (π/4)^2 * m.half_width^2 * x[2]^2 + (π/2)^2 * ρ^2
  SymTensorValue{2,Float64,3}(g11, 0.0, m.half_width^2)
end
function Gridap.Arrays.return_cache(m::MobiusMetricField, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, SymTensorValue{2,Float64,3}))
end
function Gridap.Arrays.evaluate!(cache, m::MobiusMetricField, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  @inbounds for i in eachindex(xs)
    x = xs[i]
    θ   = π*(x[1] + m.theta_offset)/2
    ρ   = m.radius + m.half_width*x[2]*cos(θ/2)
    g11 = (π/4)^2 * m.half_width^2 * x[2]^2 + (π/2)^2 * ρ^2
    cache.array[i] = SymTensorValue{2,Float64,3}(g11, 0.0, m.half_width^2)
  end
  cache.array
end

struct MobiusInvMetricField <: Field
  radius :: Float64; half_width :: Float64; theta_offset :: Float64
end
function Gridap.Arrays.evaluate!(cache, m::MobiusInvMetricField, x::Point)
  θ   = π*(x[1] + m.theta_offset)/2
  ρ   = m.radius + m.half_width*x[2]*cos(θ/2)
  g11 = (π/4)^2 * m.half_width^2 * x[2]^2 + (π/2)^2 * ρ^2
  SymTensorValue{2,Float64,3}(1/g11, 0.0, 1/m.half_width^2)
end
function Gridap.Arrays.return_cache(m::MobiusInvMetricField, xs::AbstractArray{<:Point})
  CachedArray(similar(xs, SymTensorValue{2,Float64,3}))
end
function Gridap.Arrays.evaluate!(cache, m::MobiusInvMetricField, xs::AbstractArray{<:Point})
  setsize!(cache, size(xs))
  @inbounds for i in eachindex(xs)
    x = xs[i]
    θ   = π*(x[1] + m.theta_offset)/2
    ρ   = m.radius + m.half_width*x[2]*cos(θ/2)
    g11 = (π/4)^2 * m.half_width^2 * x[2]^2 + (π/2)^2 * ρ^2
    cache.array[i] = SymTensorValue{2,Float64,3}(1/g11, 0.0, 1/m.half_width^2)
  end
  cache.array
end

inverse_metric_field(f::MobiusMetricField) =
  MobiusInvMetricField(f.radius, f.half_width, f.theta_offset)

# ── get_coarse_mesh(MobiusStripMesh) ─────────────────────────────────────────

"""
    get_coarse_mesh(m::MobiusStripMesh) → CoarseMeshInfo{2}

Coarse QUAD mesh for a Möbius strip with major radius `m.radius` and half-width
`m.half_width`.

Topology (4 nodes, 2 cells). Node coordinates are junk — only connectivity matters.
Gridap Z-order per cell: BL, BR, TL, TR.

  3 - 4 - 1
  |   |   |
  1 - 2 - 3
  [C1] [C2]

Cells:
  C1 = [1,2,3,4]   BL=1 BR=2 TL=3 TR=4   θ ∈ [0,  π]
  C2 = [2,3,4,1]   BL=2 BR=3 TL=4 TR=1   θ ∈ [π, 2π]

The left edge of C1 {1,3} is identified with the right edge of C2 {3,1}: same node
set, reversed orientation — this encodes the half-twist.  The node shift means C2's
non-interior edges {2,3} and {1,4} are distinct from all of C1's edges, so 2 cells
suffice.  For the cylinder the seam would require C2=[2,1,4,3], giving C2 the *same*
four edge vertex-pairs as C1 — every edge would be counted as shared and no boundary
edges would be detected.

Local frame for both charts: (s,t) ∈ [−1,1]², s = angular direction, t = width.
Ambient maps (s,t) → (X,Y,Z), with R = major radius, W = half_width:
  C1: θ = π(s+1)/2 ∈ [0,π]
  C2: θ = π(s+3)/2 ∈ [π,2π]
  both: ((R + W·t·cos(θ/2))·cos(θ),  (R + W·t·cos(θ/2))·sin(θ),  W·t·sin(θ/2))

Seam continuity:
  Interior (θ = π):   map_C1(1, t)  = map_C2(−1, t)
  Twist    (θ = 0≡2π): map_C1(−1, t) = map_C2(1, −t)   [t ↦ −t encodes the half-twist]
"""
function get_coarse_mesh(m::MobiusStripMesh)
  R, W = m.radius, m.half_width

  # Coordinate values are unused — AtlasGrid replaces them with cell_chart_coords.
  # Any distinct values that give a valid non-degenerate mesh work here.
  node_coords = Vector{Point{2,Float64}}([
    Point(0.0, 0.0),   # 1
    Point(1.0, 0.0),   # 2
    Point(0.0, 1.0),   # 3
    Point(1.0, 1.0),   # 4
  ])

  cell_node_data = Int32[1,2,3,4,  2,3,4,1]
  cell_node_ptrs = Int32[1,5,9]
  cell_node_ids  = Gridap.Arrays.Table(cell_node_data, cell_node_ptrs)
  cell_types     = Int32[1,1]
  reffe          = Gridap.ReferenceFEs.LagrangianRefFE(Float64, QUAD, 1)
  grid = Gridap.Geometry.UnstructuredGrid(
    node_coords, cell_node_ids, [reffe], cell_types, Gridap.Geometry.NonOriented())

  topo   = Gridap.Geometry.UnstructuredGridTopology(grid)
  labels = Gridap.Geometry.FaceLabeling(topo)
  model  = Gridap.Geometry.UnstructuredDiscreteModel(grid, topo, labels)

  ref_corners       = [Point(-1.0,-1.0), Point(1.0,-1.0), Point(-1.0,1.0), Point(1.0,1.0)]
  cell_chart_coords = [ref_corners, ref_corners]
  ambient_maps      = [MobiusChartMap(R, W, 1.0), MobiusChartMap(R, W, 3.0)]
  metric_fields     = [MobiusMetricField(R, W, 1.0), MobiusMetricField(R, W, 3.0)]

  CoarseMeshInfo(model, cell_chart_coords, ambient_maps, metric_fields)
end

# ============================================================
# CubedSphereMesh
# ============================================================

"""
    CubedSphereMesh(radius=1.0)

Six-panel atlas for a cubed sphere of the given radius.
Panels are numbered 1–6 and use the gnomonic projection `CubedSphereMap(p, radius)`,
with local (α,β) coordinates in [−π/4, π/4]².
"""
struct CubedSphereMesh <: CoarseMesh
  radius :: Float64
  CubedSphereMesh(radius=1.0) = new(radius)
end

# ── CubedSphereMap ────────────────────────────────────────────────────────────
#
# Gnomonic projection for one panel of the cubed sphere.
# All 6 panels share the base map h(α,β) = (1/ρ, tanα/ρ, tanβ/ρ) with ρ = √(1+tan²α+tan²β);
# each panel permutes/negates the three components of h.
#
# _CSPHERE_PERM[p] = (j1,s1, j2,s2, j3,s3):
#   output component k = s_k · h[j_k],   k = 1,2,3
# This same table drives both the forward map and the explicit Jacobian.
#
# Jacobian derivation (Gridap convention: J[i,k] = ∂φ_k/∂x_i):
#   Base Jh columns (i=1 → ∂/∂α, i=2 → ∂/∂β):
#     col 1: (−a·sα/ρ³,  −b·sβ/ρ³)
#     col 2: ( sα·sβ/ρ³, −a·b·sβ/ρ³)
#     col 3: (−a·b·sα/ρ³, sα·sβ/ρ³)
#   where a = tanα, b = tanβ, sα = 1+a², sβ = 1+b², ρ³ = (1+a²+b²)^(3/2).
#   Panel p Jacobian: J_p[i,k] = r · s_k · Jh[i, j_k]

const _CSPHERE_PERM = (
    (1, 1, 2, 1, 3, 1),   # panel 1: φ = r*(  h₁,  h₂,  h₃)
    (3,-1, 2, 1, 1, 1),   # panel 2: φ = r*( -h₃,  h₂,  h₁)
    (2,-1, 1, 1, 3, 1),   # panel 3: φ = r*( -h₂,  h₁,  h₃)
    (1,-1, 3, 1, 2, 1),   # panel 4: φ = r*( -h₁,  h₃,  h₂)
    (2,-1, 3, 1, 1,-1),   # panel 5: φ = r*( -h₂,  h₃, -h₁)
    (3,-1, 1,-1, 2, 1),   # panel 6: φ = r*( -h₃, -h₁,  h₂)
)

function _csphere_eval(panel::Int, r::Float64, x::Point{2})
  a, b = tan(x[1]), tan(x[2])
  ρ    = sqrt(1 + a^2 + b^2)
  h    = (1/ρ, a/ρ, b/ρ)
  j1,s1, j2,s2, j3,s3 = _CSPHERE_PERM[panel]
  r * Point(s1*h[j1], s2*h[j2], s3*h[j3])
end

function _csphere_jac(panel::Int, r::Float64, x::Point{2})
  a, b  = tan(x[1]), tan(x[2])
  ρ2    = 1 + a^2 + b^2
  ρ     = sqrt(ρ2); ρ3 = ρ2 * ρ
  sa, sb = 1+a^2, 1+b^2
  # Base Jh columns: c[k] = (Jh[1,k], Jh[2,k]) scaled by ρ³
  c = ((-a*sa, -b*sb), (sa*sb, -a*b*sb), (-a*b*sa, sa*sb))
  j1,s1, j2,s2, j3,s3 = _CSPHERE_PERM[panel]
  (r/ρ3) * TensorValue{2,3,Float64}(
    s1*c[j1][1], s1*c[j1][2],
    s2*c[j2][1], s2*c[j2][2],
    s3*c[j3][1], s3*c[j3][2],
  )
end

struct CubedSphereMap <: Field
  panel  :: Int
  radius :: Float64
end

Gridap.Arrays.evaluate!(cache, m::CubedSphereMap, x::Point{2}) =
  _csphere_eval(m.panel, m.radius, x)
function Gridap.Arrays.return_cache(m::CubedSphereMap, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, Point{3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereMap, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  p, r = m.panel, m.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_eval(p, r, xs[i])
  end
  cache.array
end

struct CubedSphereMapGrad <: Field
  panel  :: Int
  radius :: Float64
end

Gridap.Arrays.evaluate!(cache, m::CubedSphereMapGrad, x::Point{2}) =
  _csphere_jac(m.panel, m.radius, x)
function Gridap.Arrays.return_cache(m::CubedSphereMapGrad, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, TensorValue{2,3,Float64}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereMapGrad, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  p, r = m.panel, m.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_jac(p, r, xs[i])
  end
  cache.array
end

Gridap.Fields.gradient(m::CubedSphereMap) = CubedSphereMapGrad(m.panel, m.radius)

# ── CubedSphereMetricField ────────────────────────────────────────────────────
#
# Pullback metric g = JᵀJ for the gnomonic projection.  All 6 panels share the
# same formula because permuting/negating components is an isometry: sign factors
# cancel in JᵀJ (s² = 1) and the permutation maps {c[j1],c[j2],c[j3]} to
# {c[1],c[2],c[3]} in a different order but the same set, giving identical g.
#
# With a = tanα, b = tanβ, sa = 1+a², sb = 1+b², ρ² = 1+a²+b²:
#   g₁₁ = r²·sa²·sb/ρ⁴
#   g₁₂ = −r²·a·b·sa·sb/ρ⁴
#   g₂₂ = r²·sa·sb²/ρ⁴
#
# Inverse (from 2×2 formula, det g = r⁴·sa²·sb²/ρ⁶):
#   g⁻¹ = (ρ²/(r²·sa·sb)) · [[sb, a·b], [a·b, sa]]

function _csphere_metric(r::Float64, x::Point{2})
  a, b  = tan(x[1]), tan(x[2])
  sa, sb = 1 + a^2, 1 + b^2
  ρ4    = (1 + a^2 + b^2)^2
  c     = r^2 * sa * sb / ρ4
  SymTensorValue{2,Float64,3}(c*sa, -c*a*b, c*sb)
end

function _csphere_inv_metric(r::Float64, x::Point{2})
  a, b  = tan(x[1]), tan(x[2])
  sa, sb = 1 + a^2, 1 + b^2
  ρ2    = 1 + a^2 + b^2
  c     = ρ2 / (r^2 * sa * sb)
  SymTensorValue{2,Float64,3}(c*sb, c*a*b, c*sa)
end

struct CubedSphereMetricField <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CubedSphereMetricField, x::Point{2}) =
  _csphere_metric(m.radius, x)
function Gridap.Arrays.return_cache(m::CubedSphereMetricField, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, SymTensorValue{2,Float64,3}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereMetricField, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  r = m.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_metric(r, xs[i])
  end
  cache.array
end

struct CubedSphereInvMetricField <: Field
  radius :: Float64
end
Gridap.Arrays.evaluate!(cache, m::CubedSphereInvMetricField, x::Point{2}) =
  _csphere_inv_metric(m.radius, x)
function Gridap.Arrays.return_cache(m::CubedSphereInvMetricField, xs::AbstractArray{<:Point{2}})
  CachedArray(similar(xs, SymTensorValue{2,Float64,3}))
end
function Gridap.Arrays.evaluate!(cache, m::CubedSphereInvMetricField, xs::AbstractArray{<:Point{2}})
  setsize!(cache, size(xs))
  r = m.radius
  @inbounds for i in eachindex(xs)
    cache.array[i] = _csphere_inv_metric(r, xs[i])
  end
  cache.array
end

inverse_metric_field(f::CubedSphereMetricField) = CubedSphereInvMetricField(f.radius)

# ── get_coarse_mesh(CubedSphereMesh) ─────────────────────────────────────────

"""
    get_coarse_mesh(m::CubedSphereMesh) → CoarseMeshInfo{2}

Coarse QUAD mesh for a cubed sphere with radius `m.radius`.

Topology (8 nodes, 6 cells). Node coordinates are junk — only connectivity
matters. Gridap Z-order per cell: BL, BR, TL, TR.

       x=−2                              x=2
  y=2   5──────────────────────────────────6   C2
        │╲              C4               ╱│
  y=1   │  8──────────────────────────7  │
        │  │                          │  │
        │C6│         C5               │C3│
        │  │                          │  │
  y=−1  │  1──────────────────────────2  │
        │╱              C1               ╲│
  y=−2  3──────────────────────────────────4

Nodes:
  1=(−1,−1)  2=(1,−1)  3=(−2,−2)  4=(2,−2)
  5=(−2, 2)  6=(2, 2)  7=(1, 1)   8=(−1, 1)

Cells (same connectivity as _CCAM_panel_wise_node_ids):
  C1 = [1,2,3,4]   BL=1 BR=2 TL=3 TR=4
  C2 = [3,4,5,6]   BL=3 BR=4 TL=5 TR=6
  C3 = [2,7,4,6]   BL=2 BR=7 TL=4 TR=6
  C4 = [8,5,7,6]   BL=8 BR=5 TL=7 TR=6
  C5 = [1,8,2,7]   BL=1 BR=8 TL=2 TR=7
  C6 = [1,3,8,5]   BL=1 BR=3 TL=8 TR=5

The 8 nodes correspond to the 8 corners of the cube. Node sharing encodes the
12 shared edges between the 6 faces.

Local frame for all charts: (α,β) ∈ [−π/4, π/4]².
Ambient maps: `CubedSphereMap(p, radius)` for p = 1 … 6 (gnomonic projection).

Face labels: the cubed sphere is a closed manifold — no topological boundary.
FaceLabeling(topo) assigns entity 1 ("interior") to all faces, which also
satisfies the p4est precondition that all face entity ids be positive.
"""
function get_coarse_mesh(m::CubedSphereMesh)
  # Coordinate values are unused — AtlasGrid replaces them with cell_chart_coords.
  # Any distinct values that give a valid non-degenerate mesh work here.
  node_coords = Vector{Point{2,Float64}}([
    Point(-1.0, -1.0),   # 1
    Point( 1.0, -1.0),   # 2
    Point(-2.0, -2.0),   # 3
    Point( 2.0, -2.0),   # 4
    Point(-2.0,  2.0),   # 5
    Point( 2.0,  2.0),   # 6
    Point( 1.0,  1.0),   # 7
    Point(-1.0,  1.0),   # 8
  ])

  cell_node_data = Int32[1,2,3,4, 3,4,5,6, 2,7,4,6, 8,5,7,6, 1,8,2,7, 1,3,8,5]
  cell_node_ptrs = Int32[1,5,9,13,17,21,25]
  cell_node_ids  = Gridap.Arrays.Table(cell_node_data, cell_node_ptrs)
  cell_types     = Int32[1,1,1,1,1,1]
  reffe          = Gridap.ReferenceFEs.LagrangianRefFE(Float64, QUAD, 1)
  grid = Gridap.Geometry.UnstructuredGrid(
    node_coords, cell_node_ids, [reffe], cell_types, Gridap.Geometry.Oriented())

  topo   = Gridap.Geometry.UnstructuredGridTopology(grid)
  labels = Gridap.Geometry.FaceLabeling(topo)   # all entities = 1 (closed manifold)
  model  = Gridap.Geometry.UnstructuredDiscreteModel(grid, topo, labels)

  panel_corners = [
    Point(-cube_half_edge, -cube_half_edge),   # BL
    Point( cube_half_edge, -cube_half_edge),   # BR
    Point(-cube_half_edge,  cube_half_edge),   # TL
    Point( cube_half_edge,  cube_half_edge),   # TR
  ]
  cell_chart_coords = fill(panel_corners, n_panels)
  ambient_maps      = [CubedSphereMap(p, m.radius) for p in 1:n_panels]
  metric_fields     = fill(CubedSphereMetricField(m.radius), n_panels)

  CoarseMeshInfo(model, cell_chart_coords, ambient_maps, metric_fields)
end

# ============================================================
# inverse_metric_field: generic fallback
# ============================================================

"""
    inverse_metric_field(f::Field) → Field

Return a `Field` that evaluates the pointwise inverse of the metric `f`.
Concrete overrides for `CylinderMetricField`, `MobiusMetricField`, and
`CubedSphereMetricField` return explicit analytic `InvMetricField` types.
The generic fallback wraps `Operation(inv)` and uses generic matrix inversion.
"""
inverse_metric_field(f::Field) = Operation(inv)(f)
