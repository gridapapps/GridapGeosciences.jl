# CoarseMeshes.jl
#
# Library of canonical coarse meshes for AtlasGrid / AtlasDiscreteModel.
# Each shape is represented by a concrete subtype of CoarseMesh that
# carries the geometric parameters (radius, height, …).  Calling
# get_coarse_mesh(shape) returns a CoarseMeshInfo bundling the coarse
# DiscreteModel (with face labels), per-cell local-frame corner coordinates,
# and default ambient maps.
#
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
                    concrete analytic types (e.g. `CubedSphereMetric`);
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

  ambient_maps  = fill(CylinderMap(r), 9)
  metric_fields = fill(CylinderMetric(r), 9)

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
  ambient_maps      = [MobiusMap(R, W, 1.0), MobiusMap(R, W, 3.0)]
  metric_fields     = [MobiusMetric(R, W, 1.0), MobiusMetric(R, W, 3.0)]

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
    Point(-CUBE_HALF_EDGE, -CUBE_HALF_EDGE),   # BL
    Point( CUBE_HALF_EDGE, -CUBE_HALF_EDGE),   # BR
    Point(-CUBE_HALF_EDGE,  CUBE_HALF_EDGE),   # TL
    Point( CUBE_HALF_EDGE,  CUBE_HALF_EDGE),   # TR
  ]
  cell_chart_coords = fill(panel_corners, NPANELS)
  ambient_maps      = [CubedSphereMap(p, m.radius) for p in 1:NPANELS]
  metric_fields     = fill(CubedSphereMetric(m.radius), NPANELS)

  CoarseMeshInfo(model, cell_chart_coords, ambient_maps, metric_fields)
end

"""
"""
struct CubedSphereWithThicknessMesh <: CoarseMesh
  radius :: Float64
  thickness :: Float64
  CubedSphereWithThicknessMesh(radius=1.0, thickness=0.1) = new(radius, thickness)
end

"""
"""
function get_coarse_mesh(m::CubedSphereWithThicknessMesh)
  # Coordinate values are unused — AtlasGrid replaces them with cell_chart_coords.
  # Any distinct values that give a valid non-degenerate mesh work here.
  node_coords = Vector{Point{3,Float64}}([
    Point(0.0, -1.0, -1.0),   # 1
    Point(1.0, -1.0, -1.0),   # 2
    Point(0.0,  1.0, -1.0),   # 3
    Point(1.0,  1.0, -1.0),   # 4
    Point(0.0, -2.0, -2.0),   # 5
    Point(1.0, -2.0, -2.0),   # 6
    Point(0.0,  2.0, -2.0),   # 7
    Point(1.0,  2.0, -2.0),   # 8
    Point(0.0, -2.0,  2.0),   # 9
    Point(1.0, -2.0,  2.0),   # 10
    Point(0.0,  2.0,  2.0),   # 11
    Point(1.0,  2.0,  2.0),   # 12
    Point(0.0,  1.0,  1.0),   # 13
    Point(1.0,  1.0,  1.0),   # 14
    Point(0.0, -1.0,  1.0),   # 15
    Point(1.0, -1.0,  1.0),   # 16
  ])

  cell_node_data = Int32[1,2,3,4,5,6,7,8, 
                         5,6,7,8,9,10,11,12, 
                         3,4,13,14,7,8,11,12, 
                         15,16,9,10,13,14,11,12, 
                         1,2,15,16,3,4,13,14, 
                         1,2,5,6,15,16,9,10]
  cell_node_ptrs = Int32[1,9,17,25,33,41,49]
  cell_node_ids  = Gridap.Arrays.Table(cell_node_data, cell_node_ptrs)
  cell_types     = Int32[1,1,1,1,1,1]
  reffe          = Gridap.ReferenceFEs.LagrangianRefFE(Float64, HEX, 1)
  grid = Gridap.Geometry.UnstructuredGrid(
    node_coords, cell_node_ids, [reffe], cell_types, Gridap.Geometry.Oriented())

  topo   = Gridap.Geometry.UnstructuredGridTopology(grid)

  # Four entity ids: interior=1, bottom_boundary=2, top_boundary=3, intermediate_boundary=4.
  # Odd-indexed vertices are on the bottom (inner) shell; even-indexed on the top (outer) shell.
  # Edges and 2D facets are tagged by their vertex parities; mixed-parity ones are intermediate.
  labels = Gridap.Geometry.FaceLabeling(topo)

  for v in 1:Gridap.Geometry.num_faces(topo, 0)
    labels.d_to_dface_to_entity[1][v] = isodd(v) ? Int32(2) : Int32(3)
  end

  edge_to_vert = Gridap.Geometry.get_faces(topo, 1, 0)
  for e in 1:Gridap.Geometry.num_faces(topo, 1)
    vs = edge_to_vert[e]
    if all(isodd(v) for v in vs)
      labels.d_to_dface_to_entity[2][e] = Int32(2)
    elseif all(iseven(v) for v in vs)
      labels.d_to_dface_to_entity[2][e] = Int32(3)
    else
      labels.d_to_dface_to_entity[2][e] = Int32(4)
    end
  end

  facet_to_vert = Gridap.Geometry.get_faces(topo, 2, 0)
  for f in 1:Gridap.Geometry.num_faces(topo, 2)
    vs = facet_to_vert[f]
    if all(isodd(v) for v in vs)
      labels.d_to_dface_to_entity[3][f] = Int32(2)
    elseif all(iseven(v) for v in vs)
      labels.d_to_dface_to_entity[3][f] = Int32(3)
    else
      labels.d_to_dface_to_entity[3][f] = Int32(4)
    end
  end

  Gridap.Geometry.add_tag!(labels, "bottom_boundary",       [2])
  Gridap.Geometry.add_tag!(labels, "top_boundary",          [3])
  Gridap.Geometry.add_tag!(labels, "intermediate_boundary", [4])

  model  = Gridap.Geometry.UnstructuredDiscreteModel(grid, topo, labels)

  panel_corners = [
    Point(0.0, -CUBE_HALF_EDGE, -CUBE_HALF_EDGE),   # BLB
    Point(1.0, -CUBE_HALF_EDGE, -CUBE_HALF_EDGE),   # BLT
    Point(0.0, CUBE_HALF_EDGE, -CUBE_HALF_EDGE),    # BRB
    Point(1.0, CUBE_HALF_EDGE, -CUBE_HALF_EDGE),    # BRT
    Point(0.0, -CUBE_HALF_EDGE,  CUBE_HALF_EDGE),   # TLN
    Point(1.0, -CUBE_HALF_EDGE,  CUBE_HALF_EDGE),   # TLT
    Point(0.0,  CUBE_HALF_EDGE,  CUBE_HALF_EDGE),   # TRB
    Point(1.0,  CUBE_HALF_EDGE,  CUBE_HALF_EDGE),   # TRT
  ]
  cell_chart_coords = fill(panel_corners, NPANELS)
  ambient_maps      = [CubedSphereWithThicknessMap(p, m.radius, m.thickness) for p in 1:NPANELS]
  metric_fields     = fill(CubedSphereWithThicknessMetric(m.radius, m.thickness), NPANELS)

  CoarseMeshInfo(model, cell_chart_coords, ambient_maps, metric_fields)
end

struct ExtrudedCubedSphereWithThicknessMesh <: CoarseMesh
  radius :: Float64
  thickness :: Float64
  ExtrudedCubedSphereWithThicknessMesh(radius=1.0, thickness=0.1) = new(radius, thickness)
end

function get_coarse_mesh(m::ExtrudedCubedSphereWithThicknessMesh)
  cubed_sphere_mesh_info = get_coarse_mesh(CubedSphereMesh(m.radius))  
  ambient_maps      = [CubedSphereWithThicknessMap(p, m.radius, m.thickness) for p in 1:NPANELS]
  metric_fields     = fill(CubedSphereWithThicknessMetric(m.radius, m.thickness), NPANELS)
  CoarseMeshInfo(cubed_sphere_mesh_info.model, cubed_sphere_mesh_info.cell_chart_coords, ambient_maps, metric_fields)
end
