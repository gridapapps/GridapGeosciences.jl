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
