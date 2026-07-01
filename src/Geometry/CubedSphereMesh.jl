# ============================================================
# CubedSphereMesh
# ============================================================

"""
NPANELS

The number of panels of the cubed sphere manifold is always six.
This constant is used throughout the geometry construction
"""

const NPANELS = 6


"""
CUBE_HALF_EDGE

The value of the half edge of the cube is π/4. That is, every panel is the square
[-π/4,π/4]^2.
Currently the forward maps are only supported for this formulation of the panels.
"""

const CUBE_HALF_EDGE = π/4

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
