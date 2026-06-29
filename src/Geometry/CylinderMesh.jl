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
