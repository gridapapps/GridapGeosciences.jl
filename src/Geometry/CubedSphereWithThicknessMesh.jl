# ============================================================
# CubedSphereWithThicknessMesh
# ============================================================

"""
    CubedSphereWithThicknessMesh(radius=1.0, thickness=0.1)

Six-panel 3D atlas for a cubed sphere shell with inner radius `radius` and
radial thickness `thickness`.  Uses hexahedral cells (one per panel).
"""
struct CubedSphereWithThicknessMesh <: CoarseMesh
  radius :: Float64
  thickness :: Float64
  CubedSphereWithThicknessMesh(radius=1.0, thickness=0.1) = new(radius, thickness)
end

# ── get_coarse_mesh(CubedSphereWithThicknessMesh) ────────────────────────────

"""
    get_coarse_mesh(m::CubedSphereWithThicknessMesh) → CoarseMeshInfo{3}

Coarse HEX mesh for a cubed sphere shell with radius `m.radius` and thickness
`m.thickness`.

Topology (16 nodes, 6 cells). Node coordinates are junk — only connectivity
matters.

Four entity ids: interior=1, bottom_boundary=2, top_boundary=3,
intermediate_boundary=4.  Odd-indexed vertices sit on the inner (bottom) shell;
even-indexed on the outer (top) shell.  Edges and 2D facets are tagged by their
vertex parities; mixed-parity ones are intermediate.
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
