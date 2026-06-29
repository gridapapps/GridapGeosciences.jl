# ============================================================
# ExtrudedCubedSphereWithThicknessMesh
# ============================================================

"""
    ExtrudedCubedSphereWithThicknessMesh(radius=1.0, thickness=0.1)

Six-panel atlas built by reusing the 2D cubed-sphere topology from
`CubedSphereMesh` and replacing the ambient maps with the 3D thickness maps
`CubedSphereWithThicknessMap`.
"""
struct ExtrudedCubedSphereWithThicknessMesh <: CoarseMesh
  radius :: Float64
  thickness :: Float64
  ExtrudedCubedSphereWithThicknessMesh(radius=1.0, thickness=0.1) = new(radius, thickness)
end

# ── get_coarse_mesh(ExtrudedCubedSphereWithThicknessMesh) ────────────────────

function get_coarse_mesh(m::ExtrudedCubedSphereWithThicknessMesh)
  cubed_sphere_mesh_info = get_coarse_mesh(CubedSphereMesh(m.radius))
  ambient_maps      = [CubedSphereWithThicknessMap(p, m.radius, m.thickness) for p in 1:NPANELS]
  metric_fields     = fill(CubedSphereWithThicknessMetric(m.radius, m.thickness), NPANELS)
  CoarseMeshInfo(cubed_sphere_mesh_info.model, cubed_sphere_mesh_info.cell_chart_coords, ambient_maps, metric_fields)
end
