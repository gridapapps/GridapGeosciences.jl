
struct AnalyticalMapCubedSphereTriangulation{T} <: Triangulation{2,3}
  cubed_sphere_model::T
end

# Triangulation API

# Delegating to the underlying face Triangulation

Gridap.Geometry.get_cell_coordinates(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_coordinates(get_grid(trian.cubed_sphere_model.cubed_sphere_linear_model))

Gridap.Geometry.get_reffes(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_reffes(get_grid(trian.cubed_sphere_model.cubed_sphere_linear_model))

Gridap.Geometry.get_cell_type(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_type(get_grid(trian.cubed_sphere_model.cubed_sphere_linear_model))

Gridap.Geometry.get_node_coordinates(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Helpers.get_node_coordinates(get_grid(trian.cubed_sphere_model.cubed_sphere_linear_model))

Gridap.Geometry.get_cell_node_ids(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Helpers.get_cell_node_ids(get_grid(trian.cubed_sphere_model.cubed_sphere_linear_model))

Gridap.Geometry.get_cell_map(trian::AnalyticalMapCubedSphereTriangulation) = trian.cubed_sphere_model.cell_map

Gridap.Geometry.get_grid(trian::AnalyticalMapCubedSphereTriangulation) = get_grid(trian.cubed_sphere_model.cubed_sphere_linear_model)
