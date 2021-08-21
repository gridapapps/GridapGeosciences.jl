
struct AnalyticalMapCubedSphereTriangulation{T} <: Triangulation{2,3}
  cell_map::T
  btrian::Triangulation{2,3}
end

# Triangulation API

# Delegating to the underlying face Triangulation

Gridap.Geometry.get_cell_coordinates(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_coordinates(trian.btrian)

Gridap.Geometry.get_reffes(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_reffes(trian.btrian)

Gridap.Geometry.get_cell_type(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_type(trian.btrian)

Gridap.Geometry.get_node_coordinates(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Helpers.get_node_coordinates(trian.btrian)

Gridap.Geometry.get_cell_node_ids(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Helpers.get_cell_node_ids(trian.btrian)

Gridap.Geometry.get_cell_map(trian::AnalyticalMapCubedSphereTriangulation) = trian.cell_map

# Genuine methods

Gridap.Geometry.TriangulationStyle(::Type{<:AnalyticalMapCubedSphereTriangulation}) = SubTriangulation()

Gridap.Geometry.get_background_triangulation(trian::AnalyticalMapCubedSphereTriangulation) =
      Gridap.Geometry.get_background_triangulation(trian.btrian)

Gridap.Geometry.get_cell_to_bgcell(trian::AnalyticalMapCubedSphereTriangulation) = get_cell_to_bgcell(trian.btrian)

function Gridap.Geometry.get_cell_to_bgcell(
  trian_in::AnalyticalMapCubedSphereTriangulation,
  trian_out::AnalyticalMapCubedSphereTriangulation)
  Gridap.Helpers.@notimplemented
end

function Gridap.Geometry.is_included(
  trian_in::AnalyticalMapCubedSphereTriangulation,
  trian_out::AnalyticalMapCubedSphereTriangulation)
  Gridap.Helpers.@notimplemented
end

function Gridap.Geometry.get_cell_ref_map(trian::AnalyticalMapCubedSphereTriangulation)
  get_cell_ref_map(trian.btrian)
end

function Gridap.Geometry.get_cell_ref_map(
  trian_in::AnalyticalMapCubedSphereTriangulation,
  trian_out::AnalyticalMapCubedSphereTriangulation)
  Gridap.Helpers.@notimplemented
end
