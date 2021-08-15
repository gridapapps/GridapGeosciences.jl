
struct CubedSphereTriangulation <: Triangulation{2,3}
  cell_map
  btrian::BoundaryTriangulation{2,3}
end

# Triangulation API

# Delegating to the underlying face Triangulation

Gridap.Geometry.get_cell_coordinates(trian::CubedSphereTriangulation) = trian.cell_map.args[1]

Gridap.Geometry.get_reffes(trian::CubedSphereTriangulation) = Gridap.Geometry.get_reffes(trian.btrian)

Gridap.Geometry.get_cell_type(trian::CubedSphereTriangulation) = Gridap.Geometry.get_cell_type(trian.btrian)

Gridap.Geometry.get_node_coordinates(trian::CubedSphereTriangulation) = Gridap.Helpers.@notimplemented

Gridap.Geometry.get_cell_node_ids(trian::CubedSphereTriangulation) = Gridap.Helpers.@notimplemented

Gridap.Geometry.get_cell_map(trian::CubedSphereTriangulation) = trian.cell_map

# Genuine methods

Gridap.Geometry.TriangulationStyle(::Type{<:CubedSphereTriangulation}) = SubTriangulation()

Gridap.Geometry.get_background_triangulation(trian::CubedSphereTriangulation) =
      get_background_triangulation(trian.btrian)

Gridap.Geometry.get_cell_to_bgcell(trian::CubedSphereTriangulation) = get_cell_to_bgcell(trian.btrian)

function Gridap.Geometry.get_cell_to_bgcell(
  trian_in::CubedSphereTriangulation,
  trian_out::CubedSphereTriangulation)
  Gridap.Helpers.@notimplemented
end

function Gridap.Geometry.is_included(
  trian_in::CubedSphereTriangulation,
  trian_out::CubedSphereTriangulation)
  Gridap.Helpers.@notimplemented
end

function Gridap.Geometry.get_facet_normal(trian::CubedSphereTriangulation)
  Gridap.Helpers.@notimplemented
end

function Gridap.Geometry.get_cell_ref_map(trian::CubedSphereTriangulation)
  get_cell_ref_map(trian.btrian)
end

function Gridap.Geometry.get_cell_ref_map(
  trian_in::CubedSphereTriangulation,
  trian_out::CubedSphereTriangulation)
  Gridap.Helpers.@notimplemented
end
