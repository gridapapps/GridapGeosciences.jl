
struct AnalyticalMapCubedSphereTriangulation{T} <: Triangulation{2,3}
  model::T
end

const CSDTT = Union{Gridap.Geometry.BodyFittedTriangulation{2,3},AnalyticalMapCubedSphereTriangulation}

# Triangulation API
function Gridap.Geometry.get_facet_normal(trian::CSDTT)
  function _unit_outward_normal(v::Gridap.Fields.MultiValue{Tuple{2,3}},sign_flip::Bool)
    n1 = v[1,2]*v[2,3] - v[1,3]*v[2,2]
    n2 = v[1,3]*v[2,1] - v[1,1]*v[2,3]
    n3 = v[1,1]*v[2,2] - v[1,2]*v[2,1]
    n = VectorValue(n1,n2,n3)
    (-1)^sign_flip*n/norm(n)
  end

  # Get the Jacobian of the cubed sphere mesh
  map   = get_cell_map(trian)
  Jt    = lazy_map(∇,map)

  # Get the index of the panel for each element
  fl = get_face_labeling(trian.model)
  panel_id  = fl.d_to_dface_to_entity[3]
  sign_flip = [panel_id[i] == 25 || panel_id[i] == 21 || panel_id[i] == 24 for i=1:length(panel_id)]
  fsign_flip = lazy_map(Gridap.Fields.ConstantField,sign_flip)

  lazy_map(Operation(_unit_outward_normal),Jt,fsign_flip)
end

function Gridap.CellData.get_normal_vector(trian::CSDTT)
  cell_normal = Gridap.Geometry.get_facet_normal(trian)
  Gridap.CellData.GenericCellField(cell_normal,trian,ReferenceDomain())
end


# Delegating to the underlying face Triangulation

Gridap.Geometry.get_cell_coordinates(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_coordinates(get_grid(trian.model.cubed_sphere_linear_model))

Gridap.Geometry.get_reffes(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_reffes(get_grid(trian.model.cubed_sphere_linear_model))

Gridap.Geometry.get_cell_type(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_type(get_grid(trian.model.cubed_sphere_linear_model))

Gridap.Geometry.get_node_coordinates(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_node_coordinates(get_grid(trian.model.cubed_sphere_linear_model))

Gridap.Geometry.get_cell_node_ids(trian::AnalyticalMapCubedSphereTriangulation) = Gridap.Geometry.get_cell_node_ids(get_grid(trian.model.cubed_sphere_linear_model))

Gridap.Geometry.get_cell_map(trian::AnalyticalMapCubedSphereTriangulation) = trian.model.cell_map

Gridap.Geometry.get_grid(trian::AnalyticalMapCubedSphereTriangulation) = get_grid(trian.model.cubed_sphere_linear_model)

Gridap.Geometry.get_background_model(trian::AnalyticalMapCubedSphereTriangulation) = trian.model

function Gridap.Geometry.get_glue(a::AnalyticalMapCubedSphereTriangulation,D::Val{2})
  nc=num_cells(a.model)
  tface_to_mface=Gridap.Fields.IdentityVector(nc)
  tface_to_mface_map=Fill(Gridap.Fields.GenericField(identity),nc)
  mface_to_tface=tface_to_mface
  Gridap.Geometry.FaceToFaceGlue(tface_to_mface,tface_to_mface_map,mface_to_tface)
end
