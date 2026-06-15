
function get_panel_ids(atrian::AdaptedTriangulation)
  get_panel_ids(atrian.trian)
end

function get_panel_ids(trian::Triangulation)
  model = get_background_model(trian)
  panel_ids = get_panel_ids(model)
  get_panel_ids(trian,panel_ids)
end

function get_panel_ids(trian::BodyFittedTriangulation,panel_ids::AbstractArray)
  # println("body fitted triangulation")
  return panel_ids
end

function get_panel_ids(trian::BoundaryTriangulation,panel_ids::AbstractArray)
  # println("boundary triangulation")
  get_face_panel_ids(trian,panel_ids)
end

function get_panel_ids(trian::TriangulationView,panel_ids::AbstractArray)
  # println("trian view panel ids")
  get_face_panel_ids(trian,panel_ids)
end

function get_face_panel_ids(trian,panel_ids)
  panel_model = get_background_model(trian)
  Dc = num_cell_dims(panel_model)

  glue = get_glue(trian,Val(Dc))
  face_2_cell = glue.tface_to_mface
  face_panel_ids = panel_ids[face_2_cell]
  return face_panel_ids
end

function get_panel_ids(ptrian::PatchTriangulation)
  # println("patch triangulation")
  bg_model = get_background_model(ptrian)
  get_panel_ids(bg_model)
end
