
# BEGIN: AtlasDiscreteModel-related constructors
function Geometry.BoundaryTriangulation(
  model::AtlasDiscreteModel,
  bgface_to_mask::AbstractVector{Bool},
  bgface_to_lcell::AbstractVector{<:Integer}
  )
  face_to_bgface =  findall(bgface_to_mask)
  Geometry.BoundaryTriangulation(model,face_to_bgface,bgface_to_lcell)
end

# This function is almost equivalent to its counterpart in Gridap.Geometry.
# The main difference is that the face_grid depends on bgface_to_lcell, while
# the one in Gridap.Geometry does not.
function Geometry.BoundaryTriangulation(
  model::AtlasDiscreteModel,
  face_to_bgface::AbstractVector{<:Integer},
  bgface_to_lcell::AbstractVector{<:Integer})
  D = num_cell_dims(model)
  topo = get_grid_topology(model)
  face_grid = Grid(ReferenceFE{D-1},model,face_to_bgface,bgface_to_lcell)
  cell_grid = get_grid(model)
  glue = FaceToCellGlue(topo,cell_grid,face_grid,face_to_bgface,bgface_to_lcell)
  trian = BodyFittedTriangulation(model,face_grid,face_to_bgface)
  BoundaryTriangulation(trian,glue)
end

