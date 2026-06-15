
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

################################################################################
"""
get the facet normal vector on the sphere
Two different methods:
  1. push normal vector to the face in the reference space of the cell (Gridap's get_facet_normal approach)
  2. pushing the normal vector to the face in the parametric space in which the face is embedded (Santi's formula)
"""

function pushforward_normal(
      trian::BoundaryTriangulation{Dct,Da,<:BFTATDMIM{Dct,Dcm,Da,G,A,P,C,O}}) where {Dct,Da,Dcm,G,A,P,C,O} 
   pushforward_parametric_normal(trian)
end      

## This piece of code is replicated from Gridap's get_facet_normal(::BoundaryTriangulation)
## We had to replicate it because now this piece of code is not available in its own function

function _get_reference_normal(
      trian::BoundaryTriangulation{Dct,Da,<:BFTATDMIM{Dct,Dcm,Da,G,A,P,C,O}}) where {Dct,Da,Dcm,G,A,P,C,O} 
  # Reference normal
  function f(p)
    lface_to_n = get_facet_normal(p)
    lface_to_pindex_to_perm = get_face_vertex_permutations(p,num_cell_dims(p)-1)
    nlfaces = length(lface_to_n)
    lface_pindex_to_n = [ fill(lface_to_n[lface],length(lface_to_pindex_to_perm[lface])) for lface in 1:nlfaces ]
    lface_pindex_to_n
  end
  bgmodel = get_background_model(trian)
  cell_grid = get_grid(bgmodel)   
  ptops = map(get_polytope,get_reffes(cell_grid)) #fill(QUAD,num_cells(cell_grid))
  ctype_lface_pindex_to_nref = map(f, ptops)
  # ctype_lface_pindex_to_nref = map(f, get_polytopes(cell_grid))

  face_to_nref = FaceCompressedVector(ctype_lface_pindex_to_nref,trian.glue)
  face_s_nref = lazy_map(constant_field,face_to_nref)
  face_s_nref_cf = GenericCellField(face_s_nref, trian, ReferenceDomain())
end

"""
pushes the normal vector to the face in 
the reference space of the cell to ambient 
space
"""
function pushforward_reference_normal(
      trian::BoundaryTriangulation{Dct,Da,<:BFTATDMIM{Dct,Dcm,Da,G,A,P,C,O}}) where {Dct,Da,Dcm,G,A,P,C,O} 
  face_s_nref_cf = _get_reference_normal(trian)
  ambient_map = AmbientMapCellField(trian)
  face_q_invJt = pinvJt∘∇(ambient_map)
  face_s_n = Operation(push_normal)(face_q_invJt,face_s_nref_cf)
end     

"""
pushes the normal vector to the face in the parametric space 
in which the face is embedded to ambient space
"""
function pushforward_parametric_normal(
    trian::BoundaryTriangulation{Dct,Da,<:BFTATDMIM{Dct,Dcm,Da,G,A,P,C,O}}) where {Dct,Da,Dcm,G,A,P,C,O} 
  n_parametric = get_normal_vector(trian)
  inv_cf = InvMetricCellField(trian)
  ambient_map_cf = AmbientMapCellField(trian)
  J_cf = transpose∘∇(ambient_map_cf)
  _n_mapped = J_cf⋅(inv_cf⋅n_parametric)
  ff = Operation(sqrt)(n_parametric⋅(inv_cf⋅ n_parametric))
  n_mapped = _n_mapped/ff
end 

"""
pullback of the area form
returns |Jg^{-1} ̂n|
"""
function pullback_area_form(
        trian::BoundaryTriangulation{Dct,Da,<:BFTATDMIM{Dct,Dcm,Da,G,A,P,C,O}}) where {Dct,Da,Dcm,G,A,P,C,O}
    inv_metric_cf = InvMetricCellField(trian)
    ambient_map_cf = AmbientMapCellField(trian)
    J_cf = transpose∘∇(ambient_map_cf)
    n_parametric = get_normal_vector(trian)
    Operation(norm)(J_cf⋅(inv_metric_cf ⋅n_parametric))
end