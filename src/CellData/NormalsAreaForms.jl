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


function pushforward_normal(trian::SkeletonTriangulation{Dc,Dp,
                                                         <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}) where {Dc,Dp}
  plus = pushforward_normal(trian.plus)
  minus = pushforward_normal(trian.minus)
  SkeletonPair(plus,minus)
end

function pushforward_reference_normal(trian::SkeletonTriangulation{Dc,Dp,
                                                           <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}) where {Dc,Dp}
  plus = pushforward_reference_normal(trian.plus)
  minus = pushforward_reference_normal(trian.minus)
  SkeletonPair(plus,minus)
end

function pushforward_parametric_normal(trian::SkeletonTriangulation{Dc,Dp,
                                                           <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}) where {Dc,Dp}
  plus = pushforward_parametric_normal(trian.plus)
  minus = pushforward_parametric_normal(trian.minus)
  SkeletonPair(plus,minus)
end

function pushforward_reference_normal(trian::AdaptedTriangulation)
  pushforward_reference_normal(trian.trian)
end

function pushforward_parametric_normal(trian::AdaptedTriangulation)
  pushforward_parametric_normal(trian.trian)
end

function pushforward_normal(trian::AdaptedTriangulation)
  pushforward_normal(trian.trian)
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

function pullback_area_form(trian::SkeletonTriangulation{Dc,Dp,
                                                           <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}) where {Dc,Dp}
  plus = pullback_area_form(trian.plus)
  minus = pullback_area_form(trian.minus)
  SkeletonPair(plus,minus)
end

function pullback_area_form(trian::AdaptedTriangulation)
  cf = pullback_area_form(trian.trian)
  plus = GenericCellField(get_data(cf.plus),trian.trian.plus,DomainStyle(cf.plus))
  minus = GenericCellField(get_data(cf.minus),trian.trian.minus,DomainStyle(cf.minus))
  SkeletonPair(plus,minus)
end




