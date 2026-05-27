function _generate_face_to_master_cell_id(model::CubedSphereParametricDiscreteModel{Dc};
                                          cell_l2g::AbstractVector{Int}=IdentityVector(num_cells(model))) where Dc
  grid_topology = get_grid_topology(model)
  face_to_master_cell_id = Vector{Vector{Int}}(undef, Dc)
  for i=1:Dc
      face_to_master_cell_id[i] = Vector{Int}(undef, num_faces(model, i-1))
      face_cells  = get_faces(grid_topology, i-1, Dc)
      cache_face_cells = array_cache(face_cells)
      for face_id in 1:num_faces(model, i-1)
        face_cells_around = getindex!(cache_face_cells, face_cells, face_id)
        global_master_cell_id = -1
        local_master_cell_id = -1
        for cell_around in face_cells_around
            if (cell_l2g[cell_around]>global_master_cell_id)
              global_master_cell_id = cell_l2g[cell_around]
              local_master_cell_id = cell_around
            end
        end
        face_to_master_cell_id[i][face_id] = local_master_cell_id
      end
  end
  return face_to_master_cell_id
end


struct GenerateChangeOfBasisMatrixMap{T<:CubedSphereParametricDiscreteModel,S} <: Map
    model::T
    face_to_master_cell_id::S
end

function return_cache(k::GenerateChangeOfBasisMatrixMap,reffe,cell_id)
  model = k.model
  D = num_cell_dims(model)
  gtopo = get_grid_topology(model)
  T = typeof(get_faces(gtopo, D, D - 1))
  cell_faces = Vector{T}(undef, D)
  for i=0:D-1
    cell_faces[i+1] = get_faces(gtopo, D, i)
  end
  Tcache = typeof(array_cache(cell_faces[1]))
  cache_cell_faces = Vector{Tcache}(undef, D)
  for i=0:D-1
    cache_cell_faces[i+1] = array_cache(cell_faces[i+1])
  end
  panel_ids = get_panel_ids(model)
  cache_panel_ids = array_cache(panel_ids)
  cell_map = get_cell_map(model)
  cache_cell_map = array_cache(cell_map)
  cell_faces,cache_cell_faces,panel_ids,cache_panel_ids,cell_map, cache_cell_map,CachedMatrix(Float64)
end

function evaluate!(cache,k::GenerateChangeOfBasisMatrixMap,reffe,cell_id)
  model = k.model
  fwd_map_generator = get_forward_map_generator(model)
  D = num_cell_dims(model)
  face_to_master_cell_id = k.face_to_master_cell_id
  cell_faces, cache_cell_faces, panel_ids,  cache_panel_ids, cell_map, cache_cell_map,
             change_of_basis_matrix_cache = cache
  p = get_polytope(reffe)
  ndofs = num_dofs(reffe)
  setsize!(change_of_basis_matrix_cache, (ndofs, ndofs))
  change_matrix = change_of_basis_matrix_cache.array
  current_cell_map = getindex!(cache_cell_map, cell_map, cell_id)
  reffe_node_coordinates = get_node_coordinates(reffe)
  reffe_node_coordinates = evaluate(current_cell_map, reffe_node_coordinates)
  copyto!(change_matrix, I)
  current_cell_panel = getindex!(cache_panel_ids, panel_ids, cell_id)

  face_own_dofs  = get_face_own_dofs(reffe)
  face_own_nodes = get_face_own_nodes(reffe)
  for i=0:D-1
    num_faces_dim_i = num_faces(p, i)
    offset = get_offset(p, i)
    for j=1:num_faces_dim_i
       current_cell_faces=getindex!(cache_cell_faces[i+1],cell_faces[i+1],cell_id)
       face_gid = current_cell_faces[j]
       master_cell_id = face_to_master_cell_id[i+1][face_gid]
       master_cell_panel = getindex!(cache_panel_ids, panel_ids, master_cell_id)
       dof_lids_slave = face_own_dofs[offset+j]
       node_lids_slave = face_own_nodes[offset+j]
       if (master_cell_panel != current_cell_panel && length(dof_lids_slave)>0)
         master_reffe_node_coordinates = get_node_coordinates(reffe)
         master_cell_map = getindex!(cache_cell_map, cell_map, master_cell_id)
         master_reffe_node_coordinates = evaluate(master_cell_map, master_reffe_node_coordinates)
         pos_master=findfirst(x->x==face_gid, getindex!(cache_cell_faces[i+1],cell_faces[i+1],master_cell_id))
         node_lids_master = face_own_nodes[offset+pos_master]
         for (inode, node_slave) in enumerate(node_lids_slave)
           node_master = node_lids_master[inode]
           JM = J(fwd_map_generator(master_cell_panel),master_reffe_node_coordinates[node_master])
           JSinv = forward_pinv_jacobian(fwd_map_generator(current_cell_panel),reffe_node_coordinates[node_slave])
           coeffs = JSinv⋅JM
           dof_lids_slave_current_node = findall(x->x==node_slave,reffe.reffe.dofs.dof_to_node)
           change_matrix[dof_lids_slave_current_node,dof_lids_slave_current_node] .= Array(coeffs)
         end
       end
     end
  end
  return change_matrix
end

function _get_value_type(cell_reffe::AbstractArray{<:GenericLagrangianRefFE})
  prebasis = get_prebasis(testitem(cell_reffe))
  Dc = num_cell_dims(testitem(cell_reffe))
  x = VectorValue{Dc,Float64}(ntuple(_->0.0,Dc))
  T = return_type(prebasis, x)
  return eltype(T)
end

function _generate_change_of_basis_matrices(model, cell_reffe;
                                           face_to_master_cell_id=_generate_face_to_master_cell_id(model))
  T=_get_value_type(cell_reffe)
  if T <: VectorValue
      k = GenerateChangeOfBasisMatrixMap(model,face_to_master_cell_id)
      return lazy_map(k,
                      cell_reffe,
                      IdentityVector(Int32(num_cells(model))))
  else
    return nothing
  end
end

function _compute_cell_bases_changes(
  ::Type{Float64},  
  ::ReferenceFEName, 
  ::IdentityPiolaMap, 
  model::CubedSphereParametricDiscreteModel, 
  cell_reffe, 
  cell_Jt)
  return nothing
end


function _compute_cell_bases_changes(
  ::Type{<:VectorValue},
  ::ReferenceFEName, 
  ::IdentityPiolaMap, 
  model::CubedSphereParametricDiscreteModel, 
  cell_reffe, 
  cell_Jt)
  change_of_basis_matrices = _generate_change_of_basis_matrices(model, cell_reffe)
  inv_change_of_basis_matrices = lazy_map(transpose, lazy_map(inv, change_of_basis_matrices))
  (change_of_basis_matrices, inv_change_of_basis_matrices)
end

function _compute_cell_bases_changes(
  ::Type{<:TensorValue},
  ::ReferenceFEName, 
  ::IdentityPiolaMap, 
  model::CubedSphereParametricDiscreteModel, 
  cell_reffe, 
  cell_Jt)
  @notimplemented "GridapGeosciences.jl does not support grad-conforming tensor-valued finite elements"
end


function compute_cell_bases_changes(
  ref_name::ReferenceFEName, 
  map::IdentityPiolaMap, 
  model::CubedSphereParametricDiscreteModel, 
  cell_reffe, 
  cell_Jt)
  T = _get_value_type(cell_reffe)
  _compute_cell_bases_changes(T, ref_name, map, model, cell_reffe, cell_Jt)
end

function compute_cell_bases_changes(
  ref_name::ReferenceFEName, 
  map::IdentityPiolaMap, 
  model::AdaptedDiscreteModel{Dc,Dp,<:CubedSphereParametricDiscreteModel{Dc,Dp}}, 
  cell_reffe, 
  cell_Jt) where {Dc,Dp}
  T = _get_value_type(cell_reffe)
  _compute_cell_bases_changes(T, ref_name, map, model.model, cell_reffe, cell_Jt)
end

# We do not want to use CLagrangianFESpace for parametric models, because otherwise
# we cannot implement the change of basis for the vector-valued case
const ParamTrianType{Dc,Dp} = BodyFittedTriangulation{Dc,Dp,<:CubedSphereParametricDiscreteModel{Dc,Dp}}
const UnionParamTrianType{Dc,Dp} = Union{ParamTrianType{Dc,Dp},
                                         AdaptedTriangulation{Dc,Dp,<:ParamTrianType{Dc,Dp}}}

function _use_clagrangian(trian::UnionParamTrianType{Dc,Dp},
                          cell_reffe::AbstractArray{<:GenericLagrangianRefFE},
                          conf::H1Conformity) where {Dc,Dp}
    return false
end
