# function _generate_change_of_basis_matrices(model, cell_reffe, cell_l2g)
#     face_to_master_cell_id = _generate_face_to_master_cell_id(model; cell_l2g=cell_l2g)
#     _generate_change_of_basis_matrices(model, cell_reffe;
#                                        face_to_master_cell_id=face_to_master_cell_id)
# end

# function _generate_change_of_basis_matrices(model::CubedSphereParametricDistributedDiscreteModel, cell_reffes)
#     is_vector = false
#     map(cell_reffes) do cell_reffe
#       T=_get_value_type(cell_reffe)
#       if T <: VectorValue
#         is_vector = true
#       end
#     end

#     if is_vector
#         cell_gids = get_cell_gids(model)
#         change_of_basis_matrices=map(local_views(model),cell_reffes,partition(cell_gids)) do m,cell_reffe,cell_indices
#            cell_l2g = local_to_global(cell_indices)
#            _generate_change_of_basis_matrices(m, cell_reffe, cell_l2g)
#         end
#         cell_vecs = map(change_of_basis_matrices) do change_of_basis_matrices
#            JaggedArray(map(a -> reshape(a, length(a)), change_of_basis_matrices))
#         end
#         p = PVector(cell_vecs, partition(cell_gids))
#         wait(consistent!(p))
#         map(partition(p), cell_reffes) do cell_vecs, cell_reffes
#           map((a,b)->reshape(a,(num_dofs(b),num_dofs(b))), cell_vecs, cell_reffes)
#         end
#     else
#         map(cell_reffes) do cell_reffe
#           nothing
#         end
#     end
# end

# function FESpace(model::CubedSphereParametricDistributedDiscreteModel,
#                  reffe::Tuple{<:Lagrangian,Any, Any};
#                  split_own_and_ghost=false,
#                  constraint=nothing,
#                  conformity=nothing,kwargs...)

#   if (conformity==:L2)
#      spaces = map(local_views(model)) do m
#         FESpace(m, reffe; conformity=conformity, kwargs...)
#      end
#   else
#      cell_reffes = map(local_views(model)) do m
#          basis,reffe_args,reffe_kwargs = reffe
#          cell_reffe = ReferenceFE(m,basis,reffe_args...;reffe_kwargs...)
#      end
#      change_of_basis_matrices = _generate_change_of_basis_matrices(model, cell_reffes)
#      spaces = map(local_views(model),cell_reffes,change_of_basis_matrices) do m,cell_reffe,change_of_basis_matrices
#         conf = Conformity(testitem(cell_reffe),conformity)
#         cell_fe = CellFE(m,cell_reffe,conf,change_of_basis_matrices)
#         FESpace(m, cell_fe; kwargs...)
#      end
#   end
#   gids = generate_gids(model,spaces)
#   trian = DistributedTriangulation(map(get_triangulation,spaces),model)
#   vector_type = _find_vector_type(spaces,gids;split_own_and_ghost=split_own_and_ghost)
#   space=DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
#   return _add_distributed_constraint(space,reffe,constraint)
# end

# function FESpace(_trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:ParamTrianType{Dc,Dp}}},
#                  reffe::Tuple{<:Lagrangian,Any, Any};
#                  split_own_and_ghost=false,
#                  constraint=nothing,
#                  conformity=nothing,
#                  kwargs...) where {Dc,Dp}

#  dmodel, dtrian = _setup_dmodel_and_dtrian(_trian)
#  if (conformity==:L2)
#      spaces = map(local_views(dmodel)) do m
#         FESpace(m, reffe; conformity=conformity, kwargs...)
#      end
#  else
#      cell_reffes = map(local_views(dmodel)) do m
#          basis,reffe_args,reffe_kwargs = reffe
#          cell_reffe = ReferenceFE(m,basis,reffe_args...;reffe_kwargs...)
#      end
#      change_of_basis_matrices = _generate_change_of_basis_matrices(dmodel, cell_reffes)
#      spaces = map(local_views(dmodel),
#                   cell_reffes,
#                   local_views(dtrian),
#                   change_of_basis_matrices) do m,
#                                                cell_reffe,
#                                                trian,
#                                                change_of_basis_matrices
#            conf = Conformity(testitem(cell_reffe),conformity)
#            cell_fe = CellFE(m,cell_reffe,conf,change_of_basis_matrices)
#            FESpace(m, cell_fe, trian=trian; kwargs...)
#      end
#   end
#   gids = generate_gids(dmodel,spaces)
#   trian = DistributedTriangulation(map(get_triangulation,spaces),dmodel)
#   vector_type = _find_vector_type(spaces,gids;split_own_and_ghost=split_own_and_ghost)
#   space=DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
#   return _add_distributed_constraint(space,reffe,constraint)
# end
