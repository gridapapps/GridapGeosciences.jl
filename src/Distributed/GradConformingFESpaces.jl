function _generate_change_of_basis_matrices(model, cell_reffe, cell_l2g)
    face_to_master_cell_id = _generate_face_to_master_cell_id(model; cell_l2g=cell_l2g)
    _generate_change_of_basis_matrices(model, cell_reffe;
                                       face_to_master_cell_id=face_to_master_cell_id)
end

function _generate_change_of_basis_matrices(model::IntrinsicAtlasDistributedDiscreteModel, cell_reffes)
    is_vector = false
    map(cell_reffes) do cell_reffe
      T=_get_value_type(cell_reffe)
      if T <: VectorValue
        is_vector = true
      end
    end

    if is_vector
        cell_gids = get_cell_gids(model)
        change_of_basis_matrices=map(local_views(model),cell_reffes,partition(cell_gids)) do m,cell_reffe,cell_indices
           cell_l2g = local_to_global(cell_indices)
           _generate_change_of_basis_matrices(m, cell_reffe, cell_l2g)
        end
        cell_vecs = map(change_of_basis_matrices) do change_of_basis_matrices
           JaggedArray(map(a -> reshape(a, length(a)), change_of_basis_matrices))
        end
        p = PVector(cell_vecs, partition(cell_gids))
        wait(consistent!(p))
        map(partition(p), cell_reffes) do cell_vecs, cell_reffes
          map((a,b)->reshape(a,(num_dofs(b),num_dofs(b))), cell_vecs, cell_reffes)
        end
    else
        map(cell_reffes) do cell_reffe
          nothing
        end
    end
end

function DistributedSingleFieldFESpace(
  model::IntrinsicAtlasDistributedDiscreteModel{Dc,Dp}, # Active model, not bg model
  trian::DistributedTriangulation{Dc,Dp,
                                  <:AbstractArray{<:Union{<:BFTATDMIM{Dc,Dp},
                                                          <:AdaptedTriangulation{Dc,Dp,
                                                                                 <: BFTATDMIM{Dc,Dp}}}}},
  cell_gids::PRange, 
  cell_reffe::AbstractArray{<:AbstractArray{T}}; 
  labels = get_face_labeling(model), 
  split_own_and_ghost=false, 
  constraint=nothing,
  conformity=nothing,
  kwargs...
) where {Dc, Dp, T<:GenericLagrangianRefFE}

  # Construct a globally conforming CellFE
  conf = map(cell_reffe) do cell_reffe
    Conformity(testitem(cell_reffe),conformity)
  end |> getany
  cell_fe = CellFE(model, cell_reffe, conf)

  spaces = map(
    local_views(model),local_views(trian),local_views(labels), cell_fe
  ) do model, trian, labels, cell_fe
    FESpace(model,cell_fe;trian,labels,kwargs...)
  end

  gids = generate_gids(cell_gids,spaces)
  vector_type = _find_vector_type(spaces,gids;split_own_and_ghost)
  space = DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
  return _add_distributed_constraint(space,cell_reffe,constraint)
end

function compute_cell_bases_changes(
  ::ReferenceFEName, ::IdentityPiolaMap, model::IntrinsicAtlasDistributedDiscreteModel, cell_reffe, cell_Jt
) 
  change_matrices = _generate_change_of_basis_matrices(model, cell_reffe)
  map(change_matrices) do change_matrices
    if isnothing(change_matrices)
      nothing
    else
      inv_change_matrices = lazy_map(transpose, lazy_map(inv, change_matrices))
      (change_matrices, inv_change_matrices)
    end
  end 
end