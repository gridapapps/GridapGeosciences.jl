
function pullback_area_form(trian::DistributedTriangulation)
  fields = map(trian.trians) do t
    pullback_area_form(t)
  end
  GridapDistributed.DistributedCellField(fields,trian)
end

function pushforward_normal(trian::GridapDistributed.DistributedTriangulation)
  fields = map(trian.trians) do t
    pushforward_normal(t)
  end
  GridapDistributed.DistributedCellField(fields,trian)
end

function pushforward_reference_normal(trian::GridapDistributed.DistributedTriangulation)
  fields = map(trian.trians) do t
    pushforward_reference_normal(t)
  end
  GridapDistributed.DistributedCellField(fields,trian)
end

function pushforward_parametric_normal(trian::GridapDistributed.DistributedTriangulation)
  fields = map(trian.trians) do t
    pushforward_parametric_normal(t)
  end
  GridapDistributed.DistributedCellField(fields,trian)
end

"""
get_surface_normal

Is the distributed implementation of get_surface_normal.
In such function, we call get_surface_normal on the local model and then
recompute the triangulation to ensure proper handling of ghost cells in octree periodic meshes.
"""
function get_surface_normal(trian::GridapDistributed.DistributedTriangulation)
  model = trian.model
  _trian = GridapDistributed.add_ghost_cells(trian)
  fields = map(_trian.trians) do t
    get_surface_normal(t)
  end
  GridapDistributed.DistributedCellField(fields,_trian)
end


"""
BoundaryTriangulation

For the boundary triangulation of the 2D ambient model, we  want to mask all cells
as the boundary
"""
function GridapDistributed.BoundaryTriangulation(
  portion,model::CubedSphereAmbientDistributedDiscreteModel{2,3,<:CubedSphereAmbientDiscreteModel},
  labels::GridapDistributed.DistributedFaceLabeling;tags=nothing)
  Dc = num_cell_dims(model)

  topo = GridapDistributed.get_grid_topology(model)
  face_to_mask = GridapDistributed.get_isboundary_face(topo,Dc-1) # This is globally consistent

  # For the 2D model, we want to mask all cells
  _face_to_mask = map(face_to_mask) do f
    .!f
  end
  GridapDistributed.BoundaryTriangulation(portion,model,_face_to_mask)
end
