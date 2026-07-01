
################################################################################
######### Ambient model
################################################################################
## ensure to use a view and cache to avoid allocations
# @allocated view(intq,:,1)
# @allocated intq[:,1]
function bm_intergrate_ambient(cache,intq,w,J)
  s = 0.0
  for i in 1:size(intq,2)
    s += evaluate!(cache, IntegrationMap(),view(intq,:,i),w,J)
  end
end

function benchmark_ambient(order,degree,dir=@__DIR__,return_output=false)

  panel_model = CartesianDiscreteModel((-π/4,π/4,-π/4,π/4),(1,1))
  model = SingleAmbientDiscreteModel(panel_model, 1.0)

  cmap = get_cell_map(model)[1]
  reffe = LagrangianRefFE(Float64,QUAD,order)

  ϕ = get_shapefuns(reffe)

  ∇ϕ = map(x->∇(x),ϕ )
  grad_dv_array = Broadcasting(push_∇)(∇ϕ,cmap)

  ∇ϕᵀ = map(x->∇(transpose(x)),(ϕ))
  grad_du_array = Broadcasting(push_∇)(∇ϕᵀ,cmap)

  integrand = Broadcasting(Operation(⋅))(grad_dv_array,grad_du_array)

  ## Integrate in physical space
  quad_array, w = get_quadrature(degree)

  J = ∇(cmap)
  Jq = evaluate(J,quad_array)
  intq = evaluate(integrand,quad_array)

  cache = return_cache(IntegrationMap(),intq[:,1],w,J)
  out_ambient = bm_intergrate_ambient(cache,intq,w,Jq)

  # if return_output
  #   cache = return_cache(IntegrationMap(),intq[:,1],w,J)
  #   t_ambient = @belapsed bm_intergrate_ambient($cache,$intq,$w,$Jq)

  #   cache = return_cache(IntegrationMap(),intq[:,1],w,J)
  #   out_ambient = @gflops bm_intergrate_ambient($cache,$intq,$w,$Jq)

  #   !isdir(dir) && mkpath(dir)
  #   save_ouput(dir,"ambient",t_ambient,out_ambient,order)
  # end
end


function SingleAmbientDiscreteModel(panel_model::CartesianDiscreteModel,radius)
  model = UnstructuredDiscreteModel(panel_model)
  SingleAmbientDiscreteModel(model,radius)
end


function SingleAmbientDiscreteModel(panel_model::UnstructuredDiscreteModel,radius)

  panel_grid = get_grid(panel_model)
  panel_topo = get_grid_topology(panel_model)
  labels = get_face_labeling(panel_model)

  ## map: reffe -> alpha,beta
  cmap = get_cell_map(panel_grid)

  ## map: alpha,beta -> manifold
  fwd_map =  lazy_map(p -> CubedSphereMap(1,radius), collect(1:num_cells(panel_model)))

  ## map: reffe -> manifold
  geo_cmap = lazy_map(∘,fwd_map,cmap)

  ref_pts = get_cell_ref_coordinates(panel_grid)
  ambient_cell_coords = lazy_map(evaluate,geo_cmap,ref_pts)

  eT = eltype(testitem(ambient_cell_coords))
  panel_nodes = Gridap.Geometry.num_nodes(panel_grid)
  ambient_nodes = fill(zero(eT),panel_nodes)
  # Now create proper nodes from the cell-wise array of coords
  # We can do this for the ambient model because the nodes are defined uniquely
  # in ambient space
  cell_to_nodes = get_cell_node_ids(panel_grid)
  Gridap.FESpaces._free_and_dirichlet_values_fill!(
    ambient_nodes, eT[],
    array_cache(ambient_cell_coords),
    array_cache(cell_to_nodes),
    ambient_cell_coords,
    cell_to_nodes,
    eachindex(cell_to_nodes)
  )

  ## the ambient_grid has the bespoke panel_2_ambient_cmap
  ambient_grid = Gridap.Geometry.UnstructuredGrid(ambient_nodes,get_cell_node_ids(panel_grid),get_reffes(panel_grid),get_cell_type(panel_grid),OrientationStyle(panel_grid),
                      nothing,(geo_cmap))
  ambient_topo = UnstructuredGridTopology(ambient_nodes,get_cell_node_ids(panel_grid),get_cell_type(panel_topo),get_polytopes(panel_topo),OrientationStyle(panel_topo))

  UnstructuredDiscreteModel(ambient_grid,ambient_topo,labels)

end
