################################################################################
######### Parametric model (in the reference space)
################################################################################
### To save flops, we compute all metric term explicitly
function allg(αβ)

  t1 = tan(αβ[1])
  t2 = tan(αβ[2])

  c1 = cos(αβ[1])
  c2 = cos(αβ[2])

  r = 1.0 + t1*t1 + t2*t2
  r4 = r*r

  c = 1.0/( r4*c1*c1 *c2*c2 )

  f = c * ( -1.0*t1*t2 )
  e = c * ( 1.0 + (t1*t1) )
  g = c * ( 1.0 + (t2*t2) )

  sqrtg = (e*g - f*f)^(1/2)

  1.0/sqrtg*TensorValue(g, -f, -f, e )
end

## ensure to use a view and cache to avoid allocations
# @allocated view(intq,:,1)
# @allocated intq[:,1]
function bm_intergrate_reference_panel(cache,intq,w)
  s = 0.0
  for i in 1:size(intq,2)
    s += evaluate!(cache, IntegrationMap(),view(intq,:,i),w)
  end
end

function benchmark_reference_panel(order,degree,dir=@__DIR__,return_output=false)
  model = CartesianDiscreteModel((-π/4,π/4,-π/4,π/4),(1,1))
  cmap = get_cell_map(model)[1]

  reffe = LagrangianRefFE(Float64,QUAD,order)

  ϕ = get_shapefuns(reffe)

  grad_dv_array = map(x->∇(x),ϕ)
  grad_du_array = map(x->∇(transpose(x)),ϕ)

  # allg_array = Broadcasting(∘)(GenericField(allg),cmap)
  allg_array = Broadcasting(Operation(allg))(cmap)

  _I = Broadcasting(Operation(⋅))(allg_array,grad_du_array)
  integrand = Broadcasting(Operation(⋅))(grad_dv_array,_I)

  ## Integrate in reference space
  quad_array, w = get_quadrature(degree)

  intq = evaluate(integrand,quad_array)

  cache = return_cache(IntegrationMap(),view(intq,:,1),w)
  out_ref = bm_intergrate_reference_panel(cache,intq,w)

  # if return_output
  #   cache = return_cache(IntegrationMap(),view(intq,:,1),w)
  #   t_ref = @belapsed bm_intergrate_reference_panel($cache,$intq,$w)

  #   cache = return_cache(IntegrationMap(),view(intq,:,1),w)
  #   out_ref = @gflops bm_intergrate_reference_panel($cache,$intq,$w)

  #   !isdir(dir) && mkpath(dir)
  #   save_ouput(dir,"panel",t_ref,out_ref,order)
  # end
end
