"""
auto convergence tests
"""
function test_slope(slope,p_fe,Dc)
  if slope >= p_fe
    return true
  end
  # In the event that the slope for 3D models is slightly less than p_fe
  if Dc == 3 && (slope >= p_fe -1e-1)
      return true
  end
  return false
end

function p_convergence_auto_test(ps::Vector{Int},
                                 models::AbstractArray,
                                 convergence_func,
                                 dir::String,fargs...;
                                 _i_am_main=true)

  Dc = num_cell_dims(testitem(models))
  # nref == refinement level of the mesh
  # 2^nref == number of cells per panel
  # 1/(2^nref) -> to get positive slopes

  lvls = map(x->1.0/(2.0^nref(x)),models)
  # lvls = map(x->num_cells(x),models)

  for (i,p_fe) in enumerate(ps)
    errs = h_convergence_auto_test(models,convergence_func,p_fe,dir,fargs...; _i_am_main=_i_am_main)
    slope = convergence_rate(lvls,errs)
    _i_am_main && println("slope = $slope")
    @test test_slope(slope,p_fe,Dc) #slope >= p_fe
  end
end

function h_convergence_auto_test(models::AbstractArray,f,p_fe::Int,dir::String,fargs...; _i_am_main=true)
  errs = Float64[]
  for model in models
    e, = f(model,p_fe,dir,fargs...; _i_am_main=_i_am_main)
    push!(errs,e)
  end
  return errs
end



"""
nref
returns the level of refinement
  * in 3D, returns the horizontal refinement
"""
nref(nc) = Int(log2(sqrt(nc))) ## level of refinement

function nref(model::Union{<:DiscreteModel{2,Dp},<:GridapDistributed.DistributedDiscreteModel{2,Dp}}) where Dp
  nref(nc(model))
end

const AtlasDiscreteModelCubedSphere3D = AtlasDiscreteModel{3,3,<:Any,<:Any,<:AbstractVector{<:CubedSphereWithThicknessMap}}

const AdaptedAtlasDiscreteModelCubedSphere3D = AdaptedDiscreteModel{3,3,<:AtlasDiscreteModelCubedSphere3D}

const ATDMCS3D = Union{AtlasDiscreteModelCubedSphere3D,AdaptedAtlasDiscreteModelCubedSphere3D}

function nref(model::Union{<:ATDMCS3D,
                           <:GridapDistributed.DistributedDiscreteModel{3,3}})
  nref(nc_horizontal(model))
end

## nc = num cells per panel
function nc(model::Union{<:DiscreteModel{2,Dp},<:GridapDistributed.DistributedDiscreteModel{2,Dp}}) where Dp
  num_cells(model)/6
end
function nc(model::Union{<:ATDMCS3D,
                         <:GridapDistributed.DistributedDiscreteModel{3,3}})
  nc_horizontal(model) + _nc_vertical(model)
end

function nc_horizontal(model::Union{AtlasOctreeDistributedDiscreteModel{3,3},
                                    ExtrudedAtlasOctreeDistributedDiscreteModel})
    nc_horizontal(get_atlas_model(model))
end 

function nc_vertical(model::Union{AtlasOctreeDistributedDiscreteModel{3,3},
                                   ExtrudedAtlasOctreeDistributedDiscreteModel})
   nc_vertical(get_atlas_model(model))
end 

function nc_horizontal(model::Union{IntrinsicAtlasDistributedDiscreteModel{3,3},                                       
                                    AdaptedIntrinsicAtlasDistributedDiscreteModel{3,3},
                                    ExtrinsicAtlasDistributedDiscreteModel{3,3},
                                    AdaptedExtrinsicAtlasDistributedDiscreteModel{3,3}})

  grid = get_grid(model)
  gids = get_cell_gids(model)

  ## find the number of cells that are on the surface.
  ## i.e. with γ = 0.0
  ## make sure to extract only the owned
  f = map(local_views(grid),partition(gids)) do grid, cids
    cmap = _chart_maps(grid)
    pts = get_cell_ref_coordinates(grid)
    f = lazy_map(evaluate,cmap,pts)
    g = lazy_map(FindSurfaceCells(),f)
    owned_cells = own_to_local(cids)
    sum(g[owned_cells])
  end
  nsurface = sum(f)
  ncells_per_panel = Int(nsurface/6)
  return ncells_per_panel
end

## nc = num cells per panel in horizontal
function nc_horizontal(model::ATDMCS3D)
  ## find the number of cells that are on the surface.
  grid = get_grid(model)
  cmap = get_cell_map(grid)
  pts = get_cell_ref_coordinates(grid)
  f = lazy_map(evaluate,cmap,pts)
  g = lazy_map(FindSurfaceCells(),f)
  nsurface = sum(g)
  ncells_per_panel = Int(nsurface/6)
  return ncells_per_panel
end

# return square here so vertical is 'like' horitzontal
function nc_vertical(model::Union{ExtrinsicAtlasDistributedDiscreteModel{3,3},
                                  AdaptedExtrinsicAtlasDistributedDiscreteModel{3,3},
                                  IntrinsicAtlasDistributedDiscreteModel{3,3},
                                  AdaptedIntrinsicAtlasDistributedDiscreteModel{3,3}})
  n = _nc_vertical(model)
  return Int(n^2)
end

# the actual number of cells in vertical per panel
function _nc_vertical(model::Union{ExtrinsicAtlasDistributedDiscreteModel{3,3},
                                   AdaptedExtrinsicAtlasDistributedDiscreteModel{3,3},
                                   IntrinsicAtlasDistributedDiscreteModel{3,3},
                                   AdaptedIntrinsicAtlasDistributedDiscreteModel{3,3}})
  ncells_per_panel = nc_horizontal(model)
  n = num_cells(model)/6
  _n =  n /ncells_per_panel
  return Int(_n)
end

using Gridap.Arrays
using Gridap.Geometry

"""
find the cells at γ = 0.0
"""
struct FindSurfaceCells  <: Map
end

function Gridap.Arrays.return_cache(f::FindSurfaceCells,cellx::AbstractArray{<:VectorValue{3}})
  y = similar(cellx,true)
  return y
end

function Gridap.Arrays.evaluate!(cache,f::FindSurfaceCells,cellx::AbstractArray{<:VectorValue{3}} )
  y = cache
  y = !isempty(findall(x->x[1]==0.0,cellx))
  return y
end

function Gridap.Arrays.return_cache(f::FindSurfaceCells,x::VectorValue{3})
  y = true
  return y
end

function Gridap.Arrays.evaluate!(cache,f::FindSurfaceCells,x::VectorValue{3} )
  y = cache
  y = !isempty(findall(x[1]==0.0))
  return y
end


## element size
function dx(model::Union{<:DiscreteModel{2,Dp},<:GridapDistributed.DistributedDiscreteModel{2,Dp}}) where Dp
  radius = get_radius(model)
  tmp =  4*π*radius^2/num_cells(model)
  sqrt(tmp)
end

function dx(model::Union{<:DiscreteModel{3,3},<:GridapDistributed.GenericDistributedDiscreteModel{3,3}})
  horizontal = dx_horizontal(model)
  vertical = dx_vertical(model)
  horizontal*vertical
end

function dx_horizontal(model::GridapDistributed.GenericDistributedDiscreteModel{3,3})
  radius = get_radius(model)
  horizontal = 4*π*radius^2/(nc_horizontal(model)*6)
  sqrt(horizontal) ## quads so have to sqrt
end

function dx_vertical(model::GridapDistributed.GenericDistributedDiscreteModel{3,3})
  thickness = get_thickness(model)
  vertical = thickness/_nc_vertical(model)
  vertical ### single layer, so no sqrt
end


function convergence_rate(dxs,errors)
  x = log10.(dxs)
  y = log10.(abs.(errors))
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end
