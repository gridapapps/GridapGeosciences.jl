"""
In this module, we test the cell maps of the serial CubedSphereParametricDiscreteModel.
Theoretically, the panel models should return the cell coordinates of a
Cartesian panel. Test the evaluation of the cell maps on ref coords against
a cartesian model at the same level of refinement
"""

module CellMapTests
using Gridap
using GridapGeosciences
using Test
using Gridap.Geometry
using GridapGeosciences.Geometry


function test_cell_maps(panel_model,cart_model)
  lvl = nref(nc(panel_model))
  println("nref = $lvl")

  nC_per_panel = Int(nc(panel_model))

  @test nC_per_panel == num_cells(cart_model)
  @test nC_per_panel*NPANELS == num_cells(panel_model)


  trian = Triangulation(panel_model)
  cmaps = get_cell_map(trian)
  pts = get_cell_ref_coordinates(trian)

  ## Theoretically, the coordinates on each panel should be equivalent to the cartesian model
  cart_panel_coords = get_cell_coordinates(cart_model)
  _cell_coords = fill(cart_panel_coords,NPANELS)
  cell_coords = vcat(_cell_coords...)

  @test sum(lazy_map(evaluate,cmaps,pts) .≈ cell_coords) == num_cells(panel_model)

end

NPANELS = 6 # number of panels
n_ref_lvls = 4 # number of refinement levels
radius = 1.0 # radius of the sphere

## get hierarchy of refined models
panel_models  = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold(), true)

## Test the coarsest model is there
nC_per_panel = Int(nc(panel_models[end]))
@test nC_per_panel*NPANELS == num_cells(panel_models[end])


## make cartesian models with the same refinement
coarse_cart_panel = UnstructuredDiscreteModel(
  CartesianDiscreteModel((-CUBE_HALF_EDGE,CUBE_HALF_EDGE,-CUBE_HALF_EDGE,CUBE_HALF_EDGE),(nC_per_panel,nC_per_panel)))

cart_panel_models = Vector{Any}(undef,length(panel_models))
cart_panel_models[end] = coarse_cart_panel

for level in length(panel_models)-1:-1:1
  parent = cart_panel_models[level+1]
  cart_panel_models[level] = refine(parent)
end


## test the cell maps of panel models against cartesian models
for (level,(panel_model,cart_model)) in enumerate(zip(panel_models,cart_panel_models))
  test_cell_maps(panel_model,cart_model)
end

@test true


end #module
