"""
In this module, we test the refinement of the serial models by checking
1. the number of cell dims, the number of point dims
2. the refined model is a child of the parent

Replicate for parametric and ambient models
"""

module RefinementTests

using Gridap
using GridapGeosciences
using Gridap.Helpers,  Gridap.Adaptivity
using Test


radius = 1.0

################################################################################
########## Parametric model
################################################################################
### Check the Dc, Dp of the coarse model
panel_model = coarse_parametric_model(radius)

@test num_point_dims(panel_model) == num_cell_dims(panel_model) == 2

### Apply refinement, check the list of refined models
n_ref_lvls = 4
panel_models = get_intrinsic_cubed_sphere_refined_models(n_ref_lvls, radius)
for lev in 1:n_ref_lvls-1
  @test num_point_dims(panel_models[lev]) == num_cell_dims(panel_models[lev]) == 2
  @test is_child(panel_models[lev],panel_models[lev+1])
end

################################################################################
########## Ambient model
################################################################################
### Check the Dc, Dp of the coarse model
ambient_model = CubedSphereAmbientDiscreteModel(radius;num_initial_uniform_refinements=0)

@test num_point_dims(ambient_model) == 3
@test num_cell_dims(ambient_model) == 2

### Apply refinement, check the list of refined models
n_ref_lvls = 4
ambient_models = get_extrinsic_cubed_sphere_refined_models(n_ref_lvls, radius)
for lev in 1:n_ref_lvls-1
  @test num_point_dims(ambient_models[lev]) == 3
  @test num_cell_dims(ambient_models[lev]) == 2
  @test is_child(ambient_models[lev],ambient_models[lev+1])
end


@test true
end
