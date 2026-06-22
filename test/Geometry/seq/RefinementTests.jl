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
coarse_mesh = CubedSphereMesh(radius)
atlas_model = AtlasDiscreteModel(coarse_mesh,0,manifold_style=IntrinsicManifold())

@test num_point_dims(atlas_model) == num_cell_dims(atlas_model) == 2

### Apply refinement, check the list of refined models
n_ref_lvls = 4
atlas_models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
for lev in 1:n_ref_lvls-1
  @test num_point_dims(atlas_models[lev]) == num_cell_dims(atlas_models[lev]) == 2
  @test is_child(atlas_models[lev],atlas_models[lev+1])
end

################################################################################
########## Ambient model
################################################################################
### Check the Dc, Dp of the coarse model
ambient_model = AtlasDiscreteModel(coarse_mesh,0,manifold_style=ExtrinsicManifold())

@test num_point_dims(ambient_model) == 3
@test num_cell_dims(ambient_model) == 2

### Apply refinement, check the list of refined models
n_ref_lvls = 4
ambient_models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), ExtrinsicManifold())
for lev in 1:n_ref_lvls-1
  @test num_point_dims(ambient_models[lev]) == 3
  @test num_cell_dims(ambient_models[lev]) == 2
  @test is_child(ambient_models[lev],ambient_models[lev+1])
end


@test true
end
