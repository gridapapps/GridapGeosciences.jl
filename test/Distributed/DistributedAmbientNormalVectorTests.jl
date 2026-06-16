"""
In this module, test the skeleton normal vectors for the distributed ambient model are
  1. in the tangent space of the sphere
  2. equiv to the mapped skeleton normal vectors of the distributed parametric model
"""

module DistributedAmbientNormalTests

using Gridap
using GridapDistributed
using GridapGeosciences
using Test

function test_debug_equiv(p,q)
  map(p,q  )do p,q
    dif = p - q
    max_dif = map(x->maximum(norm.(x)),dif)
    @test all(max_dif .< 1e-12)
  end
end


function main(distribute,nprocs)
  ranks = distribute(LinearIndices((nprocs,)))

  n_ref_lvls = 2
  radius = 1.0
  coarse_mesh = CubedSphereMesh(radius)
 
  ambient_model = AtlasDiscreteModel(ranks, 
                                     coarse_mesh, 
                                     n_ref_lvls, 
                                     manifold_style=ExtrinsicManifold())
 
  parametric_model = AtlasDiscreteModel(ranks, 
                                        coarse_mesh, 
                                        n_ref_lvls, 
                                        manifold_style=IntrinsicManifold())

  ##############################################################################
  ########## Ambient model
  ##############################################################################
  Ω_ambient = Triangulation(ambient_model)
  n_surface_ambient = get_surface_normal(Ω_ambient)

  Λ_ambient = SkeletonTriangulation(with_ghost,ambient_model)
  n_Λ_ambient = get_normal_vector(Λ_ambient)
  pts_ambient = get_cell_points(Λ_ambient)

  ### Test the skeleton normal is in the tangent space of the sphere
  normal_component = (n_Λ_ambient⋅n_surface_ambient).plus(pts_ambient)
  max_dif = map(x->maximum(norm.(x)),normal_component)
  map(max_dif) do _d
    @test all(_d .< 1e-12)
  end


  normal_component = (n_Λ_ambient⋅n_surface_ambient).minus(pts_ambient)
  max_dif = map(x->maximum(norm.(x)),normal_component)
  map(max_dif) do _d
    @test all(_d .< 1e-12)
  end

  ##############################################################################
  ########## Parametric model
  ##############################################################################

  Λ_panel = SkeletonTriangulation(with_ghost,parametric_model)
  n_Λ_mapped = pushforward_normal(Λ_panel)
  pts_panel = get_cell_points(Λ_panel)

  ## Test equivalence with ambient model
  p = n_Λ_mapped.plus(pts_panel)#.plus(pts_panel)
  q = n_Λ_ambient.plus(pts_ambient)
  test_debug_equiv(p,q)

  p = n_Λ_mapped.minus(pts_panel)
  q = n_Λ_ambient.minus(pts_ambient)
  test_debug_equiv(p,q)



  @test true

end

end # module
