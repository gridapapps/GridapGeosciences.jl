"""
In this module, we test that the surface area computed by the serial model
is equivalent to:
1. the exact surface area 4πr^2
2. the surface area computed by the 2D P4est model at the same level of refinement (on 1 processor)
Note:  surface area = ∫ᵧ 1 = ∫ 1 √g
"""

module SurfaceArea

using Gridap
using GridapGeosciences
using GridapP4est
using Test


function compute_surface_area(model, degree::Int)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  meas_cf = MeasureCellField(Ω)
  surface_area = sum( ∫( 1.0*meas_cf )dΩ )
  return surface_area
end

function main(serial_models::AbstractArray)
  for degree in collect([2,4,6,8])
    for (s_model) in serial_models
      radius = get_sphere_radius(s_model)
      extact_area = 4*π*radius^2

      ### s_model
      s_area = compute_surface_area(s_model, degree)

      e = abs(s_area-extact_area)/extact_area
      @test e < 1e-2
    end
  end
end

function main(distribute,nprocs;n_ref_lvls=3,radii=[1.0,2.0])
  ranks = distribute(LinearIndices((nprocs,)))
  for radius in radii
    serial_models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
    coarse_mesh = CubedSphereMesh(radius)
    p4test_models = generate_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, IntrinsicManifold())
    for degree in collect([2,4,6,8])
      for (s_model,d_model) in zip(serial_models,p4test_models)

        ### s_model
        s_area = compute_surface_area(s_model, degree)

        ### d_model
        d_area = compute_surface_area(d_model, degree)

        @test s_area ≈ d_area

      end
    end
  end

end


end ## module
