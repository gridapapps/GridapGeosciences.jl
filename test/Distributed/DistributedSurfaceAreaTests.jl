"""
In this module, we test that the surface area computed by the distributed serial model
is equivalent to the surface area computed by the 2D P4est model at the same
level of refinement on more than 1 processor
i.e. surface area = ∫ᵧ 1 = ∫ 1 √g
"""

module DistributedSurfaceAreaTests

using Gridap
using Gridap.Adaptivity
using GridapGeosciences
using GridapP4est
using Test



function compute_surface_area(model::IntrinsicAtlasDistributedDiscreteModel, degree::Int)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  meas_cf = MeasureCellField(Ω)
  surface_area = sum( ∫( 1.0*meas_cf )dΩ )
  return surface_area
end

function compute_surface_area(model::AdaptedIntrinsicAtlasDistributedDiscreteModel, degree::Int)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  meas_cf = MeasureCellField(Ω)
  surface_area = sum( ∫( 1.0*meas_cf )dΩ )
  return surface_area
end

function compute_surface_area(model::ExtrinsicAtlasDistributedDiscreteModel{2,Dp}, degree::Int) where {Dp}
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  surface_area = sum( ∫( 1.0 )dΩ )
  return surface_area
end

function compute_surface_area(model::AdaptedExtrinsicAtlasDistributedDiscreteModel{2,Dp}, degree::Int) where {Dp}
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  surface_area = sum( ∫( 1.0 )dΩ )
  return surface_area
end

function compute_surface_area(model::AtlasOctreeDistributedDiscreteModel{Dc,Dp,A,B,<:IntrinsicManifold}, degree::Int) where {Dc,Dp,A,B}
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  meas_cf = MeasureCellField(Ω)
  surface_area = sum( ∫( 1.0*meas_cf )dΩ )
  return surface_area
end

function compute_surface_area(model::AtlasOctreeDistributedDiscreteModel{Dc,Dp,A,B,<:ExtrinsicManifold}, degree::Int) where {Dc,Dp,A,B}
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  surface_area = sum( ∫( 1.0 )dΩ )
  return surface_area
end

function compute_surface_area(model::ExtrudedAtlasOctreeDistributedDiscreteModel{A,B,<:IntrinsicManifold}, degree::Int) where {A,B}
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  meas_cf = MeasureCellField(Ω)
  surface_area = sum( ∫( 1.0*meas_cf )dΩ )
  return surface_area
end

function compute_surface_area(model::ExtrudedAtlasOctreeDistributedDiscreteModel{A,B,<:ExtrinsicManifold}, degree::Int) where {A,B}
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  surface_area = sum( ∫( 1.0 )dΩ )
  return surface_area
end

function test_surface_area(dist_models::AbstractArray,p4est_models::AbstractArray)
  for degree in collect([2,4,6,8])
    for (d_model,p4_model) in zip(dist_models,p4est_models)
      radius = get_sphere_radius(d_model)
      extact_area = 4*π*radius^2

      ### d_model
      d_area = compute_surface_area(d_model, degree)

      ### p4est model
      p4_area = compute_surface_area(p4_model, degree)

      @test d_area ≈ p4_area

      e = abs(d_area-extact_area)/extact_area
      @test e < 1e-2
    end
  end
end



function main(distribute,nprocs)
  ranks = distribute(LinearIndices((nprocs,)))

  ## 2D parametric models:
  n_ref_lvls = 3
  for radius in [1.0,2.0]
    dist_models = generate_distributed_refined_models(ranks, CubedSphereMesh(radius), n_ref_lvls, IntrinsicManifold())
    coarse_mesh = CubedSphereMesh(radius)
    p4est_models = generate_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, IntrinsicManifold())
    test_surface_area(dist_models,p4est_models)
  end

  ## 2D ambient models:
  for radius in [1.0,2.0]
    dist_models = generate_distributed_refined_models(ranks, CubedSphereMesh(radius), n_ref_lvls, ExtrinsicManifold())
    coarse_mesh = CubedSphereMesh(radius)
    p4est_models = generate_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, ExtrinsicManifold())
    test_surface_area(dist_models,p4est_models)
  end

  ## 3D extruded octree intrinsic models:
  n_ext_ref_lvls = 2
  for (radius, thickness) in [(1.0, 0.1), (2.0, 0.2)]
    ext_mesh  = ExtrudedCubedSphereWithThicknessMesh(radius, thickness)
    exact_vol = (4/3) * π * ((radius + thickness)^3 - radius^3)
    model = ExtrudedAtlasOctreeDistributedDiscreteModel(ranks, ext_mesh, 0, 0; manifold_style=IntrinsicManifold())
    for _ in 1:n_ext_ref_lvls
      model, _ = Gridap.Adaptivity.refine(model)
      for degree in [2, 4]
        vol = compute_surface_area(model, degree)
        e   = abs(vol - exact_vol) / exact_vol
        @test e < 1e-2
      end
    end
  end

  ## 3D extruded octree extrinsic models:
  for (radius, thickness) in [(1.0, 0.1), (2.0, 0.2)]
    ext_mesh  = ExtrudedCubedSphereWithThicknessMesh(radius, thickness)
    exact_vol = (4/3) * π * ((radius + thickness)^3 - radius^3)
    model = ExtrudedAtlasOctreeDistributedDiscreteModel(ranks, ext_mesh, 0, 0; manifold_style=ExtrinsicManifold())
    for _ in 1:n_ext_ref_lvls
      model, _ = Gridap.Adaptivity.refine(model)
      for degree in [2, 4]
        vol = compute_surface_area(model, degree)
        e   = abs(vol - exact_vol) / exact_vol
        @test e < 1e-2
      end
    end
  end

  @test true
end


end # module
