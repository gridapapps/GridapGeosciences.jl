macro publish(mod,name)
  quote
    using GridapGeosciences.$mod: $name; export $name
  end
end

@publish Adaptivity refine

@publish Fields Cartesian2SphericalMap
@publish Fields normal_vec

@publish Geometry pushforward_normal
@publish Geometry pushforward_reference_normal
@publish Geometry pushforward_parametric_normal
@publish Geometry BoundaryTriangulation
@publish Geometry SkeletonTriangulation
@publish Geometry generate_ptr
@publish Geometry coarse_cube_model
@publish Geometry get_radius
@publish Geometry get_thickness
@publish Geometry generate_refined_models
@publish Geometry get_surface_normal

# BEGIN AtlasDiscreteModels-specific exports
# At the present moment, I am solely exporting the symbols
# that are used in the tests. But we should eventually give
# a deeper thought to this
@publish Geometry AtlasDiscreteModel
@publish Geometry IntrinsicAtlasDiscreteModel
@publish Geometry ExtrinsicAtlasDiscreteModel
@publish Geometry AtlasGrid
@publish Geometry IntrinsicManifold
@publish Geometry ExtrinsicManifold
@publish Geometry MetricCellField
@publish Geometry InvMetricCellField
@publish Geometry MeasureCellField
@publish Geometry AmbientMapCellField
@publish Geometry LatLonMapCellField
@publish Geometry Δs
@publish Geometry vecΔs
@publish Geometry ∇s
@publish Geometry divs
@publish Geometry curls
@publish Geometry skew_∇s
@publish Geometry skew_divs

@publish Geometry CylinderMesh
@publish Geometry CylinderChartMap
@publish Geometry CylinderMetricField
@publish Geometry CylinderInvMetricField
@publish Geometry MobiusStripMesh
@publish Geometry MobiusChartMap
@publish Geometry MobiusMetricField
@publish Geometry MobiusInvMetricField
@publish Geometry CubedSphereMesh
@publish Geometry CubedSphereMap
@publish Geometry CubedSphereMetricField
@publish Geometry CubedSphereInvMetricField
@publish Geometry CubedSphereWithThicknessMesh
@publish Geometry CubedSphereWithThicknessMap
@publish Geometry ExtrudedCubedSphereWithThicknessMesh

@publish Geometry get_coarse_mesh
@publish Geometry get_cell_ambient_maps
@publish Geometry get_cell_metric
@publish Geometry get_cell_inv_metric
@publish Geometry JtJ
@publish Geometry get_atlas_grid
@publish Geometry get_ambient_dim
@publish Geometry dagger

# END AtlasDiscreteModels-specific exports


@publish ODEs DAEFEOperator

@publish Visualisation writevtk_with_cell_geomap
@publish Visualisation createvtk_with_cell_geomap

@publish Helpers xyz2θϕr

@publish Helpers sqrtg
@publish Helpers detg
@publish Helpers metric
@publish Helpers inv_metric

@publish Helpers surflap
@publish Helpers surfdiv
@publish Helpers sgrad

@publish Helpers ambient_surflap
@publish Helpers ambient_surfdiv
@publish Helpers ambient_sgrad

@publish Helpers panel_to_cartesian

@publish Helpers tangent_vec
@publish Helpers contra_v
@publish Helpers piola

@publish Helpers forward_jacobian
@publish Helpers covariant_basis
@publish Helpers forward_pinv_jacobian

@publish Helpers pinvJ
@publish Helpers perp

@publish Distributed AtlasDistributedDiscreteModel
@publish Distributed IntrinsicAtlasDistributedDiscreteModel
@publish Distributed ExtrinsicAtlasDistributedDiscreteModel
@publish Distributed AdaptedIntrinsicAtlasDistributedDiscreteModel
@publish Distributed AdaptedExtrinsicAtlasDistributedDiscreteModel
@publish Distributed AtlasOctreeDistributedDiscreteModel
@publish Distributed get_atlas_model

@publish Distributed create_pvtk_file_with_cell_geomap

@publish Distributed generate_distributed_refined_models
@publish Distributed generate_octree_distributed_refined_models
@publish Distributed generate_extruded_octree_distributed_refined_models

@publish Distributed CellField

@publish Distributed ExtrudedAtlasOctreeDistributedDiscreteModel

@publish MultilevelTools ModelHierarchy
@publish MultilevelTools adapt_model

@publish ConvergenceTools p_convergence_auto_test
@publish ConvergenceTools h_convergence_auto_test
@publish ConvergenceTools nref
@publish ConvergenceTools nc
@publish ConvergenceTools dx
