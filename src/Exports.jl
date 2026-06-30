macro publish(mod,name)
  quote
    using GridapGeosciences.$mod: $name; export $name
  end
end

@publish Adaptivity refine

@publish Fields Cartesian2SphericalMap

@publish Geometry pushforward_normal
@publish Geometry pushforward_reference_normal
@publish Geometry pushforward_parametric_normal
@publish Geometry pullback_area_form
@publish Geometry BoundaryTriangulation
@publish Geometry SkeletonTriangulation
@publish Geometry get_radius
@publish Geometry get_thickness
@publish Geometry generate_refined_models
@publish Geometry sphere_surface_normal_vec
@publish Geometry sphere_tangent_vec_component
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

@publish CellData MetricCellField
@publish CellData InvMetricCellField
@publish CellData MeasureCellField
@publish CellData AmbientMapCellField
@publish CellData LatLonMapCellField
@publish CellData Δs
@publish CellData vecΔs
@publish CellData ∇s
@publish CellData divs
@publish CellData curls
@publish CellData skew_∇s
@publish CellData skew_divs
@publish CellData dagger


@publish Geometry CylinderMesh
@publish Geometry MobiusStripMesh
@publish Geometry CubedSphereMesh
@publish Geometry CubedSphereWithThicknessMesh
@publish Geometry ExtrudedCubedSphereWithThicknessMesh


@publish Fields   CylinderMap
@publish Fields   CylinderMetric
@publish Fields   CylinderInvMetric
@publish Fields   MobiusMap
@publish Fields   MobiusMetric
@publish Fields   MobiusInvMetric
@publish Fields   CubedSphereMap
@publish Fields   CubedSphereInvMap
@publish Fields   CubedSphereMetric
@publish Fields   CubedSphereInvMetric
@publish Fields   CubedSphereWithThicknessMap
@publish Fields   CubedSphereWithThicknessInvMap

@publish Geometry get_coarse_mesh
@publish Geometry get_cell_ambient_maps
@publish Geometry get_cell_metric
@publish Geometry get_cell_inv_metric
@publish Geometry JtJ
@publish Geometry get_atlas_grid
@publish Geometry get_ambient_dim

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


@publish Helpers forward_jacobian
@publish Helpers covariant_basis

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
