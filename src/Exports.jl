macro publish(mod,name)
  quote
    using GridapGeosciences.$mod: $name; export $name
  end
end

# Adaptivity
@publish Adaptivity refine

# Fields
@publish Fields Cartesian2SphericalMap
@publish Fields CylinderMap
@publish Fields CylinderMetric
@publish Fields CylinderInvMetric
@publish Fields MobiusMap
@publish Fields MobiusMetric
@publish Fields MobiusInvMetric
@publish Fields CubedSphereMap
@publish Fields CubedSphereInvMap
@publish Fields CubedSphereMetric
@publish Fields CubedSphereInvMetric
@publish Fields CubedSphereWithThicknessMetric
@publish Fields CubedSphereWithThicknessInvMetric
@publish Fields CubedSphereWithThicknessMap
@publish Fields CubedSphereWithThicknessInvMap

# Geometry
@publish Geometry generate_refined_models
@publish Geometry AtlasDiscreteModel
@publish Geometry IntrinsicAtlasDiscreteModel
@publish Geometry ExtrinsicAtlasDiscreteModel
@publish Geometry AtlasGrid
@publish Geometry IntrinsicManifold
@publish Geometry ExtrinsicManifold
@publish Geometry CylinderMesh
@publish Geometry MobiusStripMesh
@publish Geometry CubedSphereMesh
@publish Geometry CubedSphereWithThicknessMesh
@publish Geometry ExtrudedCubedSphereWithThicknessMesh
@publish Geometry get_coarse_mesh
@publish Geometry get_cell_ambient_maps
@publish Geometry get_cell_metric
@publish Geometry get_cell_inv_metric
@publish Geometry JtJ
@publish Geometry get_atlas_grid
@publish Geometry get_ambient_dim

# CellData
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
@publish CellData vecΔs_2D
@publish CellData pushforward_normal
@publish CellData pushforward_reference_normal
@publish CellData pushforward_parametric_normal
@publish CellData pullback_area_form
@publish CellData get_sphere_surface_normal

# ODEs
@publish ODEs DAEFEOperator

# Visualisation
@publish Visualisation writevtk_with_cell_geomap
@publish Visualisation createvtk_with_cell_geomap

# Helpers
@publish Helpers xyz2θϕr
@publish Helpers pinvJ
@publish Helpers perp
@publish Helpers sphere_surface_normal_vec
@publish Helpers sphere_tangent_vec_component

# Distributed
@publish Distributed AtlasDistributedDiscreteModel
@publish Distributed IntrinsicAtlasDistributedDiscreteModel
@publish Distributed ExtrinsicAtlasDistributedDiscreteModel
@publish Distributed AdaptedIntrinsicAtlasDistributedDiscreteModel
@publish Distributed AdaptedExtrinsicAtlasDistributedDiscreteModel
@publish Distributed AtlasOctreeDistributedDiscreteModel
@publish Distributed ExtrudedAtlasOctreeDistributedDiscreteModel
@publish Distributed get_atlas_model
@publish Distributed create_pvtk_file_with_cell_geomap
@publish Distributed generate_distributed_refined_models
@publish Distributed generate_octree_distributed_refined_models
@publish Distributed generate_extruded_octree_distributed_refined_models

# MultilevelTools
@publish MultilevelTools adapt_model

# ConvergenceTools
@publish ConvergenceTools p_convergence_auto_test
@publish ConvergenceTools h_convergence_auto_test
@publish ConvergenceTools nref
@publish ConvergenceTools nc
@publish ConvergenceTools dx
@publish ConvergenceTools get_sphere_radius
@publish ConvergenceTools get_sphere_thickness
