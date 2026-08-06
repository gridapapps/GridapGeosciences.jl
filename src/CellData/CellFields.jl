"""
    MetricCellField(trian)

Return the per-cell pullback metric `g` stored in the underlying `AtlasGrid` as a
`CellField` on `trian`.  Each cell's metric is a `SymTensorValue{Dc,Dc}` field
evaluated in chart coordinates.

Use `InvMetricCellField(Ω)` for `g⁻¹` (preferred over `Operation(inv)(MetricCellField(Ω))`
for built-in shapes — uses the explicit analytic formula).
Use `Operation(x -> sqrt(det(x)))(MetricCellField(Ω))` for `√det g`.
This is the correct intrinsic source for the metric — it uses the analytic metric
stored in `CoarseMeshInfo.metric_fields`, independent of any ambient embedding.
"""
function MetricCellField(
    trian :: BFTATDM{Dc,Dp},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_metric(model),
                                   trian,
                                   Gridap.CellData.PhysicalDomain())
end

function MetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = MetricCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function MetricCellField(
    trian ::  Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  Gridap.CellData.GenericCellField(get_cell_metric(trian.trian.grid),
                                   trian,
                                   Gridap.CellData.PhysicalDomain())
end

function MetricCellField(
    trian :: SkeletonTriangulation{Dc,Dp,<:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  m_cf_plus_bt = MetricCellField(trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = MetricCellField(trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function MetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:SkeletonTriangulation{Dc,Dp,
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}},
) where {Dc,Dp}
  m_cf_plus_bt = MetricCellField(trian.trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = MetricCellField(trian.trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function MetricCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = MetricCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function MetricCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  cf = MetricCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function MeasureCellField(trian :: BFTATDM{Dc,Dp}) where {Dc,Dp}
    sqrt∘det∘MetricCellField(trian)
end

function MeasureCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = MeasureCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function MeasureCellField(
    trian ::  Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  sqrt∘det∘MetricCellField(trian)
end

function MeasureCellField(
    trian :: SkeletonTriangulation{Dc,Dp,<:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  m_cf_plus_bt = MeasureCellField(trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = MeasureCellField(trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function MeasureCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:SkeletonTriangulation{Dc,Dp,
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}},
) where {Dc,Dp}
  m_cf_plus_bt = MeasureCellField(trian.trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = MeasureCellField(trian.trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function MeasureCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  sqrt∘det∘MetricCellField(trian)
end

function MeasureCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  sqrt∘det∘MetricCellField(trian)
end

"""
    InvMetricCellField(trian)

Return the per-cell inverse pullback metric `g⁻¹` as a `CellField` on `trian`.
For built-in shapes (`CylinderMesh`, `MobiusStripMesh`, `CubedSphereMesh`) this
uses an explicit analytic formula via `inverse_metric_field`; the generic fallback
applies `Operation(inv)` at each quadrature point.
"""
function InvMetricCellField(
    trian :: BFTATDM{Dc,Dp},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_inv_metric(model), trian, Gridap.CellData.PhysicalDomain())
end

function InvMetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = InvMetricCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function InvMetricCellField(
    trian ::  Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  Gridap.CellData.GenericCellField(get_cell_inv_metric(trian.trian.grid),
                                   trian,
                                   Gridap.CellData.PhysicalDomain())
end

function InvMetricCellField(
    trian :: SkeletonTriangulation{Dc,Dp,<:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  m_cf_plus_bt = InvMetricCellField(trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = InvMetricCellField(trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function InvMetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:SkeletonTriangulation{Dc,Dp,
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}},
) where {Dc,Dp}
  m_cf_plus_bt = InvMetricCellField(trian.trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = InvMetricCellField(trian.trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function InvMetricCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = InvMetricCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function InvMetricCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  cf = InvMetricCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function AmbientMapCellField(
    trian :: BFTATDM{Dc,Dp},
) where {Dc,Dp}
  model = Gridap.Geometry.get_background_model(trian)
  Gridap.CellData.GenericCellField(get_cell_ambient_maps(model), trian, Gridap.CellData.PhysicalDomain())
end

function AmbientMapCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = AmbientMapCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function AmbientMapCellField(
    trian :: Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  Gridap.CellData.GenericCellField(get_cell_ambient_maps(trian.trian.grid),
                                   trian,
                                   Gridap.CellData.PhysicalDomain())
end

function AmbientMapCellField(
    trian :: SkeletonTriangulation{Dc,Dp,<:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  m_cf_plus_bt = AmbientMapCellField(trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = AmbientMapCellField(trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function AmbientMapCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:SkeletonTriangulation{Dc,Dp,
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}},
) where {Dc,Dp}
  m_cf_plus_bt = AmbientMapCellField(trian.trian.plus)
  plus = GenericCellField(Gridap.CellData.get_data(m_cf_plus_bt), trian, Gridap.CellData.DomainStyle(m_cf_plus_bt))
  m_cf_minus_bt = AmbientMapCellField(trian.trian.minus)
  minus = GenericCellField(Gridap.CellData.get_data(m_cf_minus_bt), trian, Gridap.CellData.DomainStyle(m_cf_minus_bt))
  SkeletonPair(plus,minus)
end

function AmbientMapCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = AmbientMapCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

function AmbientMapCellField(
    trian :: Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
) where {Dc,Dp}
  cf = AmbientMapCellField(trian.parent)
  data = lazy_map(Reindex(get_data(cf)), trian.cell_to_parent_cell)
  Gridap.CellData.GenericCellField(data, trian, Gridap.CellData.DomainStyle(cf))
end

# Only supported for the cubed sphere mesh
# For visualisation purposes
function LatLonMapCellField(trian::Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},
                                        C,
                                        O,
                                        M}}) where {Dc,Da,G,A,C,O,M}
  Operation(Cartesian2SphericalMap())(AmbientMapCellField(trian))
end

function LatLonMapCellField(trian::AdaptedTriangulation{Dc,Da,<:Gridap.Geometry.BodyFittedTriangulation{Dc,Da,
                                        <:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},
                                        C,
                                        O,
                                        M}}}) where {Dc,Da,G,A,C,O,M}
  cf = LatLonMapCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

# Right now only supported by AtlasDiscreteModel of the sphere
function InvAmbientMapCellField(
    trian :: AdaptedTriangulation{Dc,Da,<:Gridap.Geometry.BodyFittedTriangulation{Dc,Da,
                                        <:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:CubedSphereMap},
                                        C,
                                        O,
                                        <:ExtrinsicManifold}}},
) where {Dc,Da,G,A,C,O}
  cf = InvAmbientMapCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function InvAmbientMapCellField(
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:CubedSphereMap},
                                        C,
                                        O,
                                        <:ExtrinsicManifold}}) where {Dc,Da,G,A,C,O}
  model = Gridap.Geometry.get_background_model(trian)
  cell_ambient_maps = get_cell_ambient_maps(model)
  ptrs = cell_ambient_maps.ptrs
  ambient_maps = cell_ambient_maps.values
  radius = ambient_maps[1].radius
  inv_ambient_maps = [CubedSphereInvMap(panel, radius) for panel in 1:length(ambient_maps)]
  cell_inv_ambient_maps = CompressedArray(inv_ambient_maps, ptrs)
  Gridap.CellData.GenericCellField(cell_inv_ambient_maps, trian, Gridap.CellData.PhysicalDomain())
end

# Right now only supported by AtlasDiscreteModel of the sphere
function InvAmbientMapCellField(
    trian :: AdaptedTriangulation{Dc,Da,<:Gridap.Geometry.BodyFittedTriangulation{Dc,Da,
                                        <:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:CubedSphereWithThicknessMap},
                                        C,
                                        O,
                                        <:ExtrinsicManifold}}},
) where {Dc,Da,G,A,C,O}
  cf = InvAmbientMapCellField(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

function InvAmbientMapCellField(
    trian :: Gridap.Geometry.BodyFittedTriangulation{Dc,Da,<:AtlasDiscreteModel{Dc,Da,
                                        G,
                                        A,
                                        <:AbstractVector{<:CubedSphereWithThicknessMap},
                                        C,
                                        O,
                                        <:ExtrinsicManifold}}) where {Dc,Da,G,A,C,O}
  model = Gridap.Geometry.get_background_model(trian)
  cell_ambient_maps = get_cell_ambient_maps(model)
  ptrs = cell_ambient_maps.ptrs
  ambient_maps = cell_ambient_maps.values
  radius = ambient_maps[1].radius
  inv_ambient_maps =
     [CubedSphereWithThicknessInvMap(panel, radius, ambient_maps[panel].thickness) for panel in 1:length(ambient_maps)]
  cell_inv_ambient_maps = CompressedArray(inv_ambient_maps, ptrs)
  Gridap.CellData.GenericCellField(cell_inv_ambient_maps, trian, Gridap.CellData.PhysicalDomain())
end

## The perp of the metric. i.e. R*g. Probably a cleaner way to do this with OperationCellFields
## Just leaving it here now ...
perp_metric(m::Field) = x -> perp(metric(m,x))

function PerpMetricCellField(Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}) where {Dc, Da, G, A, P, C, O}
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(perp_metric(m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function PerpMetricCellField(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = PerpMetric(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end
