function LatLonMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{
              Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel{Dc,Dp,G,A,
                <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},C,O,M}},
              Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,
                <:AtlasDiscreteModel{Dc,Dp,G,A,
                  <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},C,O,M}}}}}}
) where {Dc,Dp,G,A,C,O,M}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    LatLonMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                   Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{
              Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
              Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                   Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{
              Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
              Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function InvMetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                   Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    InvMetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function InvMetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    InvMetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function InvMetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    InvMetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                   Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{
              Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
              Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end