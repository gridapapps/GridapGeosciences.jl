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


function pullback_area_form(trian::DistributedTriangulation)
  fields = map(trian.trians) do t
    pullback_area_form(t)
  end
  GridapDistributed.DistributedCellField(fields,trian)
end

function pushforward_normal(trian::GridapDistributed.DistributedTriangulation)
  fields = map(trian.trians) do t
    pushforward_normal(t)
  end
  GridapDistributed.DistributedCellField(fields,trian)
end

function pushforward_reference_normal(trian::GridapDistributed.DistributedTriangulation)
  fields = map(trian.trians) do t
    pushforward_reference_normal(t)
  end
  GridapDistributed.DistributedCellField(fields,trian)
end

function pushforward_parametric_normal(trian::GridapDistributed.DistributedTriangulation)
  fields = map(trian.trians) do t
    pushforward_parametric_normal(t)
  end
  GridapDistributed.DistributedCellField(fields,trian)
end

"""
get_sphere_surface_normal

Is the distributed implementation of get_sphere_surface_normal.
In such function, we call get_sphere_surface_normal on the local model and then
recompute the triangulation to ensure proper handling of ghost cells in octree periodic meshes.
"""
function get_sphere_surface_normal(trian::GridapDistributed.DistributedTriangulation)
  model = trian.model
  _trian = GridapDistributed.add_ghost_cells(trian)
  fields = map(_trian.trians) do t
    get_sphere_surface_normal(t)
  end
  GridapDistributed.DistributedCellField(fields,_trian)
end