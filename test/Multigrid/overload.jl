using Gridap.Geometry
using Gridap.Adaptivity
using Gridap.CellData
using GridapGeosciences
import GridapGeosciences.Geometry: BFTATDM
using GridapDistributed

function GridapGeosciences.CellData.MetricCellField(
    trian::Union{PatchTriangulation{Dc,Dp,<:BFTATDM},PatchTriangulation{Dc,Dp,<:AdaptedTriangulation{Dc,Dp,<:BFTATDM } }}
) where {Dc,Dp}
  println("patch overload")
  bmodel = get_background_model(trian)
  bmodel_trian = Triangulation(bmodel)

  Gridap.CellData.GenericCellField(get_cell_metric(bmodel), bmodel_trian,  Gridap.CellData.PhysicalDomain())
end

function GridapGeosciences.CellData.MeasureCellField(
    trian::Union{PatchTriangulation{Dc,Dp,<:BFTATDM},PatchTriangulation{Dc,Dp,<:AdaptedTriangulation{Dc,Dp,<:BFTATDM } }}
) where {Dc,Dp}
  println("patch overload")
    sqrt∘det∘MetricCellField(trian)
end


function GridapGeosciences.CellData.MetricCellField(
    trian::GridapDistributed.DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{PatchTriangulation{Dc,Dp,<:BFTATDM},
                                    PatchTriangulation{Dc,Dp,<:AdaptedTriangulation{Dc,Dp,<:BFTATDM}}} }}
) where {Dc,Dp}
  ghosted_trian = GridapDistributed.add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function GridapGeosciences.CellData.MeasureCellField(
    trian::GridapDistributed.DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{PatchTriangulation{Dc,Dp,<:BFTATDM},
                                    PatchTriangulation{Dc,Dp,<:AdaptedTriangulation{Dc,Dp,<:BFTATDM}}} }}
) where {Dc,Dp}
  ghosted_trian = GridapDistributed.add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end
