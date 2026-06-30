function Δs(f::Function,
            trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                           Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}};
            use_automatic_differentiation=false) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    Δs(f, t; use_automatic_differentiation)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function ∇s(f::Function,
            trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                           Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}};
            use_automatic_differentiation=false) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    ∇s(f, t; use_automatic_differentiation)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end