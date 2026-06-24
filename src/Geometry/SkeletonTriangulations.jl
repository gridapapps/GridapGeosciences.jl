
function pushforward_normal(trian::SkeletonTriangulation{Dc,Dp,
                                                         <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}) where {Dc,Dp}
  plus = pushforward_normal(trian.plus)
  minus = pushforward_normal(trian.minus)
  SkeletonPair(plus,minus)
end

function pushforward_reference_normal(trian::SkeletonTriangulation{Dc,Dp,
                                                           <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}) where {Dc,Dp}
  plus = pushforward_reference_normal(trian.plus)
  minus = pushforward_reference_normal(trian.minus)
  SkeletonPair(plus,minus)
end

function pushforward_parametric_normal(trian::SkeletonTriangulation{Dc,Dp,
                                                           <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}) where {Dc,Dp}
  plus = pushforward_parametric_normal(trian.plus)
  minus = pushforward_parametric_normal(trian.minus)
  SkeletonPair(plus,minus)
end

function pullback_area_form(trian::SkeletonTriangulation{Dc,Dp,
                                                           <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}, <:BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}) where {Dc,Dp}
  plus = pullback_area_form(trian.plus)
  minus = pullback_area_form(trian.minus)
  SkeletonPair(plus,minus)
end