function pushforward_reference_normal(trian::AdaptedTriangulation)
  pushforward_reference_normal(trian.trian)
end

function pushforward_parametric_normal(trian::AdaptedTriangulation)
  pushforward_parametric_normal(trian.trian)
end

function pushforward_normal(trian::AdaptedTriangulation)
  pushforward_normal(trian.trian)
end

function pullback_area_form(atrian::AdaptedTriangulation)
  cf = pullback_area_form(atrian.trian)
  plus = GenericCellField(get_data(cf.plus),atrian,DomainStyle(cf.plus))
  minus = GenericCellField(get_data(cf.minus),atrian,DomainStyle(cf.minus))
  SkeletonPair(plus,minus)
end
