

function get_surface_normal(trian::AdaptedTriangulation)
  ns = get_surface_normal(trian.trian)
  GenericCellField(get_data(ns),trian,DomainStyle(ns))
end
