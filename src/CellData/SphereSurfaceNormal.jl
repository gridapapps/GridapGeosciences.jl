function get_sphere_surface_normal(
    trian::Gridap.Geometry.BodyFittedTriangulation{Dc,3,<:ExtrinsicAtlasDiscreteModel{Dcm,3,G,A,<:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}}}}
) where {Dc,Dcm,G,A}
  CellField(sphere_surface_normal_vec, trian)
end

function get_sphere_surface_normal(trian::AdaptedTriangulation)
  ns = get_sphere_surface_normal(trian.trian)
  GenericCellField(get_data(ns),trian,DomainStyle(ns))
end