"""
AmbientCellField

A AmbientCellField is returns an GenericCellField object, where the cell_field is an
array of cell-wise functions.
The user must define a function that given an forward_map, defines the inverse map,
and returns a function that takes coordinates in ambient space as input, and returns the value of the function.
This means AmbientCellField is different to CellField, where the user passes
a function that takes points in physical space and returns the function evaluated in physical space.

Example usage is:

```
function ambient_sgrad(f::Function,forward_map::Field)
  function _gradf(x)
    inverse_map  = CubedSphereInverseMap(forward_map)
    αβ = inverse_map(x)
    sgrad(panel_f(f),forward_map)(αβ)
  end
end


f_cf = AmbientCellField(ambient_sgrad,Ω_ambient)
```


The input function `f` takes a CubedSphereInverseMap object, and returns another function `_f`.
Such function takes points in the ambient coordinate system of the manifold, and returns the
ambient field evaluated at parametric points.
Note, this will only work for manifolds where the inverse map can be defined analytically


"""


function AmbientCellField(f::Function,
  trian::BodyFittedTriangulation{Dc,Dp,<:CubedSphereAmbientDiscreteModel}) where {Dc,Dp}

  ambient_model = get_background_model(trian)
  panel_model = get_parametric_model(ambient_model)
  panel_ids = get_panel_ids(panel_model)
  @check length(panel_ids) == num_cells(trian) "\n Incorrect panel ids"

  fwd_map_generator = get_forward_map_generator(ambient_model)
  forward_maps = lazy_map(fwd_map_generator,panel_ids)
  cell_field = lazy_map(m->GenericField(f(m)),forward_maps)
  CellData.GenericCellField(cell_field,trian,PhysicalDomain())
end


function AmbientCellField(f::Function,atrian::AdaptedTriangulation)
  cf = AmbientCellField(f,atrian.trian)
  AmbientCellField(cf,atrian)
end

function AmbientCellField(cf::CellField,atrian::AdaptedTriangulation)
  GenericCellField(get_data(cf),atrian,DomainStyle(cf))
end
