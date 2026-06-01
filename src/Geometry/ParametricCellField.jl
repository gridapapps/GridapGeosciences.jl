"""
ParametricCellField

A ParametricCellField is returns an GenericCellField object, where the cell_field is an
array of cell-wise functions.
The user must define a function that given a forward map, returns a function
that takes coordinates in parametric space as input, and returns the value of the function.
This means ParametricCellField is different to CellField, where the user passes
a function that takes points in physical space and returns the function evaluated in physical space.

Example usage is:

```
function fX(forward_map)
  function _f(αβ)
    x = forward_map(αβ)
    x[1]*x[2]*x[3]
  end
end

f_cf = ParametricCellField(fX,Ω_panel)
```


The input function `fX` takes a CubedSphereForwardMap object, and returns another function `_f`.
Such function takes points in the local coordinate system of the chart, and returns the
ambient field evaluated at parametric points.


"""



function ParametricCellField(f::Function,
                             trian::BodyFittedTriangulation{Dc,Dp,<:CubedSphereParametricDiscreteModel},
                             panel_ids::AbstractArray{Int}) where {Dc,Dp}
  @check length(panel_ids) == num_cells(trian) "\n Incorrect panel ids"
  model = get_background_model(trian)
  fwd_map_generator = get_forward_map_generator(model)
  forward_maps = lazy_map(fwd_map_generator,panel_ids)
  cell_field = lazy_map(m->GenericField(f(m)),forward_maps)
  CellData.GenericCellField(cell_field,trian,PhysicalDomain())
end

function ParametricCellField(f::Function,trian::BodyFittedTriangulation{Dc,Dp,<:CubedSphereParametricDiscreteModel}) where {Dc,Dp}
  panel_ids = get_panel_ids(trian)
  ParametricCellField(f,trian,panel_ids)
end

### For Adapted Triangulations, return cellfield on the atrian
function ParametricCellField(f::Function,atrian::AdaptedTriangulation,panel_ids::AbstractArray{Int})
  cf = ParametricCellField(f,atrian.trian,panel_ids)
  ParametricCellField(cf,atrian)
end

function ParametricCellField(f::Function,atrian::AdaptedTriangulation)
  cf = ParametricCellField(f,atrian.trian)
  ParametricCellField(cf,atrian)
end

function ParametricCellField(cf::CellField,atrian::AdaptedTriangulation)
  GenericCellField(get_data(cf),atrian,DomainStyle(cf))
end

function ParametricCellField(cf::SkeletonPair,atrian::AdaptedTriangulation)
  plus = GenericCellField(get_data(cf.plus),atrian,ReferenceDomain())
  minus = GenericCellField(get_data(cf.minus),atrian,ReferenceDomain())
  return SkeletonPair(plus,minus)
end


function ParametricCellField(f::Function,trian::TriangulationView)
  ParametricCellField(f,trian.parent)
end


### For Skeleton and Boundary Triangulations, map with map from reference face
### Thus, return cellfield on the reference domain

### Boundary
function ParametricCellField(f::Function,trian::BoundaryTriangulation)
  face_panel_ids = get_panel_ids(trian)
  ParametricCellField(f,trian,face_panel_ids)
end

function ParametricCellField(f::Function,trian::BoundaryTriangulation,face_panel_ids::AbstractArray{Int})
  _face_cf = _boundary_cell_data(f,trian,face_panel_ids)
  face_cf = GenericCellField(_face_cf,trian,ReferenceDomain())
  return face_cf
end

function _boundary_cell_data(f::Function,trian,face_panel_ids::AbstractArray)
  # face_panel_ids = get_panel_ids(trian)

  @check length(face_panel_ids) == num_cells(trian) "\n Incorrect panel ids"

  model = get_background_model(trian)
  fwd_map_generator = get_forward_map_generator(model)
  forward_maps = lazy_map(fwd_map_generator,face_panel_ids)

  # make physical cf
  cell_field = map(m->GenericField(f(m)),forward_maps)
  cf = CellData.GenericCellField(cell_field,trian,PhysicalDomain())

  glue = get_glue(trian,Val(2))

  panel_model = get_background_model(trian)
  cmap = get_cell_map(get_grid(panel_model))
  fcmap = lazy_map(Reindex(cmap),glue.tface_to_mface)

  # compose with the map from reference face -> refernce cell -> physical cell
  ref_face_2_phys_cell_map = lazy_map(∘,fcmap,glue.tface_to_mface_map)
  face_cf = lazy_map(∘,get_data(cf),ref_face_2_phys_cell_map)
  return face_cf
end

### Skeleton - left and right boundary triangulations returned as SkeletonPair
function ParametricCellField(f::Function,trian::SkeletonTriangulation)
  # panel_ids = get_panel_ids(trian)
  ### plus
  # _face_cf_plus = _boundary_cell_data(f,trian.plus,panel_ids.plus)
  _face_cf_plus = ParametricCellField(f,trian.plus)
  plus = GenericCellField(get_data(_face_cf_plus),trian,ReferenceDomain())

  #### minus
  # _face_cf_minus = _boundary_cell_data(f,trian.minus,panel_ids.minus)
  _face_cf_minus = ParametricCellField(f,trian.minus)
  minus = GenericCellField(get_data(_face_cf_minus),trian,ReferenceDomain())

  SkeletonPair(plus,minus)
end
