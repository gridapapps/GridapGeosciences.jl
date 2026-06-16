"""
To visualise on the sphere (and other mapped domains), we trick vtk to plot
cellvalues on a visualisation mesh that is different to the one used for
evaluation of cell/node data.
This is achieved by evaluating a cellwise geo_map in _vtkpoints.
"""

"""
when geo_map <: CellField, extract the array of maps
"""
function writevtk_with_cell_geomap(geo_map::CellField,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)
  writevtk_with_cell_geomap(get_data(geo_map),args...;
  compress=compress,append=append,ascii=ascii,vtkversion=vtkversion,kwargs...)
end

function createvtk_with_cell_geomap(geo_map::CellField,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)
  createvtk_with_cell_geomap(get_data(geo_map),args...;
  compress=compress,append=append,ascii=ascii,vtkversion=vtkversion,kwargs...)
end

"""
when geo_map <: SkeletonPair, extract the cellfield of the plus side, which is
used for visualisation
"""
function writevtk_with_cell_geomap(geo_map::SkeletonPair{<:CellField},args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)
  writevtk_with_cell_geomap(geo_map.plus,args...;
  compress=compress,append=append,ascii=ascii,vtkversion=vtkversion,kwargs...)
end

function createvtk_with_cell_geomap(geo_map::SkeletonPair{<:CellField},args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)
  createvtk_with_cell_geomap(geo_map.plus,args...;
  compress=compress,append=append,ascii=ascii,vtkversion=vtkversion,kwargs...)
end


function writevtk_with_cell_geomap(geo_map::AbstractArray,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)
  map(Gridap.Visualization.visualization_data(args...;kwargs...)) do visdata
    write_vtk_file_with_cell_geomap(geo_map,
      visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata,
      compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
    )
  end
end

"""
"""
function createvtk_with_cell_geomap(geo_map::AbstractArray,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)
  v = Gridap.Visualization.visualization_data(args...;kwargs...)
  @notimplementedif length(v) != 1
  visdata = first(v)
  create_vtk_file_with_cell_geomap(geo_map,
    visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )
end

function write_vtk_file_with_cell_geomap(
  geo_map::AbstractArray,
  trian::Grid, filebase; celldata=Dict(), nodaldata=Dict(),
  compress=false, append=true, ascii=false, vtkversion=:default
)
  vtkfile = create_vtk_file_with_cell_geomap(geo_map,
    trian, filebase, celldata=celldata, nodaldata=nodaldata,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )
  outfiles = Gridap.Visualization.vtk_save(vtkfile)
end

function create_vtk_file_with_cell_geomap(geo_map::AbstractArray,
  trian::Grid, filebase; celldata=Dict(), nodaldata=Dict(),
  compress=false, append=true, ascii=false, vtkversion=:default
)

  ## Map the points to ambient space
  points = mapped_vtkpoints(trian,geo_map)

  cells = Gridap.Visualization._vtkcells(trian)
  vtkfile = Gridap.Visualization.vtk_grid(
    filebase, points, cells,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )

  if num_cells(trian)>0
    for (k,v) in celldata
      component_names = Gridap.Visualization._data_component_names(v)
      Gridap.Visualization.vtk_cell_data(vtkfile, Gridap.Visualization._prepare_data(v), k; component_names)
    end
    for (k,v) in nodaldata
      component_names = Gridap.Visualization._data_component_names(v)
      Gridap.Visualization.vtk_point_data(vtkfile, Gridap.Visualization._prepare_data(v), k; component_names)
    end
  end

  return vtkfile
end


## Can map directly the coordinates of the trian.
## This is because these coords are evaluated from the cell maps on the ref points
## See https://github.com/gridap/Gridap.jl/blob/b75e623687b6df5de2b49952bbd794e85193c70a/src/Visualization/VisualizationData.jl#L78
function mapped_vtkpoints(trian,geo_map::AbstractArray)
  # println("mapped vkpoints")

  # apply the geo_map to cell_coords on trian, then convert to nodes
  cellx = get_cell_coordinates(trian)

  cellx_mapped = lazy_map(evaluate,geo_map,cellx)

  # println(cellx_mapped)
  x_mapped, cell_to_offset = Gridap.Visualization._prepare_node_to_coords(cellx_mapped)

  T = eltype(x_mapped)
  D = num_components(T)

  xflat = collect(x_mapped)
  reshape(reinterpret(Float64,xflat),(D,length(x_mapped)))
end
