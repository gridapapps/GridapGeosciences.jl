"""
To visualise on the sphere (and other mapped domains), we trick vtk to plot
cellvalues on a visualisation mesh that is different to the one used for
evaluation of cell/node data.
This is achieved by evaluating a cellwise geo_map in _vtkpoints.
"""
function writevtk_with_cell_geomap(geo_map::CellField,trian::Triangulation,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)
  @check isa(DomainStyle(geo_map),PhysicalDomain)
  geo_map_to_trian = change_domain(geo_map,trian,DomainStyle(geo_map))
  map(Gridap.Visualization.visualization_data(trian, args...;kwargs...)) do visdata
    write_vtk_file_with_cell_geomap(geo_map_to_trian,
      visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata,
      compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
    )
  end
end

function createvtk_with_cell_geomap(geo_map::CellField,trian::Triangulation,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)
  @check isa(DomainStyle(geo_map),PhysicalDomain)
  geo_map_to_trian = change_domain(geo_map,trian,DomainStyle(geo_map))
  v = Gridap.Visualization.visualization_data(args...;kwargs...)
  @notimplementedif length(v) != 1
  visdata = first(v)
  create_vtk_file_with_cell_geomap(geo_map_to_trian,
    visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )
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


function write_vtk_file_with_cell_geomap(
  geo_map::CellField,
  vis_data_grid::Grid, filebase; celldata=Dict(), nodaldata=Dict(),
  compress=false, append=true, ascii=false, vtkversion=:default
)
  vtkfile = create_vtk_file_with_cell_geomap(geo_map,
    vis_data_grid, filebase, celldata=celldata, nodaldata=nodaldata,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )
  outfiles = Gridap.Visualization.vtk_save(vtkfile)
end

function create_vtk_file_with_cell_geomap(geo_map::CellField,
  vis_data_grid::Grid, filebase; celldata=Dict(), nodaldata=Dict(),
  compress=false, append=true, ascii=false, vtkversion=:default
)

  ## Map the points to ambient space
  points = mapped_vtkpoints(vis_data_grid,geo_map)

  cells = Gridap.Visualization._vtkcells(vis_data_grid)
  vtkfile = Gridap.Visualization.vtk_grid(
    filebase, points, cells,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )

  if num_cells(vis_data_grid)>0
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
function mapped_vtkpoints(vis_data_grid,geo_map::CellField)
  # get the cell coordinates of the vis_data_grid.
  # these are DG-style coordinates in the parametric domain of 
  # of the manifold, and thus suitable for geo_map evaluation
  cellx = get_cell_coordinates(vis_data_grid)

  # apply the geo_map to cell_coords on vis_data_grid, then convert to nodes
  cellx_mapped = lazy_map(evaluate,get_data(geo_map),cellx)

  x_mapped, _ = Gridap.Visualization._prepare_node_to_coords(cellx_mapped)

  T = eltype(x_mapped)
  D = num_components(T)

  xflat = collect(x_mapped)
  reshape(reinterpret(Float64,xflat),(D,length(x_mapped)))
end
