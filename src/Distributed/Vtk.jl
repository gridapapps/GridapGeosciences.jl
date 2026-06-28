function write_vtk_file_with_cell_geomap(geo_map::AbstractArray,
  parts::AbstractArray,
  grid::AbstractArray{<:Gridap.Geometry.Grid}, filebase; celldata, nodaldata,
  compress=false,append=true,ascii=false,vtkversion=:default
  )
  pvtk = create_vtk_file_with_cell_geomap(geo_map,
    parts,grid,filebase;celldata=celldata,nodaldata=nodaldata,
    compress=compress,append=append,ascii=ascii,vtkversion=vtkversion
  )
  map(Gridap.Visualization.vtk_save,pvtk)
end

function create_vtk_file_with_cell_geomap(geo_map::AbstractArray,
  parts::AbstractArray,
  grid::AbstractArray{<:Gridap.Geometry.Grid},
  filebase;
  celldata, nodaldata,
  compress=false,append=true,ascii=false,vtkversion=:default
)
  nparts = length(parts)
  map(parts,grid,celldata,nodaldata,geo_map) do part,g,c,n,gm
    create_pvtk_file_with_cell_geomap(gm,
      g,filebase;
      part=part,nparts=nparts,
      celldata=c,nodaldata=n,
      compress=compress,append=append,ascii=ascii,vtkversion=vtkversion
    )
  end
end

function writevtk_with_cell_geomap(geo_map::DistributedCellField,
  arg::GridapDistributed.DistributedTriangulation,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...
)
  gid = get_cell_gids(arg.model)
  owned_geo_maps = map(local_views(geo_map),partition(gid)) do gm,cids
    owned_cells = own_to_local(cids)
    get_data(gm)[owned_cells]
  end

  writevtk_with_cell_geomap(owned_geo_maps, arg, args...;
    compress=compress,append=append,ascii=ascii,vtkversion=vtkversion,kwargs...)
end

function createvtk_with_cell_geomap(geo_map::DistributedCellField,
  arg::GridapDistributed.DistributedTriangulation,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...
)
  gid = get_cell_gids(arg.model)
  owned_geo_maps = map(local_views(geo_map),partition(gid)) do gm,cids
    owned_cells = own_to_local(cids)
    get_data(gm)[owned_cells]
  end

  createvtk_with_cell_geomap(owned_geo_maps, arg, args...;
    compress=compress,append=append,ascii=ascii,vtkversion=vtkversion,kwargs...)
end

function writevtk_with_cell_geomap(geo_map::AbstractArray,
  arg::GridapDistributed.DistributedModelOrTriangulation,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...
)
  parts=get_parts(arg)
  map(GridapDistributed.visualization_data(arg,args...;kwargs...)) do visdata
    write_vtk_file_with_cell_geomap(geo_map,
      parts,visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata,
      compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
    )
  end
end

function createvtk_with_cell_geomap(geo_map::AbstractArray,
  arg::GridapDistributed.DistributedModelOrTriangulation,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...
)
  v = Gridap.Visualization.visualization_data(arg,args...;kwargs...)
  parts=get_parts(arg)
  @Gridap.Helpers.notimplementedif length(v) != 1
  visdata = first(v)
  create_vtk_file_with_cell_geomap(geo_map,
    parts,visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )
end

function create_pvtk_file_with_cell_geomap(geo_map::AbstractArray,
  trian::Gridap.Geometry.Grid, filebase; part, nparts, ismain=(part==1), celldata=Dict(), nodaldata=Dict(),
  compress=false, append=true, ascii=false, vtkversion=:default
)
  # println("my distributed vis")

  # println(typeof(geo_map)<:AbstractArray)
  ## Map the points to ambient space
  points = mapped_vtkpoints(trian,geo_map)


  cells = Gridap.Visualization._vtkcells(trian)
  vtkfile = Gridap.Visualization.pvtk_grid(
    filebase, points, cells;part=part, nparts=nparts, ismain=ismain,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )

  if num_cells(trian) > 0
    for (k, v) in celldata
      # component_names are actually always nothing as there are no field in ptvk atm
      component_names = Gridap.Visualization._data_component_names(v)
      vtkfile[k, Gridap.Visualization.VTKCellData(), component_names=component_names] = Gridap.Visualization._prepare_data(v)
    end
    for (k, v) in nodaldata
      component_names = Gridap.Visualization._data_component_names(v)
      vtkfile[k, Gridap.Visualization.VTKPointData(), component_names=component_names] = Gridap.Visualization._prepare_data(v)
    end
  end
  return vtkfile
end
