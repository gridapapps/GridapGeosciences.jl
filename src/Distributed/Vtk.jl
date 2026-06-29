function _change_domain_geo_map(geo_map, trian)
 geo_map_to_trian_cell_fields = 
    map(local_views(geo_map), local_views(trian)) do gm, trian
      change_domain(gm,trian,DomainStyle(gm))
    end
 DistributedCellField(geo_map_to_trian_cell_fields, trian) 
end 


function writevtk_with_cell_geomap(geo_map::DistributedCellField,
  trian::GridapDistributed.DistributedTriangulation,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...)

  @check isa(DomainStyle(geo_map),PhysicalDomain)
  geo_map_to_trian = _change_domain_geo_map(geo_map, trian)

  parts=get_parts(trian)
  map(GridapDistributed.visualization_data(trian,args...;kwargs...)) do visdata
    write_vtk_file_with_cell_geomap(geo_map_to_trian,
      parts,visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata,
      compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
    )
  end
end

function createvtk_with_cell_geomap(geo_map::DistributedCellField,
  trian::GridapDistributed.DistributedTriangulation,args...;
  compress=false,append=true,ascii=false,vtkversion=:default,kwargs...
)
  @check isa(DomainStyle(geo_map),PhysicalDomain)
  geo_map_to_trian = _change_domain_geo_map(geo_map, trian)
  
  v = Gridap.Visualization.visualization_data(trian, args...;kwargs...)
  parts=get_parts(trian)
  @Gridap.Helpers.notimplementedif length(v) != 1
  visdata = first(v)
  create_vtk_file_with_cell_geomap(geo_map_to_trian,
    parts,visdata.grid,visdata.filebase,celldata=visdata.celldata,nodaldata=visdata.nodaldata,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )
end

function write_vtk_file_with_cell_geomap(geo_map::DistributedCellField,
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

function create_vtk_file_with_cell_geomap(geo_map::DistributedCellField,
  parts::AbstractArray,
  grid::AbstractArray{<:Gridap.Geometry.Grid},
  filebase;
  celldata, nodaldata,
  compress=false,append=true,ascii=false,vtkversion=:default
)
  nparts = length(parts)
  map(parts,grid,celldata,nodaldata,local_views(geo_map)) do part,g,c,n,gm
    create_pvtk_file_with_cell_geomap(gm,
      g,
      filebase;
      part=part,nparts=nparts,
      celldata=c,nodaldata=n,
      compress=compress,append=append,ascii=ascii,vtkversion=vtkversion
    )
  end
end


function create_pvtk_file_with_cell_geomap(geo_map::CellField,
  vis_data_grid::Gridap.Geometry.Grid, filebase; part, nparts, ismain=(part==1), celldata=Dict(), nodaldata=Dict(),
  compress=false, append=true, ascii=false, vtkversion=:default
)
  ## Map the points to ambient space
  points = mapped_vtkpoints(vis_data_grid,geo_map)

  cells = Gridap.Visualization._vtkcells(vis_data_grid)
  vtkfile = Gridap.Visualization.pvtk_grid(
    filebase, points, cells;part=part, nparts=nparts, ismain=ismain,
    compress=compress, append=append, ascii=ascii, vtkversion=vtkversion
  )

  if num_cells(vis_data_grid) > 0
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