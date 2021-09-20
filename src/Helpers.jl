"""copy a FEFunction on a FE Space"""
clone_fe_function(space,f)=FEFunction(space,copy(get_free_dof_values(f)))

function new_vtk_step(Ω,file,hn,un,wn)
  createvtk(Ω,
            file,
            cellfields=["hn"=>hn, "un"=>un, "wn"=>wn],
            nsubcells=4)
end
