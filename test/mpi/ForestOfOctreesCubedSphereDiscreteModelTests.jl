module ForestOfOctreesCubedSphereDiscreteModelTests
    using GridapGeosciences
    using GridapP4est
    using PartitionedArrays
    using MPI
    using Gridap 
    using GridapDistributed

    function f(x)
        sin(2*pi*x[1])*cos(2*pi*x[2])*cos(2*pi*x[3])
    end

    function calculate_error_indicators(model,fh)
        Ω=Triangulation(model)
        dΩ=Measure(Ω,10)
        eh=fh-f
        dc=∫(eh*eh)*dΩ
        error_indicators=map(local_views(dc)) do dc 
            sqrt.(get_array(dc))
        end
    end

    function adapt_model(ranks,model,error_indicators)
        cell_partition=get_cell_gids(model.octree_model.dmodel)
        ref_coarse_flags=map(ranks,partition(cell_partition)) do rank,indices
            flags=zeros(Cint,length(indices))
            flags.=nothing_flag        
        end
        ref_fraction=0.2
        coarsen_fraction=0.05
        adaptive_strategy=FixedFractionAdaptiveFlagsMarkingStrategy(ref_fraction,coarsen_fraction)
        update_adaptivity_flags!(ref_coarse_flags,
                                adaptive_strategy,
                                partition(cell_partition),
                                error_indicators;
                                verbose=true)
        GridapP4est.adapt(model,ref_coarse_flags)
    end 

    function setup_fe_space(model)
        reffe=ReferenceFE(lagrangian,Float64,1)
        Vh=FESpace(model,reffe)
    end 

    with_mpi() do distribute
        ranks=distribute(LinearIndices((MPI.Comm_size(MPI.COMM_WORLD),)))
        model=CubedSphereDiscreteModel(ranks,4;adaptive=true)
        Vh=setup_fe_space(model)
        for step=1:4
            fh=interpolate(f,Vh)
            error_indicators=calculate_error_indicators(model,fh)
            model,_=adapt_model(ranks, model, error_indicators)
            Vh=setup_fe_space(model)
            fh=interpolate(f,Vh)
            writevtk(Triangulation(model),"cubed_sphere_amr_step_$(step)",cellfields=["f"=>fh])
        end
    end  
end
