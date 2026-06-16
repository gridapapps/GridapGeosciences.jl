const AtlasDistributedDiscreteModel{Dc,Dp,G,A,P,C,O,M} = 
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:AtlasDiscreteModel{Dc,Dp,G,A,P,C,O,M}}}
const IntrinsicAtlasDistributedDiscreteModel{Dc,Dp,G,A,P,C,O} = 
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:IntrinsicAtlasDiscreteModel{Dc,Dp,G,A,P,C,O}}}
const ExtrinsicAtlasDistributedDiscreteModel{Dc,Dp,G,A,P,C,O} = 
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:ExtrinsicAtlasDiscreteModel{Dc,Dp,G,A,P,C,O}}}


function AtlasDiscreteModel(ranks::AbstractArray{<:Integer},
                            mesh::CoarseMesh,
                            num_refinements::Int;
                            orientation_style=nothing,
                            manifold_style=ExtrinsicManifold())

    # Build the full serial AtlasDiscreteModel on every rank.
    # Not scalable to very large cell counts, but correct for moderate meshes —
    # the same trade-off as GridapDistributed.Geometry.DiscreteModel(parts, serial_model, ...).
    info               = get_coarse_mesh(mesh)
    serial_model       = AtlasDiscreteModel(info, num_refinements;
                                            orientation_style, manifold_style)
    g                  = get_atlas_grid(serial_model)
    ncells             = Gridap.Geometry.num_cells(serial_model)
    nparts             = length(ranks)

    # Build the underlying param model so that Geometry.restrict can produce
    # local UnstructuredGridTopology + FaceLabeling for each rank.
    param_model = Gridap.Geometry.UnstructuredDiscreteModel(
        g.param_grid,
        serial_model.grid_topology,
        serial_model.face_labeling,
    )

    # Simple linear partition: cell_to_part[i] = owning rank for global cell i.
    cell_to_part = Int32[div((i - 1) * nparts, ncells) + 1 for i in 1:ncells]

    # Cell-adjacency graph (vertex connectivity) for one-layer ghost detection.
    cell_graph = GridapDistributed.compute_cell_graph(param_model)

    # For each rank: collect owned cells plus their face-adjacent ghost cells.
    lcell_to_cell, lcell_to_part = map(ranks) do part
        cell_to_mask = fill(false, ncells)
        ptrs = cell_graph.colptr
        vals = cell_graph.rowval
        for icell in 1:ncells
            if cell_to_part[icell] == part
                cell_to_mask[icell] = true
                for p in ptrs[icell]:(ptrs[icell + 1] - 1)
                    cell_to_mask[vals[p]] = true
                end
            end
        end
        loc_to_glob = findall(cell_to_mask)
        loc_to_part = collect(Int32, view(cell_to_part, loc_to_glob))
        loc_to_glob, loc_to_part
    end |> tuple_of_arrays

    # Build PRange describing global cell id layout.
    partition = map(ranks, lcell_to_cell, lcell_to_part) do part, l2g, l2p
        LocalIndices(ncells, part, l2g, l2p)
    end
    gids = PRange(partition)

    # Build per-rank AtlasDiscreteModel (owned cells + ghost cells).
    models = map(lcell_to_cell) do local_to_global
        # Restrict param model → local GridPortion + UnstructuredGridTopology + FaceLabeling.
        local_param_model = Gridap.Geometry.restrict(param_model, local_to_global)
        local_param_grid  = Gridap.Geometry.get_grid(local_param_model)
        local_topology    = Gridap.Geometry.get_grid_topology(local_param_model)
        local_labeling    = Gridap.Geometry.get_face_labeling(local_param_model)

        # Gather atlas data for the local cell subset.
        local_chart_coords = lazy_map(Reindex(g.cell_chart_coords), local_to_global)
        local_ambient_maps = lazy_map(Reindex(g.cell_ambient_maps), local_to_global)
        local_metric       = lazy_map(Reindex(g.cell_metric), local_to_global)

        local_atlas_grid = AtlasGrid(
            local_param_grid,
            local_chart_coords,
            local_ambient_maps,
            local_metric,
            Gridap.Geometry.OrientationStyle(g),
            manifold_style,
        )

        AtlasDiscreteModel(local_atlas_grid, local_topology, local_labeling)
    end
    return GenericDistributedDiscreteModel(models, gids)
end

function get_distributed_refined_models(ranks,
                                        coarse_mesh,
                                        n_ref_lvls,
                                        manifold_style,
                                        coarse_model=false)
  models = Vector{GenericDistributedDiscreteModel}(undef,n_ref_lvls)
  for (i,n) in enumerate(n_ref_lvls:-1:1)
    model = AtlasDiscreteModel(ranks, coarse_mesh, n; manifold_style=manifold_style)
    models[i] = model
  end
  if coarse_model
    push!(models,AtlasDiscreteModel(ranks, coarse_mesh, 0; manifold_style=manifold_style))
  end
  models
end 

function get_distributed_cubed_sphere_refined_models(ranks,
                                                     n_ref_lvls::Int, 
                                                     radius::Real, 
                                                     manifold_style, 
                                                     coarse_model=false)
  coarse_mesh = CubedSphereMesh(radius)
  get_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, manifold_style, coarse_model)
end


function get_distributed_intrinsic_cubed_sphere_refined_models(ranks,
                                                               n_ref_lvls::Int,
                                                               radius::Real,
                                                               coarse_model=false)
  get_distributed_cubed_sphere_refined_models(ranks, n_ref_lvls, radius, IntrinsicManifold(), coarse_model)
end

function get_distributed_extrinsic_cubed_sphere_refined_models(ranks,
                                                               n_ref_lvls::Int,
                                                               radius::Real,
                                                               coarse_model=false)
  get_distributed_cubed_sphere_refined_models(ranks, n_ref_lvls, radius, ExtrinsicManifold(), coarse_model)
end

function LatLonMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,
              <:AtlasDiscreteModel{Dc,Dp,G,A,
                <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},
                C,O,M}}}}
) where {Dc,Dp,G,A,C,O,M}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    LatLonMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:BFTATDM{Dc,Dp}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:BFTATDM{Dc,Dp}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function InvMetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:BFTATDM{Dc,Dp}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    InvMetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:BFTATDM{Dc,Dp}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end



function Δs(f::Function,
            trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:BFTATDM{Dc,Dp}}};
            use_automatic_differentiation=false) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    Δs(f, t; use_automatic_differentiation)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function ∇s(f::Function,
            trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:BFTATDM{Dc,Dp}}};
            use_automatic_differentiation=false) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    ∇s(f, t; use_automatic_differentiation)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

