const AtlasDistributedDiscreteModel{Dc,Dp,G,A,P,C,O,M} =
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:AtlasDiscreteModel{Dc,Dp,G,A,P,C,O,M}}}
const AdaptedAtlasDistributedDiscreteModel{Dc,Dp} =
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:Gridap.Adaptivity.AdaptedDiscreteModel{Dc,Dp,<:AtlasDiscreteModel{Dc,Dp}}}}
const IntrinsicAtlasDistributedDiscreteModel{Dc,Dp,G,A,P,C,O} =
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:IntrinsicAtlasDiscreteModel{Dc,Dp,G,A,P,C,O}}}
const ExtrinsicAtlasDistributedDiscreteModel{Dc,Dp,G,A,P,C,O} =
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:ExtrinsicAtlasDiscreteModel{Dc,Dp,G,A,P,C,O}}}
const AdaptedIntrinsicAtlasDistributedDiscreteModel{Dc,Dp} =
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:Gridap.Adaptivity.AdaptedDiscreteModel{Dc,Dp,<:IntrinsicAtlasDiscreteModel{Dc,Dp}}}}
const AdaptedExtrinsicAtlasDistributedDiscreteModel{Dc,Dp} =
     GenericDistributedDiscreteModel{Dc,Dp,<:AbstractVector{<:Gridap.Adaptivity.AdaptedDiscreteModel{Dc,Dp,<:ExtrinsicAtlasDiscreteModel{Dc,Dp}}}}


# Distribute a (small) serial AtlasDiscreteModel across MPI ranks.
# Assigns cells with a linear partition, then gives each rank owned cells
# plus their vertex-adjacent ghosts.  Only called on the coarse mesh (tiny),
# so the sequential work here is not a scalability concern.
function _distribute_serial_atlas_model(ranks, serial_model::AtlasDiscreteModel, manifold_style)
    g      = get_atlas_grid(serial_model)
    ncells = Gridap.Geometry.num_cells(serial_model)
    nparts = length(ranks)

    param_model = Gridap.Geometry.UnstructuredDiscreteModel(
        g.param_grid,
        serial_model.grid_topology,
        serial_model.face_labeling,
    )

    cell_to_part = Int32[div((i - 1) * nparts, ncells) + 1 for i in 1:ncells]
    cell_graph   = GridapDistributed.compute_cell_graph(param_model)

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

    partition_arr = map(ranks, lcell_to_cell, lcell_to_part) do part, l2g, l2p
        LocalIndices(ncells, part, l2g, l2p)
    end
    gids = PRange(partition_arr)

    models = map(lcell_to_cell) do local_to_global
        local_param_model = Gridap.Geometry.restrict(param_model, local_to_global)
        local_atlas_grid  = AtlasGrid(
            Gridap.Geometry.get_grid(local_param_model),
            lazy_map(Reindex(g.cell_chart_coords), local_to_global),
            lazy_map(Reindex(g.cell_ambient_maps),  local_to_global),
            lazy_map(Reindex(g.cell_metric),        local_to_global),
            Gridap.Geometry.OrientationStyle(g),
            manifold_style,
        )
        AtlasDiscreteModel(
            local_atlas_grid,
            Gridap.Geometry.get_grid_topology(local_param_model),
            Gridap.Geometry.get_face_labeling(local_param_model),
        )
    end
    GenericDistributedDiscreteModel(models, gids)
end

function AtlasDiscreteModel(ranks::AbstractArray{<:Integer},
                            mesh::CoarseMesh,
                            num_refinements::Int;
                            orientation_style=nothing,
                            manifold_style=ExtrinsicManifold())

    info   = get_coarse_mesh(mesh)
    nparts = length(ranks)
    Dc     = Gridap.Geometry.num_cell_dims(info.model)

    # Decide how many refinements to do sequentially before distributing.
    # We refine serially until ncells >= nparts so every rank can own at least
    # one cell.  Each uniform refinement multiplies the cell count by 2^Dc.
    ncells    = Gridap.Geometry.num_cells(info.model)
    n_seq_ref = 0
    while ncells < nparts && n_seq_ref < num_refinements
        ncells   *= 2^Dc
        n_seq_ref += 1
    end
    n_dist_ref = num_refinements - n_seq_ref

    # Build the sequential serial model in one shot (uses a single call to
    # Gridap's refine machinery internally, so no repeated allocation).
    serial_model = AtlasDiscreteModel(info, n_seq_ref; orientation_style, manifold_style)
    model        = _distribute_serial_atlas_model(ranks, serial_model, manifold_style)

    # Refine distributedly for the remaining levels: each step is fully parallel
    # with only a narrow MPI exchange for global cell-id assignment.
    for _ in 1:n_dist_ref
        refined = Gridap.Adaptivity.refine(model)
        # Unwrap AdaptedDiscreteModel → plain AtlasDiscreteModel so the
        # AtlasDistributedDiscreteModel type alias keeps matching on subsequent iterations.
        plain_models = map(local_views(refined)) do lm
            Gridap.Adaptivity.get_model(lm)
        end
        model = GenericDistributedDiscreteModel(plain_models, get_cell_gids(refined))
    end
    return model
end

"""
    _atlas_model_portion(model::AtlasDiscreteModel, cell_ids) -> AtlasDiscreteModel

Return a new `AtlasDiscreteModel` restricted to the subset of cells given by
`cell_ids` (a vector of local cell indices).  Topology and face labels are
restricted via `Gridap.Geometry.restrict`; atlas fields (`cell_chart_coords`,
`cell_ambient_maps`, `cell_metric`) are restricted with `Reindex`.
"""
function _atlas_model_portion(model::AtlasDiscreteModel, cell_ids)
  g = model.atlas_grid

  param_model = Gridap.Geometry.UnstructuredDiscreteModel(
    g.param_grid,
    model.grid_topology,
    model.face_labeling,
  )
  restricted_param    = Gridap.Geometry.restrict(param_model, cell_ids)
  restricted_grid     = Gridap.Geometry.get_grid(restricted_param)
  restricted_topology = Gridap.Geometry.get_grid_topology(restricted_param)
  restricted_labeling = Gridap.Geometry.get_face_labeling(restricted_param)

  restricted_atlas_grid = AtlasGrid(
    restricted_grid,
    lazy_map(Reindex(g.cell_chart_coords), cell_ids),
    lazy_map(Reindex(g.cell_ambient_maps), cell_ids),
    lazy_map(Reindex(g.cell_metric),       cell_ids),
    Gridap.Geometry.OrientationStyle(g),
    ManifoldStyle(g),
  )
  AtlasDiscreteModel(restricted_atlas_grid, restricted_topology, restricted_labeling)
end

# Common implementation for distributed atlas refinement.
# `fmodels_full[r]` = result of locally refining rank r's model (an AdaptedDiscreteModel).
# `local_views(cmodel)[r]` is used as the parent stored in the output AdaptedDiscreteModel,
# so the parent chain is preserved regardless of whether the input local models are plain
# AtlasDiscreteModels or AdaptedDiscreteModels of AtlasDiscreteModels.
function _refine_atlas_distributed(cmodel, fmodels_full, Dc)
  cgids   = partition(get_cell_gids(cmodel))
  cmodels = local_views(cmodel)

  Df = 0
  f_own_or_ghost_ids, f_own_ids = map(cgids, fmodels_full) do cgids, fmodel
    glue  = Gridap.Adaptivity.get_adaptivity_glue(fmodel)
    f2c   = glue.n2o_faces_map[Dc+1]
    ftopo = Gridap.Geometry.get_grid_topology(fmodel)
    c_l2o = local_to_own(cgids)

    f_cell_to_vertex = Gridap.Geometry.get_faces(ftopo, Dc, Df)
    f_vertex_to_cell = Gridap.Geometry.get_faces(ftopo, Df, Dc)
    c2v_cache        = array_cache(f_cell_to_vertex)
    v2c_cache        = array_cache(f_vertex_to_cell)

    f_own_mask          = fill(false, length(f2c))
    f_own_or_ghost_mask = fill(false, length(f2c))
    for (fcell, ccell) in enumerate(f2c)
      if !iszero(c_l2o[ccell])
        f_own_mask[fcell] = true
        for vertex in getindex!(c2v_cache, f_cell_to_vertex, fcell)
          for vcell in getindex!(v2c_cache, f_vertex_to_cell, vertex)
            f_own_or_ghost_mask[vcell] = true
          end
        end
      end
    end

    oog_ids = findall(f_own_or_ghost_mask)
    own_ids = findall(i -> f_own_mask[i], oog_ids)
    oog_ids, own_ids
  end |> tuple_of_arrays

  fmodels = map(fmodels_full, cmodels, f_own_or_ghost_ids) do fmodel, parent_model, oog_ids
    fine_atlas = Gridap.Adaptivity.get_model(fmodel)
    _glue      = Gridap.Adaptivity.get_adaptivity_glue(fmodel)

    restricted_atlas = _atlas_model_portion(fine_atlas, oog_ids)

    # Only populate the cell-level face map; AdaptivityGlue only accesses
    # n2o_faces_map[end] to build o2n_faces_map.
    n2o_faces_map        = Vector{Vector{Int}}(undef, Dc+1)
    n2o_faces_map[Dc+1]  = _glue.n2o_faces_map[Dc+1][oog_ids]
    n2o_cell_to_child_id = _glue.n2o_cell_to_child_id[oog_ids]
    new_glue = Gridap.Adaptivity.AdaptivityGlue(
      n2o_faces_map, n2o_cell_to_child_id, _glue.refinement_rules
    )

    Gridap.Adaptivity.AdaptedDiscreteModel(restricted_atlas, parent_model, new_glue)
  end

  fgids = GridapDistributed.refine_cell_gids(cmodel, fmodels, f_own_ids)
  GenericDistributedDiscreteModel(fmodels, fgids)
end

"""
    Gridap.Adaptivity.refine(cmodel::AtlasDistributedDiscreteModel) -> GenericDistributedDiscreteModel

Uniformly refine a distributed `AtlasDistributedDiscreteModel` once.

Each rank refines its local `AtlasDiscreteModel` independently — no global
communication is needed for the local refinement step.  A one-layer ghost
layer (vertex-adjacent) is then selected by the same filter used by
GridapDistributed for unstructured model refinement, and global cell gids are
updated via a narrow MPI exchange (`GridapDistributed.refine_cell_gids`).
"""
function Gridap.Adaptivity.refine(cmodel::AtlasDistributedDiscreteModel{Dc}) where Dc
  fmodels_full = map(local_views(cmodel)) do lm
    Gridap.Adaptivity.refine(lm)
  end
  _refine_atlas_distributed(cmodel, fmodels_full, Dc)
end

"""
    Gridap.Adaptivity.refine(cmodel::AdaptedAtlasDistributedDiscreteModel) -> GenericDistributedDiscreteModel

Uniformly refine a distributed model whose local views are already
`AdaptedDiscreteModel{<:AtlasDiscreteModel}` (i.e., the result of a previous
distributed refinement).  The inner `AtlasDiscreteModel` is extracted from each
local `AdaptedDiscreteModel` and refined; the input `AdaptedDiscreteModel` is
stored as the parent in the output, preserving the full refinement chain.
"""
function Gridap.Adaptivity.refine(cmodel::AdaptedAtlasDistributedDiscreteModel{Dc}) where Dc
  fmodels_full = map(local_views(cmodel)) do lm
    Gridap.Adaptivity.refine(Gridap.Adaptivity.get_model(lm))
  end
  _refine_atlas_distributed(cmodel, fmodels_full, Dc)
end

function generate_distributed_refined_models(ranks,
                                        coarse_mesh,
                                        n_ref_lvls,
                                        manifold_style,
                                        coarse_model=false)

  models = Vector{GenericDistributedDiscreteModel}(undef,n_ref_lvls)
  cmodel = AtlasDiscreteModel(ranks, coarse_mesh, 0; manifold_style=manifold_style)
  model = cmodel
  for n in n_ref_lvls:-1:1
    model = Gridap.Adaptivity.refine(model)
    models[n] = model
  end
  if coarse_model
    push!(models,cmodel)
  end
  models
end

function LatLonMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{
              Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,<:AtlasDiscreteModel{Dc,Dp,G,A,
                <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},C,O,M}},
              Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:Gridap.Geometry.BodyFittedTriangulation{Dc,Dp,
                <:AtlasDiscreteModel{Dc,Dp,G,A,
                  <:AbstractVector{<:Union{<:CubedSphereMap,<:CubedSphereWithThicknessMap}},C,O,M}}}}}}
) where {Dc,Dp,G,A,C,O,M}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    LatLonMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                   Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
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
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{
              Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
              Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function AmbientMapCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    AmbientMapCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                   Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
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
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{
              Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
              Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function InvMetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                   Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    InvMetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function InvMetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    InvMetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function InvMetricCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    InvMetricCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                   Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
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
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{
              Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}},
              Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:Gridap.Geometry.SkeletonTriangulation{Dc,Dp,
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
                <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:BFTATDM{Dc,Dp}}}}
) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)
  fields = map(ghosted_trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function MeasureCellField(
    trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Gridap.Geometry.TriangulationView{Dc,Dp,
              <:Gridap.Geometry.BoundaryTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}}
) where {Dc,Dp}
  fields = map(trian.trians) do t
    MeasureCellField(t)
  end
  GridapDistributed.DistributedCellField(fields, trian)
end

function Δs(f::Function,
            trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                           Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}};
            use_automatic_differentiation=false) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    Δs(f, t; use_automatic_differentiation)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

function ∇s(f::Function,
            trian::DistributedTriangulation{Dc,Dp,<:AbstractArray{<:Union{BFTATDM{Dc,Dp},
                                                                           Gridap.Adaptivity.AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}}}}};
            use_automatic_differentiation=false) where {Dc,Dp}
  ghosted_trian = add_ghost_cells(trian)

  fields = map(ghosted_trian.trians) do t
    ∇s(f, t; use_automatic_differentiation)
  end
  GridapDistributed.DistributedCellField(fields, ghosted_trian)
end

