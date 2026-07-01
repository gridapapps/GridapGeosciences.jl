# CoarseMeshes.jl
#
# Library of canonical coarse meshes for AtlasGrid / AtlasDiscreteModel.
# Each shape is represented by a concrete subtype of CoarseMesh that
# carries the geometric parameters (radius, height, …).  Calling
# get_coarse_mesh(shape) returns a CoarseMeshInfo bundling the coarse
# DiscreteModel (with face labels), per-cell local-frame corner coordinates,
# and default ambient maps.
#
# Subtypes are defined in separate files (one per subtype), all included at
# the bottom of this file.
#
# ============================================================
# CoarseMeshInfo
# ============================================================

"""
    CoarseMeshInfo{Dc, Dm, A, M, G}

Bundles a coarse `DiscreteModel{Dc,Dc}` (with face labels) with per-cell
local-frame corner coordinates, per-chart ambient maps, and per-chart metric fields.
Returned by `get_coarse_mesh`; consumed by the `AtlasGrid` and `AtlasDiscreteModel`
convenience constructors.

- `model`         — coarse `DiscreteModel{Dc,Dc}` carrying topology and
                    `FaceLabeling` (node coordinates are junk — only connectivity
                    matters).  For meshes with physical boundaries (e.g. cylinder),
                    boundary edges/nodes are tagged by `get_coarse_mesh`.
- `cell_chart_coords`  — one entry per coarse cell; `cell_chart_coords[k]` is a vector of
                    `Point{Dc}` giving the corners of chart k in its local frame.
- `ambient_maps`  — one `Field` per chart: `Point{Dc} → Point{Da}`.
- `metric_fields` — one `Field` per chart: `Point{Dc} → SymTensorValue{Dc}`,
                    the pullback metric `g`.  For built-in shapes these are
                    concrete analytic types (e.g. `CubedSphereMetric`);
                    user-defined shapes may use `_pullback_metrics(ambient_maps)`
                    as a generic fallback.  The explicit inverse is obtained via
                    `inverse_metric_field(metric_field)`.
"""
struct CoarseMeshInfo{Dc,
                      Dm <: Gridap.Geometry.DiscreteModel{Dc,Dc},
                      A  <: AbstractVector,
                      M,
                      G}
  model              :: Dm
  cell_chart_coords  :: A
  ambient_maps       :: M
  metric_fields      :: G

  function CoarseMeshInfo(
      model             :: Gridap.Geometry.DiscreteModel{Dc,Dc},
      cell_chart_coords :: A,
      ambient_maps      :: M,
      metric_fields     :: G,
  ) where {Dc, A <: AbstractVector, M, G}
    Dm = typeof(model)
    new{Dc,Dm,A,M,G}(model, cell_chart_coords, ambient_maps, metric_fields)
  end
end

# ============================================================
# CoarseMesh
# ============================================================

"""
    CoarseMesh

Supertype for all canonical coarse-mesh descriptors.
Subtypes carry the geometric parameters (radius, height, …) and are passed
to `get_coarse_mesh` to obtain a `CoarseMeshInfo`.
"""
abstract type CoarseMesh end
