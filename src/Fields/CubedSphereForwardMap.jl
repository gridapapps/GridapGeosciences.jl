struct ForwardMap2D{T<:Real} <: Field
  panel::Int64
  radius::T
end

struct ForwardMap2DGenerator{T<:Real} <: Map
  radius::T
  forward_maps::Vector{ForwardMap2D{T}}
  ForwardMap2DGenerator(radius::T) where {T<:Real}  =
     new{T}(radius, [ForwardMap2D(p, radius) for p in 1:6])
end

Gridap.Arrays.evaluate!(cache,f::ForwardMap2DGenerator,p::Integer) = f.forward_maps[p]

struct ForwardMap3D{T<:Real} <: Field
  panel::Int64
  radius::T
  thickness::T
end

struct ForwardMap3DGenerator{T<:Real} <: Map
  radius::T
  thickness::T
  forward_maps::Vector{ForwardMap3D{T}}
  ForwardMap3DGenerator(radius::T, thickness::T) where {T<:Real} =
     new{T}(radius, thickness, [ForwardMap3D(p, radius, thickness) for p in 1:6])
end

Gridap.Arrays.evaluate!(cache,f::ForwardMap3DGenerator,p::Integer) = f.forward_maps[p]

function CubedSphereForwardMap(p::Integer,radius)
  return ForwardMap2D(p, radius)
end

function CubedSphereForwardMap(p::Integer,radius,thickness)
  return ForwardMap3D(p, radius, thickness)
end

const ForwardMap2Dor3D = Union{ForwardMap2D, ForwardMap3D}
const ForwardMap2Dor3DGenerator = Union{ForwardMap2DGenerator, ForwardMap3DGenerator}

J(m,x) = _evaluate_forward_jacobian_2d(m.panel,m.radius,x)
J(m::ForwardMap3D,x) = _evaluate_forward_jacobian_3d(m.panel,m.radius,m.thickness,x)

################################################################################
########## 2D ########
################################################################################

function rho(α,β)
  return sqrt(1.0 + tan(α)*tan(α) + tan(β)*tan(β) )
end

## IMPORTANT NOTE: We cannot annotate αβ as a VectorValue{3} because Forward AD
## algorithms will not be able to differentiate through the function if we do.
function _evaluate_forward_map_2d(p::Val{1},radius,αβ)
    α,β = αβ
    ρ = rho(α,β)
    X = 1.0/ρ
    Y = 1.0/ρ * tan(α)
    Z = 1.0/ρ * tan(β)
    return radius*Point(X,Y,Z)
end

function _evaluate_forward_map_2d(p::Val{2},radius,αβ)
    α,β = αβ
    ρ = rho(α,β)
    X = -1.0/ρ * tan(β)
    Y = 1.0/ρ * tan(α)
    Z = 1.0/ρ
    return radius*Point(X,Y,Z)
end

function _evaluate_forward_map_2d(p::Val{3},radius,αβ)
    α,β = αβ
    ρ = rho(α,β)
    X = -1.0/ρ * tan(α)
    Y = 1.0/ρ
    Z = 1.0/ρ * tan(β)
    return radius*Point(X,Y,Z)
end

function _evaluate_forward_map_2d(p::Val{4},radius,αβ)
    α,β = αβ
    ρ = rho(α,β)
    X = -1.0/ρ
    Y = 1.0/ρ * tan(β)
    Z = 1.0/ρ * tan(α)
    return radius*Point(X,Y,Z)
end

function _evaluate_forward_map_2d(p::Val{5},radius,αβ)
    α,β = αβ
    ρ = rho(α,β)
    X = -1.0/ρ * tan(α)
    Y = 1.0/ρ * tan(β)
    Z = -1.0/ρ
    return radius*Point(X,Y,Z)
end

function _evaluate_forward_map_2d(p::Val{6},radius,αβ)
    α,β = αβ
    ρ = rho(α,β)
    X = -1.0/ρ * tan(β)
    Y = -1.0/ρ
    Z = 1.0/ρ * tan(α)
    return radius*Point(X,Y,Z)
end

_evaluate_forward_jacobian_2d(p::Integer,radius::Real,αβ) = transpose(gradient(forward_map_2d(p,radius))(αβ))
forward_map_2d(p::Integer,radius::Real) = αβ -> _evaluate_forward_map_2d(Val(p),radius,αβ)


function Gridap.Arrays.return_cache(f::ForwardMap2D,cellx::AbstractArray{<:VectorValue{2}})
  return similar(cellx,VectorValue{3,Float64})
end

function Gridap.Arrays.evaluate!(cache,f::ForwardMap2D,cellx::AbstractArray{<:VectorValue{2}})
  cache .= _evaluate_forward_map_2d.(Val(f.panel),f.radius,cellx)
  return cache
end

function Gridap.Arrays.evaluate!(cache,f::ForwardMap2D,x::VectorValue{2})
  return _evaluate_forward_map_2d(Val(f.panel),f.radius,x)
end

"""
The jacobian is 3 x 2
  J = [dXda dXdb
       dYda dYdb
       dZda dZdb  ]
As a TensorValue data = (dXda,dYda,dZda,   dXdb, dYdb, dZdb)

Gridap convention dictates we return the transpose (https://github.com/gridap/Gridap.jl/issues/822)

The transpose is 2 x 3
  JT = [dXda dYda dZda
        dXdb dYdb dZdb  ]
As a TensorValue data = (dXda,dXdb,  dYda,dYdb,  dZda,dZdb)
"""
function Gridap.Arrays.return_cache(f::FieldGradient{1,<:ForwardMap2D},
                                    cellx::AbstractArray{<:VectorValue{2}})
  y = similar(cellx,TensorValue{2,3,Float64})
  c = CachedArray(y)
  return c
end

function Gridap.Arrays.evaluate!(cache,
                                 f::FieldGradient{1,<:ForwardMap2D},
                                 cellx::AbstractArray{<:VectorValue{2}})
  setsize!(cache,size(cellx))
  radius = f.object.radius
  cache.array .= transpose.(_evaluate_forward_jacobian_2d.(f.object.panel,radius,cellx))
  return cache.array
end

function Gridap.Arrays.evaluate!(cache,f::FieldGradient{1,<:ForwardMap2D},x::VectorValue{2})
  radius = f.object.radius
  return transpose(_evaluate_forward_jacobian_2d(f.object.panel,radius,x))
end

################################################################################
########## 3D ########
################################################################################
# unit normal
normal_vec(XYZ) = 1/sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2] + XYZ[3]*XYZ[3])*VectorValue(XYZ[1],XYZ[2],XYZ[3])


## IMPORTANT NOTE: We cannot annotate γαβ as a VectorValue{3} because Forward AD
## algorithms will not be able to differentiate through the function if we do.
function _evaluate_forward_map_3d(p::Val{P},radius::T,thickness::T,γαβ) where {P,T<:Real}
  #### recall the first coordinate in P6est is the extrusion!
  γ,α,β = γαβ

  #### compute XYZ point on surface of inner sphere using 2D forward_map
  αβ = Point(α,β)
  XYZ_surf = _evaluate_forward_map_2d(p,radius,αβ)

  #### extrude surface point in radial direction
  return XYZ_surf + thickness*γ*normal_vec(XYZ_surf)
end

forward_map_3d(p::Integer,radius::T,thickness::T) where T<:Real = γαβ -> _evaluate_forward_map_3d(Val(p),radius,thickness,γαβ)
_evaluate_forward_jacobian_3d(p::Integer,radius::T,thickness::T,γαβ) where T<:Real = transpose(gradient(forward_map_3d(p,radius,thickness))(γαβ))

function Gridap.Arrays.return_cache(f::ForwardMap3D,cellx::AbstractArray{<:VectorValue{3}})
  return similar(cellx,VectorValue{3,Float64})
end

function Gridap.Arrays.evaluate!(cache,
                                 f::ForwardMap3D,
                                 cellx::AbstractArray{<:VectorValue{3}} )
  cache .= _evaluate_forward_map_3d.(Val(f.panel),f.radius,f.thickness,cellx)
end


function Gridap.Arrays.evaluate!(cache,f::ForwardMap3D,x::VectorValue{3})
  return _evaluate_forward_map_3d(Val(f.panel),f.radius,f.thickness,x)
end

"""
The jacobian is 3 x 2
  J = [dXdg dXda dXdb
       dYdg dYda dYdb
       dZdg dZda dZdb  ]
As a TensorValue data = (dXdg,dYdg,dZdg,  dXda,dYda,dZda,   dXdb, dYdb, dZdb)

Gridap convention dictates we return the transpose (https://github.com/gridap/Gridap.jl/issues/822)

The transpose is 2 x 3
  JT = [dXdg dYdg dZdg
        dXda dYda dZda
        dXdb dYdb dZdb  ]
As a TensorValue data = (dXdg,dXda,dXdb,  dYdg,dYda,dYdb,  dZdg,dZda,dZdb)
"""
function Gridap.Arrays.return_cache(cache,f::FieldGradient{1,<:ForwardMap3D},
  cellx::AbstractArray{<:VectorValue{3}})
  y = similar(cellx,TensorValue{3,3,Float64})
  c = CachedArray(y)
  return c
end

function Gridap.Arrays.evaluate!(c,
                                 f::FieldGradient{1,<:ForwardMap3D},
                                 cellx::AbstractArray{<:VectorValue{3}})
  cache, = c
  setsize!(cache,size(cellx))
  radius = f.object.radius
  thickness = f.object.thickness
  cache.array .= transpose.(_evaluate_forward_jacobian_3d.(f.object.panel,radius,thickness,cellx))
  return cache.array
end

function Gridap.Arrays.return_cache(cache,f::FieldGradient{1,<:ForwardMap3D},x::VectorValue{3})
  zero(TensorValue{3,3,Float64})
end

function Gridap.Arrays.evaluate!(cache,f::FieldGradient{1,<:ForwardMap3D},x::VectorValue{3})
  radius = f.object.radius
  thickness = f.object.thickness
  return transpose(_evaluate_forward_jacobian_3d(f.object.panel,radius,thickness,x))
end
