### Technically do not need radius for inverse map, but want to be able to generate
### forward map from inverse map
struct InverseMap2D{T<:Real} <: Field
  panel::Int64
  radius::T
end

struct InverseMap2DGenerator{T<:Real} <: Map
  inverse_maps::Vector{InverseMap2D}
  InverseMap2DGenerator(radius::T) where {T<:Real}  =
     new{T}([InverseMap2D(p,radius) for p in 1:6])
end

Gridap.Arrays.evaluate!(cache,f::InverseMap2DGenerator,p::Integer) = f.inverse_maps[p]

struct InverseMap3D{T<:Real} <: Field
  panel::Int64
  radius::T
  thickness::T
end

struct InverseMap3DGenerator{T<:Real} <: Map
  radius::T
  thickness::T
  inverse_maps::Vector{InverseMap3D{T}}
  InverseMap3DGenerator(radius::T, thickness::T) where {T<:Real} =
     new{T}(radius, thickness, [InverseMap3D(p, radius, thickness) for p in 1:6])
end

Gridap.Arrays.evaluate!(cache,f::InverseMap3DGenerator,p::Integer) = f.inverse_maps[p]


## Given a forward_map, generate the corresponding inverse map
function CubedSphereInverseMap(m::ForwardMap2D)
  CubedSphereInverseMap(m.panel,m.radius)
end

function CubedSphereInverseMap(m::ForwardMap3D)
  CubedSphereInverseMap(m.panel,m.radius,m.thickness)
end

function CubedSphereInverseMap(p::Integer,radius)
  return InverseMap2D(p, radius)
end

function CubedSphereInverseMap(p::Integer,radius,thickness)
  InverseMap3D(p, radius,thickness)
end

function Gridap.Arrays.return_cache(f::InverseMap2D,cellx::AbstractArray{<:VectorValue{3}})
  x = first(cellx)
  T = typeof(_evaluate_inverse_map_2d(Val(f.panel),x))
  y = similar(cellx,T)
  return y
end


function Gridap.Arrays.evaluate!(cache,f::InverseMap2D,cellx::AbstractArray{<:VectorValue{3}} )
  y = cache
  cache .= _evaluate_inverse_map_2d.(Val(f.panel),cellx)
  return y
end

function Gridap.Arrays.evaluate!(cache,f::InverseMap2D,x::VectorValue{3})
  return _evaluate_inverse_map_2d(Val(f.panel),x)

end

### gradient is just the coefficient matrix
# Gridap convention dictates we return the transpose (https://github.com/gridap/Gridap.jl/issues/822)
function Gridap.Arrays.return_cache(cache,f::FieldGradient{1,<:InverseMap2D},
  cellx::AbstractArray{<:VectorValue{3}})

  x = first(cellx)
  T = typeof(transpose(_evaluate_inverse_jacobian_2d(f.object.panel,x)))

  y = similar(cellx,T)
  c = CachedArray(y)
  return c
end

function Gridap.Arrays.evaluate!(c,f::FieldGradient{1,<:InverseMap2D},
  cellx::AbstractArray{<:VectorValue{3}})
  setsize!(cache,size(cellx))
  cache.array .= transpose.(_evaluate_inverse_jacobian_2d.(f.object.panel,cellx))
  return cache.array
end


function Gridap.Arrays.evaluate!(cache,f::FieldGradient{1,<:InverseMap2D},x::VectorValue{3})
  return transpose(_evaluate_inverse_jacobian_2d(f.object.panel,x))
end


################################################################################
########## 2D inverse map ########
################################################################################
function _evaluate_inverse_map_2d(p::Val{1},XYZ)
  X,Y,Z = XYZ
  α = atan(Y,X)
  β = atan(Z,X)
  return  Point(α,β)
end

function _evaluate_inverse_map_2d(p::Val{2},XYZ)
  X,Y,Z = XYZ
  α = atan(Y,Z)
  β = -atan(X,Z)
  return  Point(α,β)
end

function _evaluate_inverse_map_2d(p::Val{3},XYZ)
  X,Y,Z = XYZ
  α = -atan(X,Y)
  β = atan(Z,Y)
  return  Point(α,β)
end

function _evaluate_inverse_map_2d(p::Val{4},XYZ)
  X,Y,Z = XYZ
  α = atan(Z,-X)
  β = -atan(-Y,-X)
  return  Point(α,β)
end

function _evaluate_inverse_map_2d(p::Val{5},XYZ)
  X,Y,Z = XYZ
  α = -atan(X,-Z)
  β = atan(Y,-Z)
  return  Point(α,β)
end

function _evaluate_inverse_map_2d(p::Val{6},XYZ)
  X,Y,Z = XYZ
  α = atan(Z,-Y)
  β = -atan(X,-Y)
  return  Point(α,β)
end

inverse_map_2d(p::Integer) = xyz -> _evaluate_inverse_map_2d(Val(p),xyz)
_evaluate_inverse_jacobian_2d(p::Integer,xyz) = transpose(gradient(inverse_map_2d(p))(xyz))


# ################################################################################
# ########## 2D inverse map jacobian ########
# ### Use analytic expressions rather than auto-diff to ensure atan2 function
# ################################################################################
# function _evaluate_inverse_jacobian_2d(p::Val{1},XYZ)
#   X,Y,Z = XYZ
#   dadX = - Y/(X^2 + Y^2)
#   dadY = X/(X^2 + Y^2)
#   dadZ = 0.0
#   dbdX = -Z/(X^2 + Z^2)
#   dbdY = 0.0
#   dbdZ = X/(X^2 + Z^2)
#   return TensorValue{2,3}(dadX,dbdX, dadY,dbdY, dadZ,dbdZ)
# end

# function _evaluate_inverse_jacobian_2d(p::Val{2},XYZ)
#   X,Y,Z = XYZ
#   dadX = 0.0
#   dadY = Z/(Y^2 + Z^2)
#   dadZ = -Y/(Y^2 + Z^2)
#   dbdX = -Z/(X^2 + Z^2)
#   dbdY = 0.0
#   dbdZ = X/(X^2 + Z^2)
#   return TensorValue{2,3}(dadX,dbdX, dadY,dbdY, dadZ,dbdZ)
# end

# function _evaluate_inverse_jacobian_2d(p::Val{3},XYZ)
#   X,Y,Z = XYZ
#   dadX = -Y/(X^2 + Y^2)
#   dadY = X/(X^2 + Y^2)
#   dadZ = 0.0
#   dbdX = 0.0
#   dbdY = -Z/(Y^2 + Z^2)
#   dbdZ = Y/(Y^2 + Z^2)
#   return TensorValue{2,3}(dadX,dbdX, dadY,dbdY, dadZ,dbdZ)
# end

# function _evaluate_inverse_jacobian_2d(p::Val{4},XYZ)
#   X,Y,Z = XYZ
#   dadX = Z/(Z^2 + X^2)
#   dadY = 0.0
#   dadZ = -X/(Z^2 + X^2)
#   dbdX = Y/(X^2 + Y^2)
#   dbdY = -X/(X^2 + Y^2)
#   dbdZ = 0.0
#   return TensorValue{2,3}(dadX,dbdX, dadY,dbdY, dadZ,dbdZ)
# end

# function _evaluate_inverse_jacobian_2d(p::Val{5},XYZ)
#   X,Y,Z = XYZ
#   dadX = Z/(X^2 + Z^2)
#   dadY = 0.0
#   dadZ = -X/(X^2 + Z^2)
#   dbdX = 0.0
#   dbdY = -Z/(Y^2 + Z^2)
#   dbdZ = Y/(Y^2 + Z^2)
#   return TensorValue{2,3}(dadX,dbdX, dadY,dbdY, dadZ,dbdZ)
# end

# function _evaluate_inverse_jacobian_2d(p::Val{6},XYZ)
#   X,Y,Z = XYZ
#   dadX = 0.0
#   dadY = Z/(Y^2 + Z^2)
#   dadZ = -Y/(Y^2 + Z^2)
#   dbdX = Y/(X^2 + Y^2)
#   dbdY = -X/(X^2 + Y^2)
#   dbdZ = 0.0
#   return TensorValue{2,3}(dadX,dbdX, dadY,dbdY, dadZ,dbdZ)
# end

# _evaluate_inverse_jacobian_2d(p::Integer,xyz) =  _evaluate_inverse_jacobian_2d(Val(p),xyz)


################################################################################
########## 3D ########
################################################################################
function _evaluate_inverse_map_3d(p::Val{P},radius::T,thickness::T,XYZ_shell) where {P,T<:Real}
  ## first compute γ using the radius in comparison to the radius_top
  rtop = radius + thickness
  x,y,z = XYZ_shell
  r = sqrt(x^2 + y^2 + z^2)
  _γ = (rtop - r)/thickness
  γ = 1.0 - _γ

  ## reverse to extrusion to the surface
  ## Note, normal_vec(XYZ_shell) = normal_vec(XYZ_surf)
  XYZ_surf = XYZ_shell - thickness*_γ*normal_vec(XYZ_shell)
  α,β =  _evaluate_inverse_map_2d(p,XYZ_surf)

  return Point(γ,α,β)
end


inverse_map_3d(p::Integer,radius::T,thickness::T) where T<:Real = xyz_shell -> _evaluate_inverse_map_3d(Val(p),radius,thickness,xyz_shell)
_evaluate_inverse_jacobian_3d(p::Integer,radius::T,thickness::T,xyz_shell) where T<:Real = transpose(gradient(inverse_map_3d(p,radius,thickness))(xyz_shell))

function Gridap.Arrays.return_cache(f::InverseMap3D,cellx::AbstractArray{<:VectorValue{3}})
  return similar(cellx,VectorValue{3,Float64})
end

function Gridap.Arrays.evaluate!(cache,
                                 f::InverseMap3D,
                                 cellx::AbstractArray{<:VectorValue{3}} )
  cache .= _evaluate_inverse_map_3d.(Val(f.panel),f.radius,f.thickness,cellx)
end


function Gridap.Arrays.evaluate!(cache,f::InverseMap3D,x::VectorValue{3})
  return _evaluate_inverse_map_3d(Val(f.panel),f.radius,f.thickness,x)
end


function Gridap.Arrays.return_cache(cache,f::FieldGradient{1,<:InverseMap3D},
  cellx::AbstractArray{<:VectorValue{3}})
  y = similar(cellx,TensorValue{3,3,Float64})
  c = CachedArray(y)
  return c
end

function Gridap.Arrays.evaluate!(cache,
                                 f::FieldGradient{1,<:InverseMap3D},
                                 cellx::AbstractArray{<:VectorValue{3}})
  setsize!(cache,size(cellx))
  radius = f.object.radius
  thickness = f.object.thickness
  cache.array .= transpose.(_evaluate_inverse_jacobian_3d.(f.object.panel,radius,thickness,cellx))
  return cache.array
end

function Gridap.Arrays.return_cache(cache,f::FieldGradient{1,<:InverseMap3D},x::VectorValue{3})
  zero(TensorValue{3,3,Float64})
end

function Gridap.Arrays.evaluate!(cache,f::FieldGradient{1,<:InverseMap3D},x::VectorValue{3})
  radius = f.object.radius
  thickness = f.object.thickness
  return transpose(_evaluate_inverse_jacobian_3d(f.object.panel,radius,thickness,x))
end
