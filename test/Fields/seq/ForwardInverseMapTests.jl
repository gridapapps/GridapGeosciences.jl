"""
Test the evaluation of the CubedSphereMap and CubedSphereInvMap in 2D and 3D
"""

module ForwardInverseMapTests

using GridapGeosciences
using GridapGeosciences.Geometry
using Gridap
using Test

radius = 1.0
thickness = 0.25

################################################################################
########## Mapping of points
################################################################################

#### 2D test: four corners of the panel
pts_αβ = CUBE_HALF_EDGE.*[Point(-1.0,-1.0),Point(1.0,-1.0),Point(-1.0,1.0),Point(1.0,1.0)]

for panel in collect(1:NPANELS)
  fwd_maps = fill(CubedSphereMap(panel,radius),length(pts_αβ))
  inv_maps = fill(CubedSphereInvMap(panel,radius),length(pts_αβ))
  pts_x = lazy_map(evaluate,fwd_maps,pts_αβ)
  pts_αβ_inv = lazy_map(evaluate,inv_maps,pts_x)
  @test pts_αβ ≈ pts_αβ_inv
end

#### 3D test: a range of γ ∈ [0, 1.0] values
for γ in [0.0, 0.5, 1.0]
  pts_γαβ = map(x->Point(1.0,x[1],x[2]) ,pts_αβ)
  for panel in collect(1:NPANELS)
    fwd_maps = fill(CubedSphereWithThicknessMap(panel,radius,thickness),length(pts_γαβ))
    inv_maps = fill(CubedSphereWithThicknessInvMap(panel,radius,thickness),length(pts_γαβ))
    pts_x = lazy_map(evaluate,fwd_maps,pts_γαβ)
    pts_γαβ_inv = lazy_map(evaluate,inv_maps,pts_x)
    @test pts_γαβ ≈ pts_γαβ_inv
  end
end


################################################################################
########## Jacobian of inverse map
########## No analytic expression in 3D
################################################################################

#### 2D test: test the auto-diff of the inverse_map for panel 1 aganist analytic expression
panel = 1
αβ = Point(CUBE_HALF_EDGE,CUBE_HALF_EDGE)
m = CubedSphereMap(panel,radius)
minv = CubedSphereInvMap(panel,radius)
xyz = m(αβ)

Jt_minv = ∇(minv)(xyz)

X,Y,Z = xyz
dadX = - Y/(X^2 + Y^2)
dadY = X/(X^2 + Y^2)
dadZ = 0.0
dbdX = -Z/(X^2 + Z^2)
dbdY = 0.0
dbdZ = X/(X^2 + Z^2)
J_minv = TensorValue{2,3}(dadX,dbdX, dadY,dbdY, dadZ,dbdZ)
Jt_minv_analytic = transpose(J_minv)
@test Jt_minv_analytic ≈ Jt_minv


end # module
