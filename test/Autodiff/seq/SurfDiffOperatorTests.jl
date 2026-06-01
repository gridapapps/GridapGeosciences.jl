"""
In this module, test that the sgrad, sdiv, slap produce the same result as the
ambient_sgrad, ambient_sdiv, and ambient_slap
"""

module SurfDiffOperatorTests

using GridapGeosciences
using GridapGeosciences.Geometry
using Gridap
using Test

function test_operators(α,m)
  x = m(α)

  sgrad_ambient = ambient_sgrad(ambient_fX)(m)(x)
  sgrad_panel = sgrad(panel_fX)(m)(α)
  dif = sgrad_ambient .- sgrad_panel
  @test norm(dif) < 1e-12

  sdiv_ambient = ambient_surfdiv(ambient_vecX)(m)(x)
  sdiv_panel = surfdiv(contra_v(panel_uX))(m)(α)
  dif = sdiv_ambient - sdiv_panel
  @test dif < 1e-12

  slap_ambient = ambient_surflap(ambient_fX)(m)(x)
  slap_panel = surflap(panel_fX)(m)(α)
  dif = slap_ambient .- slap_panel
  @test dif < 1e-12

end


function ambient_fX(xyz)
  x,y,z = xyz
  x^2 + y^2*z
end


function ambient_vecX(xyz)
  x,y,z = xyz
  VectorValue(y^2,-x^2,0.0)
end


panel_fX = panel_to_cartesian(ambient_fX)
panel_uX = panel_to_cartesian(ambient_vecX)


panel_id = 1
radius = 1.0
thickness = 0.5

################################################################################
########## 2D
################################################################################
αβ = CUBE_HALF_EDGE*Point(-1.0,-1.0)
m = CubedSphereForwardMap(panel_id,radius)
test_operators(αβ,m)

################################################################################
########## 3D
################################################################################
γαβ = Point(0.5,-CUBE_HALF_EDGE,-CUBE_HALF_EDGE)
m = CubedSphereForwardMap(panel_id,radius,thickness)
test_operators(γαβ,m)


end # module
