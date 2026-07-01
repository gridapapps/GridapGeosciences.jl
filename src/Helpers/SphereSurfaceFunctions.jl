# unit normal
sphere_surface_normal_vec(XYZ) = 
  1.0/sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2] + XYZ[3]*XYZ[3])*VectorValue(XYZ[1],XYZ[2],XYZ[3])

# tangent component of arbitrary 3D vector vecX
sphere_tangent_vec_component(vecX::Function) = 
   XYZ -> vecX(XYZ) - (vecX(XYZ)⋅sphere_surface_normal_vec(XYZ))⋅sphere_surface_normal_vec(XYZ)