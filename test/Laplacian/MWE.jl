using GridapGeosciences
using Gridap


ℓ = 1
radius = 1.0
coarse_mesh = CubedSphereMesh(radius)
atlas_model = AtlasDiscreteModel(coarse_mesh,ℓ,manifold_style=IntrinsicManifold())

p_fe = 2 # Taylor hood pair requires p≥2


degree = 6*(p_fe+1)
Ω_atlas = Triangulation(atlas_model)
dΩ = Measure(Ω_atlas,degree)

meas_cf = MeasureCellField(Ω_atlas)

## FE spaces: Taylor hood pair
reffe_u  = ReferenceFE(lagrangian,VectorValue{2, Float64},p_fe)
reffe_p = ReferenceFE(lagrangian,Float64,p_fe-1)

V = TestFESpace(Ω_atlas, reffe_u; conformity=:H1)
U = TrialFESpace(V)

Q = TestFESpace(Ω_atlas, reffe_p; conformity=:L2)
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])



dx = get_trial_fe_basis(X)
dy = get_fe_basis(Y)

du = get_trial_fe_basis(U)
dp = get_trial_fe_basis(P)
dv = get_fe_basis(V)
dq = get_fe_basis(Q)



a1((u,p),(v,q)) = ∫( ( p*( ∇⋅(v*meas_cf)  ) )*meas_cf  )dΩ
a1((du,dp),(dv,dq)) ## FAIL!
a1(dx,dy) ## FAIL!!

a2(p,v) =∫( ( p*( (∇⋅v)*meas_cf  ) )*meas_cf  )dΩ
a2(dp,dv) ### Okay

a3(p,v) =∫( ( p*( v⋅(∇(meas_cf))  ) )*meas_cf  )dΩ
a3(dp,dv) ### FAIL!
