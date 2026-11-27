using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapGeosciences
using BenchmarkTools
using DrWatson
using DataFrames

dir = datadir()*"/assembly"
!isdir(dir) && mkpath(dir)


RADIUS = 1.0

p_fe = 1
degree = 6*(p_fe+1)


for NUM_REF in collect(1:6)

  ### intrinisic
  atlas_model = AtlasDiscreteModel(CubedSphereMesh(RADIUS), NUM_REF; manifold_style=IntrinsicManifold())

  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)

  V = TestFESpace(atlas_model, ReferenceFE(lagrangian,Float64,p_fe); conformity=:H1)
  U = TrialFESpace(V)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)

  poisson_int(u,v) =  ∫( ( gradient(v)⋅ (inv_metric_cf⋅ gradient(u) ) )*meas_cf )dΩ
  A_int = assemble_matrix(poisson_int,U,V )


  make_poisson_int(dΩ, ginv, meas) = (u,v) -> ∫( (∇(v) ⋅ (ginv ⋅ ∇(u))) * meas )dΩ


  #### extrinisic

  ext_model = AtlasDiscreteModel(CubedSphereMesh(RADIUS), NUM_REF; manifold_style=ExtrinsicManifold())

  Ω_ext = Triangulation(ext_model)
  dΩ_ext = Measure(Ω_ext,degree)

  V_ext = TestFESpace(ext_model, ReferenceFE(lagrangian,Float64,p_fe); conformity=:H1)
  U_ext = TrialFESpace(V_ext)

  poisson_ext(u,v) =  ∫(  gradient(v)⋅ gradient(u)  )dΩ_ext
  A_ext =  assemble_matrix(poisson_ext,U_ext,V_ext )

  make_poisson_ext(dΩ)             = (u,v) -> ∫( ∇(v) ⋅ ∇(u) )dΩ


  ###

  a_int = make_poisson_int(dΩ, inv_metric_cf, meas_cf)
  a_ext = make_poisson_ext(dΩ_ext)

  b_int = @benchmark assemble_matrix($a_int, $U, $V)
  b_ext = @benchmark assemble_matrix($a_ext, $U_ext, $V_ext)

  # println("Time === Int $(time(m_int)) vs. Ext $(time(m_ext)) (nanoseconds)")
  # println("Memomry === Int $(memory(m_int)) vs. Ext $(memory(m_ext)) (bytes)")
  # println("Allocs === Int $(allocs(m_int)) vs. Ext $(allocs(m_ext)) ()")


  ## convergence output for DrWatson
  output = @strdict b_int b_ext p_fe NUM_REF
  safesave(datadir(dir, ("assembly_nref$(NUM_REF)_p$(p_fe).jld2")), output)

end



### load and plot
!isdir(plotsdir()) && mkpath(plotsdir())
using Plots
markers= [:circle :rect  :diamond ]
markersize = [7 8 6]
linestyle = [:solid :dash]
_colors = palette(:tab10)
colors = [_colors[1] _colors[2] _colors[3]]

dir = datadir()*"/assembly"
df = collect_results(dir)


for p in [1,2]
  b_ext = df[df.p_fe .== p,:b_ext]
  b_int = df[df.p_fe .== p, :b_int]


  m_int = median.(b_int)
  m_ext = median.(b_ext)

  ts_int = time.(m_int)
  ts_ext = time.(m_ext)
  ratio_ts = ts_int./ts_ext

  mem_int = memory.(m_int)
  mem_ext = memory.(m_ext)
  ratio_mem = mem_int./mem_ext

  alloc_int = allocs.(m_int)
  alloc_ext = allocs.(m_ext)

  ratio_alloc = alloc_int./alloc_ext



  xplot = [1,2,3,4,5,6]
  plot()
  plot!(xplot, [ratio_ts, ratio_mem, ratio_mem],
        lw=3,
        markers=markers, ms=markersize,color=colors,
      label=["Time" "Mem" "Alloc"])
  plot!(shape=:auto,
      xlabel="Ref level",
      ylabel="Int / Ext ",
      title="p=$(p)",
      framestyle = :box)
  savefig(plotsdir("p$(p).pdf"))

end
