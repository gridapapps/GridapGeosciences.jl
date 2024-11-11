function Gridap.Algebra.symbolic_setup(solver::GMRESSolver, A::LinearMap)
   return GridapSolvers.LinearSolvers.GMRESSymbolicSetup(solver)
end

function Gridap.Algebra.numerical_setup(ss::GridapSolvers.LinearSolvers.GMRESSymbolicSetup, A)
   solver = ss.solver
   Pr_ns  = isa(solver.Pr,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pr,A),A)
   Pl_ns  = isa(solver.Pl,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pl,A),A)
   caches = GridapSolvers.LinearSolvers.get_solver_caches(solver,A)
   return GridapSolvers.LinearSolvers.GMRESNumericalSetup(solver,A,Pr_ns,Pl_ns,caches)
end
  
function Gridap.Algebra.numerical_setup!(ns::GridapSolvers.LinearSolvers.GMRESNumericalSetup, A::LinearMap)
   if !isa(ns.Pr_ns,Nothing)
       numerical_setup!(ns.Pr_ns,A)
   end
   if !isa(ns.Pl_ns,Nothing)
       numerical_setup!(ns.Pl_ns,A)
   end
   ns.A = A
end

function GridapDistributed.allocate_in_domain(A::LinearMap)
    T = eltype(A)
    return pfill(zero(T),partition(axes(A,2)))
end
  
function GridapSolvers.LinearSolvers.get_solver_caches(solver::GMRESSolver,A::LinearMap)
    m, Pl, Pr = solver.m, solver.Pl, solver.Pr

    V  = [GridapDistributed.allocate_in_domain(A) for i in 1:m+1]
    zr = !isa(Pr,Nothing) ? GridapDistributed.allocate_in_domain(A) : nothing
    zl = GridapDistributed.allocate_in_domain(A)

    H = zeros(m+1,m)  # Hessenberg matrix
    g = zeros(m+1)    # Residual vector
    c = zeros(m)      # Givens rotation cosines
    s = zeros(m)      # Givens rotation sines
    return (V,zr,zl,H,g,c,s)
end