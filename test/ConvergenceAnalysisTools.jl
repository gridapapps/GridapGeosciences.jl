function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

"""
Generates the values of n used for the convergence study within range
[1:min(10^k,n_max)] using a logarithmic distribution of the sample points.
"""
function generate_n_values(k;n_max=120)
  n_values=Int[]
  current=1
  for power=1:k
    for coefficient=1:10
       n=coefficient*current
       if (n>n_max)
        return n_values[2:end]
       end
       append!(n_values,n)
    end
    current=current*10
  end
  return n_values[2:end]
end

"""
   Performs a convergence study.

   Returns a tuple with three entries.
     [1] Values of h.
     [2] Relative errors.
     [3] Slope of the log(h)-log(err) relative error convergence curve.
"""
function convergence_study(f,n_values,geo_order,order,degree)
   hs=Float64[]
   errors=Float64[]
   for n in n_values
     model=CubedSphereDiscreteModel(n,geo_order)
     err=f(model,order,degree)
     append!(errors,err)
   end
   hs=[2.0/n for n in n_values]
   return hs,errors,slope(hs,errors)
end

function convergence_study(f,n_values,order,degree)
  hs=Float64[]
  errors=Float64[]
  for n in n_values
     model=CubedSphereDiscreteModel(n)
     err=f(model,order,degree)
     append!(errors,err)
  end
  hs=[2.0/n for n in n_values]
  return hs,errors,slope(hs,errors)
end
