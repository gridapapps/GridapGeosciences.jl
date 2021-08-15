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
        return n_values
       end
       append!(n_values,n)
    end
    current=current*10
  end
  return n_values
end
