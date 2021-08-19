"""
  perp(u,n)

  cross product of vector-valued field u and manifold's outward unit normal
"""
function perp(u::CellField,n::CellField)
   u×n
end
