module TMP

using ChainRulesCore
import ChainRulesCore: rrule

function f(x)
  x[1]+x[2]*x[3]
end

const A = [1 1 1; 0 1 1; 0 0 1]

function g(q)
  A\q
end

function h(q)
  x = g(q)
  y = f(x)
  y
end

function rrule(::typeof(f),x)
  function f_pullback(dy)
    NO_FIELDS, dy*[1,x[3],x[2]]
  end
  f(x), f_pullback
end

function rrule(::typeof(g),q)
  function g_pullback(dx)
    NO_FIELDS, transpose(A)\dx
  end
  g(q), g_pullback
end

q = [1,2,3]
x, g_pullback = rrule(g,q)
y, f_pullback = rrule(f,x)

dy = 1
_, dx = f_pullback(dy)
_, dq = g_pullback(dx)

@show q
@show x
@show y
@show dy
@show dx
@show dq

using ForwardDiff
dq3 = ForwardDiff.gradient(h,q)
@show dq3

using Zygote
dq2, = Zygote.gradient(h,q)
@show dq2

end # module

