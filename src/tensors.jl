using IterTools
using LinearAlgebra
using Tullio

function tensor_tight(n,m,l)
  T = zeros(n,m,l)
  t = div(n+m+l+3,2)
  for (i,j,k) in product(1:n,1:m,1:l)
    if i+j+k == t
      T[i,j,k] = randn()
    end
  end
  Q = collect(qr(randn(n,n)).Q)
  @tullio S[i,j,k] := Q[i,i1]*T[i1,j,k]
  Q = collect(qr(randn(m,m)).Q)
  @tullio T[i,j,k] = Q[j,j1]*S[i,j1,k]
  Q = collect(qr(randn(l,l)).Q)
  @tullio S[i,j,k] = Q[k,k1]*T[i,j,k1]
  return S
end
