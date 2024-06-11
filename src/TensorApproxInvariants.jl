module TensorApproxInvariants
export DerApprox

include("tensors.jl")
export tensor_tight

using Tullio
using KrylovKit
using LoopVectorization

function DerApprox(T,howmany=3)
  a,b,c = size(T)
  function A(omega)
    x=reshape(omega[1:a^2],a,a)
    y=reshape(omega[a^2+1:a^2+b^2],b,b)
    z=reshape(omega[a^2+b^2+1:end],c,c)
    @tullio Tp[i,j,k] := x[i,ip]*T[ip,j,k]
    @tullio Tp[i,j,k] += y[j,jp]*T[i,jp,k]
    @tullio Tp[i,j,k] += z[k,kp]*T[i,j,kp]
    return reshape(Tp,:)
  end
  function At(Tp)
    Tp = reshape(Tp,size(T))
    xout = zeros(eltype(T),a,a)
    yout = zeros(eltype(T),b,b)
    zout = zeros(eltype(T),c,c)
    @tullio xout[i,ip] = T[ip,j,k]*Tp[i,j,k]
    @tullio yout[j,jp] = T[i,jp,k]*Tp[i,j,k]
    @tullio zout[k,kp] = T[i,j,kp]*Tp[i,j,k]
    return vcat(reshape(xout,:), reshape(yout,:), reshape(zout,:))
  end
    

  vals, lvecs, rvecs, info = svdsolve((At,A), randn(eltype(T),a^2+b^2+c^2), howmany, :SR) 
  return (vals,lvecs,rvecs,info)

  # return A,At, randn(a*b*c), randn(a^2+b^2+c^2)

  
end

end # module TensorApproxInvariants
