
using LinearAlgebra
using MultiIndexArrays
using OffsetArrays
using Random
using ReducedBasisMethods
using Test
using LinearAlgebra


function _cartesian(ax::MultiIndexAxis, ind::CartesianIndex, i, j)
    CartesianIndex(mod1(ind[1]+i, size(ax,1)), mod1(ind[2]+j, size(ax,2)))
end

function eachstencilindex(ax::MultiIndexAxis, ind::Int, width::Int)
    cind = ax[ind]
    ( ax[_cartesian(ax, cind, i, j)] for i in -width:+width, j in -width:+width )
end

struct ReducedTensor{DT, PT <: AbstractMultiIndexArray{DT,3}, PM1, PM2} <: AbstractArray{DT,3}
    tensor::PT
    projection_i::PM1
    projection_j::PM2
    stencil_width::Int

    function ReducedTensor(tensor::AbstractMultiIndexArray{DT}, Pi::PM1, Pj::PM2) where {DT, PM1, PM2}
        @assert size(Pi, 2) == size(tensor, 1)
        @assert size(Pj, 1) == size(tensor, 2)
        new{DT, typeof(tensor), PM1, PM2}(tensor, Pi, Pj, 1)
    end
end

Base.size(rt::ReducedTensor) = (size(rt.projection_i, 1), size(rt.projection_j, 2), size(rt.tensor, 3))
Base.size(rt::ReducedTensor, i) = size(rt)[i]
Base.axes(rt::ReducedTensor, i) = Base.OneTo(size(rt, i))

function Base.getindex(rt::ReducedTensor{DT}, i::Int, j::Int, k::Int) where {DT}
    @assert i ≥ 1 && i ≤ size(rt, 1)
    @assert j ≥ 1 && j ≤ size(rt, 2)
    @assert k ≥ 1 && k ≤ size(rt, 3)

    local x = zero(DT)
    local w = rt.stencil_width
    
    # Account for stencil size and loop only over nonzero entries
    for m in eachstencilindex(axes(rt.tensor, 1), k, w)
        for n in eachstencilindex(axes(rt.tensor, 2), k, w)
            x += rt.projection_i[i,m] * rt.tensor[m,n,k] * rt.projection_j[n,j]
        end
    end

    return x
end

function Base.materialize(rt::ReducedTensor)
    [ rt[i,j,k] for i in axes(rt,1), j in axes(rt,2), k in axes(rt,3) ]
end



struct PoissonOperator{DT, PT, HT} <: AbstractMatrix{DT}
    tensor::PT
    hamiltonian::HT

    function PoissonOperator(tensor::Union{AbstractMultiIndexArray{DT}, ReducedTensor{DT}}, h::HT) where {DT, HT}
        new{DT, typeof(tensor), HT}(tensor, h)
    end
end

Base.size(po::PoissonOperator) = (size(po.tensor, 1), size(po.tensor, 2))
Base.size(po::PoissonOperator, i) = size(po)[i]

Base.axes(po::PoissonOperator, i) = Base.OneTo(size(po.tensor, i))

function Base.getindex(po::PoissonOperator{DT}, i::Int, j::Int) where {DT}
    @assert i ≥ 1 && i ≤ size(po, 1)
    @assert j ≥ 1 && j ≤ size(po, 2)

    local x = zero(DT)

    @inbounds for k in eachindex(po.hamiltonian)
        x += po.tensor[i,j,k] * po.hamiltonian[k]
    end

    return x
end

function Base.materialize(rt::PoissonOperator{DT}) where {DT}
    [ rt[i,j] for i in axes(rt,1), j in axes(rt,2) ]
end



### Arakawa ###

"""

jpp = 
    + f[i-1, j  ] * h[i,   j-1]
    - f[i-1, j  ] * h[i,   j+1]
    - f[i,   j-1] * h[i-1, j  ]
    + f[i,   j-1] * h[i+1, j  ]
    + f[i,   j+1] * h[i-1, j  ]
    - f[i,   j+1] * h[i+1, j  ]
    - f[i+1, j  ] * h[i,   j-1]
    + f[i+1, j  ] * h[i,   j+1]

jpc =
    + f[i-1, j  ] * h[i-1, j-1]
    - f[i-1, j  ] * h[i-1, j+1]
    - f[i,   j-1] * h[i-1, j-1]
    + f[i,   j-1] * h[i+1, j-1]
    + f[i,   j+1] * h[i-1, j+1]
    - f[i,   j+1] * h[i+1, j+1]
    - f[i+1, j  ] * h[i+1, j-1]
    + f[i+1, j  ] * h[i+1, j+1]

jcp = 
    - f[i-1, j-1] * h[i-1, j  ]
    + f[i-1, j-1] * h[i,   j-1]
    + f[i-1, j+1] * h[i-1, j  ]
    - f[i-1, j+1] * h[i,   j+1]
    - f[i+1, j-1] * h[i,   j-1]
    + f[i+1, j-1] * h[i+1, j  ]
    + f[i+1, j+1] * h[i,   j+1]
    - f[i+1, j+1] * h[i+1, j  ]

return (jpp + jpc + jcp) * inv(hx) * inv(hv) / 12

"""
struct Arakawa{DT}
    nx::Int
    nv::Int
    hx::DT
    hv::DT
    factor::DT

    JPP::OffsetArray{Int, 4, Array{Int, 4}}
    JPC::OffsetArray{Int, 4, Array{Int, 4}}
    JCP::OffsetArray{Int, 4, Array{Int, 4}}

    function Arakawa(nx::Int, nv::Int, hx::DT, hv::DT) where {DT}
        JPP = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)
        JPC = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)
        JCP = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)
    
        JPP[-1,  0,  0, -1] = +1
        JPP[-1,  0,  0, +1] = -1
        JPP[ 0, -1, -1,  0] = -1
        JPP[ 0, -1, +1,  0] = +1
        JPP[ 0, +1, -1,  0] = +1
        JPP[ 0, +1, +1,  0] = -1
        JPP[+1,  0,  0, -1] = -1
        JPP[+1,  0,  0, +1] = +1
    
        JPC[-1,  0, -1, -1] = +1
        JPC[-1,  0, -1, +1] = -1
        JPC[ 0, -1, -1, -1] = -1
        JPC[ 0, -1, +1, -1] = +1
        JPC[ 0, +1, -1, +1] = +1
        JPC[ 0, +1, +1, +1] = -1
        JPC[+1,  0, +1, -1] = -1
        JPC[+1,  0, +1, +1] = +1
    
        JCP[-1, -1, -1,  0] = -1
        JCP[-1, -1,  0, -1] = +1
        JCP[-1, +1, -1,  0] = +1
        JCP[-1, +1,  0, +1] = -1
        JCP[+1, -1,  0, -1] = -1
        JCP[+1, -1, +1,  0] = +1
        JCP[+1, +1,  0, +1] = +1
        JCP[+1, +1, +1,  0] = -1
    
        factor = inv(hx) * inv(hv) / 12

        new{DT}(nx, nv, hx, hv, factor, JPP, JPC, JCP)
    end
end

mymod(i, n, w=1) = abs(i) ≥ n - w ? i - sign(i) * n : i

function Base.isvalid(I::CartesianIndex, arakawa::Arakawa)
    I[1] ≥ 1 && I[1] ≤ arakawa.nx &&
    I[2] ≥ 1 && I[2] ≤ arakawa.nv
end

function (arakawa::Arakawa{DT})(I, J, K) where {DT}
    @assert isvalid(I, arakawa)
    @assert isvalid(J, arakawa)
    @assert isvalid(K, arakawa)

    fi = mymod.(Tuple(J - I), (arakawa.nx, arakawa.nv))
    hi = mymod.(Tuple(K - I), (arakawa.nx, arakawa.nv))

    if any(fi .< -1) || any(fi .> +1) || any(hi .< -1) || any(hi .> +1)
        return zero(DT)
    end

    ( arakawa.JPP[fi..., hi...] +
      arakawa.JPC[fi..., hi...] +
      arakawa.JCP[fi..., hi...] ) * arakawa.factor
end


### Tests ###

function test_arakawa()

    nx = 25
    nv = 33
    hx = 1 / (nx-1)
    hv = 1 / (nv-1)
    n = nx * nv

    arakawa = Arakawa(nx, nv, hx, hv)

    Random.seed!(1234)

    f = rand(n)
    g = rand(n)
    h = rand(n)

    P_tens = MultiIndexLazyArray( Float64, (inds...) -> arakawa(inds...), (nx,nv), (nx,nv), (nx,nv) )

    P_tens_op = PoissonOperator( P_tens, h )

    # P_tens_op_arr = Base.materialize(P_tens_op)

    println( dot(g, P_tens_op, f) )

end

# test_arakawa()


function test_reduction()

    nx = 128
    nv = 256
    hx = 1 / (nx-1)
    hv = 1 / (nv-1)
    n = nx * nv
    m = 24

    arakawa = Arakawa(nx, nv, hx, hv)

    projection = rand(n,m)

    P_tens = MultiIndexLazyArray( Float64, (inds...) -> arakawa(inds...), (nx,nv), (nx,nv), (nx,nv) )

    P_tens_red = ReducedTensor( P_tens, projection', projection )

    @time Base.materialize(P_tens_red)
    @time Base.materialize(P_tens_red)

end

test_reduction()
