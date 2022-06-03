
using OffsetArrays
#using ReducedBasisMethods
#using Test
using LinearAlgebra

### Indices

function Base.isvalid(I::CartesianIndex, nx, nv)
    I[1] ≥ 1 && I[1] ≤ nx &&
    I[2] ≥ 1 && I[2] ≤ nv
end

function multiindex(i, nx, nv)
    @assert i ≥ 1 && i ≤ nx*nv
    CartesianIndex(mod1(i, nx), div(i-1, nx) + 1)
end

function linearindex(I, nx, nv)
    i, j = Tuple(I)
    @assert i ≥ 1 && i ≤ nx
    @assert j ≥ 1 && i ≤ nv
    (j-1) * nx + i
end

# Given a linear index i, convert it to a CartesianIndex (i₁, i₂, … )
# Find the indices in the stencil (width w) around i: (i₁ ± w₁, i₂ ± w₂, … )
# Convert these back to linear indices and return them in a Tuple with (2w-1)^d - 1 entries
# Boundaries are periodic
function _stencil_indices(i::Int, w::Int, nx, nv)
    indices = zeros(Int,(2*w+1)^2)
    ij = Tuple( multiindex(i, nx, nv) )
    k = 1
    for d1 in -w:w, d2 in -w:w
        indices[k] = linearindex( CartesianIndex( mod1.( ij .+ (d1,d2), (nx, nv) ) ), nx, nv )
        k += 1
    end
    return indices
end

### Poisson Tensor (used with h)
# A N × N × N tensor that discretizes the weak form of the Arakawa bracket g[f,h]

struct PoissonTensor{DT,FT}
    nx::Int
    nv::Int
    f::FT

    function PoissonTensor(DT, nx, nv, f)
        new{DT, typeof(f)}(nx, nv, f)
    end
end

Base.size(pt::PoissonTensor) = tuple(pt.nx * pt.nv * ones(Int,3)...)
Base.size(pt::PoissonTensor, i) = i ≥ 1 && i ≤ 3 ? pt.nx * pt.nv : 1


function Base.getindex(pt::PoissonTensor, I::CartesianIndex, J::CartesianIndex, K::CartesianIndex)
    @assert isvalid(I, pt.nx, pt.nv)
    @assert isvalid(J, pt.nx, pt.nv)
    @assert isvalid(K, pt.nx, pt.nv)

    pt.f(I, J, K)
end

function Base.getindex(pt::PoissonTensor, i::Int, j::Int, k::Int)
    I = multiindex(i, pt.nx, pt.nv)
    J = multiindex(j, pt.nx, pt.nv)
    K = multiindex(k, pt.nx, pt.nv)

    pt[I,J,K]
end

function Base.materialize(rt::PoissonTensor)
    [ rt[i,j,k] for i in 1:size(rt,1), j in 1:size(rt,2), k in 1:size(rt,3) ]
end

### Reduced Tensor (using with full h)
# A m × m × N tensor where the first two indices are reduced with projection matrices

struct ReducedTensor{DT, PT <: PoissonTensor{DT}, PM1, PM2} <: AbstractArray{DT,3}
    tensor::PT
    projection_i::PM1
    projection_j::PM2

    function ReducedTensor(tensor::PoissonTensor{DT}, Pi::PM1, Pj::PM2) where {DT, PM1, PM2}
        @assert size(Pi, 1) == size(tensor, 1)
        @assert size(Pj, 1) == size(tensor, 2)
        new{DT, typeof(tensor), PM1, PM2}(tensor, Pi, Pj)
    end
end

Base.size(rt::ReducedTensor) = (size(rt.projection_i, 2), size(rt.projection_j, 2), size(rt.tensor, 3))
Base.size(rt::ReducedTensor, i) = size(rt)[i]
Base.axes(rt::ReducedTensor, i) = Base.OneTo(size(rt, i))

function Base.getindex(rt::ReducedTensor{DT}, i::Int, j::Int, k::Int) where {DT}
    @assert i ≥ 1 && i ≤ size(rt, 1)
    @assert j ≥ 1 && j ≤ size(rt, 2)
    @assert k ≥ 1 && k ≤ size(rt, 3)

    local x = zero(DT)

    nk = _stencil_indices(k, 1, _nx(rt), _nv(rt))
    
    for m in nk
        for n in nk
            x += rt.tensor[m,n,k] * rt.projection_i[m,i] * rt.projection_j[n,j]
        end
    end

    return x
end

### Poisson Operator
# weak form of the Operator f ↦ [f,h]

struct PoissonOperator{DT, PT, HT} <: AbstractMatrix{DT}
    tensor::PT
    hamiltonian::HT

    function PoissonOperator(tensor::PoissonTensor{DT}, h::HT) where {DT, HT}
        new{DT, typeof(tensor), HT}(tensor, h)
    end
end

Base.size(po::PoissonOperator) = (size(po.tensor, 1), size(po.tensor, 2))
Base.size(po::PoissonOperator, i) = size(po)[i]

_nx(t::PoissonTensor) = t.nx
_nv(t::PoissonTensor) = t.nv

_nx(t::PoissonOperator) = _nx(t.tensor)
_nv(t::PoissonOperator) = _nv(t.tensor)

_nx(t::ReducedTensor) = _nx(t.tensor)
_nv(t::ReducedTensor) = _nv(t.tensor)

function Base.getindex(po::PoissonOperator{DT}, i::Int, j::Int) where {DT}
    @assert i ≥ 1 && i ≤ size(po, 1)
    @assert j ≥ 1 && j ≤ size(po, 2)

    local x = zero(DT)

    ni = _stencil_indices(i, 1, _nx(po), _nv(po)) # neighboring indices of i
    #nj = _stencil_indices(j, 1, po.tensor.nx, po.tensor.nv) # neighboring indices of j

    @inbounds for k in ni
        x += po.tensor[i,j,k] * po.hamiltonian[k]
    end

    return x
end

function Base.materialize(rt::PoissonOperator)
    [ rt[i,j] for i in 1:size(rt,1), j in 1:size(rt,2) ]
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

mymod(i, n, w=1) = abs(i) ≥ n - w ? i - n * sign(i) : i

function (arakawa::Arakawa{DT})(I, J, K) where {DT}
    fi = mymod.(Tuple(J - I), (arakawa.nx, arakawa.nv))
    hi = mymod.(Tuple(K - I), (arakawa.nx, arakawa.nv))

    if any(fi .< -1) || any(fi .> +1) || any(hi .< -1) || any(hi .> +1)
        return zero(DT)
    end

    ( arakawa.JPP[fi..., hi...] +
      arakawa.JPC[fi..., hi...] +
      arakawa.JCP[fi..., hi...] ) * arakawa.factor
end

### Reduced Tensor (used with full ϕ)
# A m × m × m₁ tensor where the first two indices are reduced using a projection on the phase space modes of f and the third on the modes of ϕ
# P̃[i,j,k] = ∑(l,m,(n₁,n₂)) P[l,m,n] Π¹[l,i] Π²[m,j] Π³[n₁(n),k]  

struct PotentialReducedTensor{DT, PT <: PoissonTensor{DT}, PM1, PM2, PM3} <: AbstractArray{DT,3}
    tensor::PT
    projection_i::PM1
    projection_j::PM2
    projection_α::PM3

    function PotentialReducedTensor(tensor::PoissonTensor{DT}, Pi::PM1, Pj::PM2, Pα::PM3) where {DT, PM1, PM2, PM3}
        @assert size(Pi, 1) == size(tensor, 1)
        @assert size(Pj, 1) == size(tensor, 2)
        @assert size(Pα, 1) == tensor.nx
        new{DT, typeof(tensor), PM1, PM2, PM3}(tensor, Pi, Pj, Pα)
    end
end

Base.size(rt::PotentialReducedTensor) = (size(rt.projection_i, 2), size(rt.projection_j, 2), size(rt.projection_α, 2))
Base.size(rt::PotentialReducedTensor, i) = size(rt)[i]
Base.axes(rt::PotentialReducedTensor, i) = Base.OneTo(size(rt, i))

function Base.getindex(rt::PotentialReducedTensor{DT}, i::Int, j::Int, α::Int) where {DT}
    @assert i ≥ 1 && i ≤ size(rt, 1)
    @assert j ≥ 1 && j ≤ size(rt, 2)
    @assert α ≥ 1 && α ≤ size(rt, 3)

    local nx = rt.tensor.nx
    local nv = rt.tensor.nv

    local r = zero(DT)

    # k1 here is the first index of the CartesianIndex K that describes the x,v space

    for k in 1:nx*nv
        nk = _stencil_indices(k, 1, nx, nv)
        k1 = Tuple(multiindex(k, nx, nv))[1]
        for m in nk
            for n in nk
                r += rt.tensor[m,n,k] * rt.projection_i[m,i] * rt.projection_j[n,j] * rt.projection_α[k1,α]
            end
        end
    end

    return r
end

#function Base.materialize(rt::Union{ReducedTensor,PotentialReducedTensor})
#    [ rt[i,j,k] for i in axes(rt,1), j in axes(rt,2), k in axes(rt,3) ]
#end

### Velocity Reduced Matrix
# A m × m matrix where the first two indices are reduced using a projection on the phase space modes of f and the third dimension was contarcted with v²/2
# P̃[i,j] = ∑(l,m,(n₁,n₂)) P[l,m,n] Π¹[l,i] Π²[m,j] 1/2 v²[n₂(n)]

struct VelocityReducedMatrix{DT, PT <: PoissonTensor{DT}, PM1, PM2, PV <: AbstractVector{DT}} <: AbstractArray{DT,2}
    tensor::PT
    projection_i::PM1
    projection_j::PM2
    v::PV

    function VelocityReducedMatrix(tensor::PoissonTensor{DT}, Pi::PM1, Pj::PM2, v::PV) where {DT, PM1, PM2, PV}
        @assert size(Pi, 1) == size(tensor, 1)
        @assert size(Pj, 1) == size(tensor, 2)
        new{DT, typeof(tensor), PM1, PM2, PV}(tensor, Pi, Pj, v)
    end
end

Base.size(rt::VelocityReducedMatrix) = (size(rt.projection_i, 2), size(rt.projection_j, 2))
Base.size(rt::VelocityReducedMatrix, i) = size(rt)[i]
Base.axes(rt::VelocityReducedMatrix, i) = Base.OneTo(size(rt, i))

function Base.getindex(rt::VelocityReducedMatrix{DT}, i::Int, j::Int) where {DT}
    @assert i ≥ 1 && i ≤ size(rt, 1)
    @assert j ≥ 1 && j ≤ size(rt, 2)

    local nx = rt.tensor.nx
    local nv = rt.tensor.nv

    local r = zero(DT)

    for k in 1:nx*nv
        nk = _stencil_indices(k, 1, nx, nv)
        k2 = Tuple(multiindex(k, nx, nv))[2]
        for m in nk
            for n in nk
                r += rt.tensor[m,n,k] * rt.projection_i[m,i] * rt.projection_j[n,j] * 0.5 * rt.v[k2]^2
            end
        end
    end

    return r
end