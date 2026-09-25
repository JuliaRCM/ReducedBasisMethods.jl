### Reduced Tensor
# A m × m × N tensor where the first two indices are reduced with projection matrices

struct ReducedTensor{DT, PT <: PoissonTensor{DT}, PM1, PM2} <: AbstractArray{DT, 3}
    tensor::PT
    projection_i::PM1
    projection_j::PM2

    function ReducedTensor(tensor::PoissonTensor{DT}, Pi::PM1, Pj::PM2) where {DT, PM1, PM2}
        @assert size(Pi, 1) == size(tensor, 1)
        @assert size(Pj, 1) == size(tensor, 2)
        new{DT, typeof(tensor), PM1, PM2}(tensor, Pi, Pj)
    end
end

function Base.size(rt::ReducedTensor)
    (size(rt.projection_i, 2), size(rt.projection_j, 2), size(rt.tensor, 3))
end

function Base.getindex(rt::ReducedTensor{DT}, i::Int, j::Int, k::Int) where {DT}
    @assert i ≥ 1 && i ≤ size(rt, 1)
    @assert j ≥ 1 && j ≤ size(rt, 2)
    @assert k ≥ 1 && k ≤ size(rt, 3)

    local x = zero(DT)

    nk = _stencil_indices(k, 1, _nx(rt), _nv(rt))

    for m in nk
        for n in nk
            x += rt.tensor[m, n, k] * rt.projection_i[m, i] * rt.projection_j[n, j]
        end
    end

    return x
end

_nx(t::ReducedTensor) = _nx(t.tensor)
_nv(t::ReducedTensor) = _nv(t.tensor)
