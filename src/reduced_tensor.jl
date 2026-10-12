### Reduced Tensor
# A m₁ × m₂ × N tensor where the first two indices are projected onto the columns of two matrices

"""
    ReducedTensor(tensor::GridTensor{DT, 3}, Pi::AbstractMatrix, Pj::AbstractMatrix)

The lazy `size(Pi, 2) × size(Pj, 2) × size(tensor, 3)` array that projects the first two indices of
the rank-3 `GridTensor` `tensor` onto the columns of `Pi` and `Pj`: `rt[i, j, k]` is
`tensor[m, n, k] * Pi[m, i] * Pj[n, j]` summed over the tensor's entries.

The sum follows the tensor's locality contract. With `S(m)` the periodic stencil of the tensor's
own width that `stencil_indices(tensor, m)` returns, `tensor[m, n, k]` is nonzero only where
`n ∈ S(m)` and `k ∈ S(m)`, so

```math
rt[i, j, k] = \\sum_{m ∈ S(k)} \\sum_{n ∈ S(m)} tensor[m, n, k] \\, Pi[m, i] \\, Pj[n, j] .
```

The contract is what this type assumes of the tensor and never checks: a check would read every
entry. Where the tensor's stencil names a node twice, which a direction shorter than `2w + 1`
gives, the sum runs over every index of the tensor instead; that branch also reads entries outside
the contract.

`Pi` and `Pj` need `size(tensor, 1)` and `size(tensor, 2)` rows, or the constructor throws a
`DimensionMismatch`, and their element type has to be `DT`, or it throws an `ArgumentError`. An
index off the array throws a `BoundsError`. The projections are one-based; offset axes are neither
checked nor claimed.

```julia
julia> using MultiIndexArrays: GridTensor, MultiIndexLazyArray

julia> gt = GridTensor(MultiIndexLazyArray(Float64, (I, J, K) -> 1.0,
           (3, 4), (3, 4), (3, 4)), 1);

julia> rt = ReducedTensor(gt, ones(12, 2), ones(12, 1));

julia> size(rt)
(2, 1, 12)
```
"""
struct ReducedTensor{DT, GT <: GridTensor{DT, 3}, PM1, PM2} <: AbstractArray{DT, 3}
    tensor::GT
    projection_i::PM1
    projection_j::PM2
    stencil::Bool

    function ReducedTensor(tensor::GridTensor{DT, 3}, Pi::AbstractMatrix,
            Pj::AbstractMatrix) where {DT}
        size(Pi, 1) == size(tensor, 1) || throw(DimensionMismatch(
            "Pi needs $(size(tensor, 1)) rows, one per grid node, got $(size(Pi, 1))"))
        size(Pj, 1) == size(tensor, 2) || throw(DimensionMismatch(
            "Pj needs $(size(tensor, 2)) rows, one per grid node, got $(size(Pj, 1))"))
        eltype(Pi) == DT || throw(ArgumentError(
            "Pi needs the element type $DT of the tensor, got $(eltype(Pi))"))
        eltype(Pj) == DT || throw(ArgumentError(
            "Pj needs the element type $DT of the tensor, got $(eltype(Pj))"))
        return new{DT, typeof(tensor), typeof(Pi), typeof(Pj)}(tensor, Pi, Pj,
            _stencil(tensor))
    end
end

# The sums may be restricted to the tensor's stencil only where the stencil names no node twice,
# which a periodic uniform stencil does at every node; a direction shorter than `2w + 1` repeats
# one, and the repeated node would be counted twice.
_stencil(gt::GridTensor) = allunique(stencil_indices(gt, 1))

function Base.size(rt::ReducedTensor)
    (size(rt.projection_i, 2), size(rt.projection_j, 2), size(rt.tensor, 3))
end

function Base.getindex(rt::ReducedTensor{DT}, i::Int, j::Int, k::Int) where {DT}
    checkbounds(rt, i, j, k)
    return rt.stencil ? stencil_sum(rt, i, j, k) : dense_sum(rt, i, j, k)
end

# The nested sum over the tensor's own stencil: `m` runs over the stencil of `k` and `n` over the
# stencil of `m`, whose order `stencil_indices` keeps. One function per branch keeps each loop on
# one concrete iterator, which is what makes `getindex` type-stable and allocation-free.
function stencil_sum(rt::ReducedTensor{DT}, i::Int, j::Int, k::Int) where {DT}
    local x = zero(DT)

    for m in stencil_indices(rt.tensor, k)
        for n in stencil_indices(rt.tensor, m)
            x += rt.tensor[m, n, k] * rt.projection_i[m, i] * rt.projection_j[n, j]
        end
    end

    return x
end

# The sum over every index of the tensor, for a stencil that names a node twice.
function dense_sum(rt::ReducedTensor{DT}, i::Int, j::Int, k::Int) where {DT}
    local x = zero(DT)

    for m in axes(rt.tensor, 1)
        for n in axes(rt.tensor, 2)
            x += rt.tensor[m, n, k] * rt.projection_i[m, i] * rt.projection_j[n, j]
        end
    end

    return x
end
