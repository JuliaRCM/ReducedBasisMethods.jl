### Reduced Tensor
# A m₁ × m₂ × N tensor where the first two indices are projected onto the columns of two matrices

"""
    ReducedTensor(tensor::GridTensor{DT, 3}, Pi::AbstractMatrix, Pj::AbstractMatrix)

The lazy `size(Pi, 2) × size(Pj, 2) × size(tensor, 3)` array that projects the first two indices of
the rank-3 `GridTensor` `tensor` onto the columns of `Pi` and `Pj`: `rt[i, j, k]` is
`tensor[m, n, k] * Pi[m, i] * Pj[n, j]` summed over the tensor's entries.

The sum follows the tensor's locality contract. With `S(m)` the periodic stencil of the tensor's
own width that `stencil_indices(tensor, m)` returns, `tensor[m, n, k]` is nonzero only where
`n ∈ S(m)` and `k ∈ S(m)`. The stencil is symmetric, `k ∈ S(m)` exactly where `m ∈ S(k)`, so

```math
rt[i, j, k] = \\sum_{m ∈ S(k)} \\sum_{n ∈ S(m)} tensor[m, n, k] \\, Pi[m, i] \\, Pj[n, j] .
```

The contract is what this type assumes of the tensor and never checks: a check would read every
entry.

Every direction of the grid needs at least `2w + 1` nodes for the stencil width `w`, or the
stencil names a node twice and the constructor throws an `ArgumentError`. `Pi` and `Pj` need
`size(tensor, 1)` and `size(tensor, 2)` rows, or the constructor throws a `DimensionMismatch`, and
their element type has to be `DT`, or it throws an `ArgumentError`. An index off the array throws a
`BoundsError`. The projections are one-based; offset axes are neither checked nor claimed.

```julia
julia> using MultiIndexArrays: GridTensor, MultiIndexLazyArray

julia> gt = GridTensor(MultiIndexLazyArray(Float64, (I, J, K) -> 1.0,
           (3, 3), (3, 3), (3, 3)), 1);   # the width-1 stencil covers the 3 × 3 grid

julia> rt = ReducedTensor(gt, ones(9, 2), ones(9, 1));

julia> size(rt)
(2, 1, 9)
```
"""
struct ReducedTensor{DT, GT <: GridTensor{DT, 3}, PM1, PM2} <: AbstractArray{DT, 3}
    tensor::GT
    projection_i::PM1
    projection_j::PM2

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
        # a uniform periodic stencil repeats a node at every centre or at none
        allunique(stencil_indices(tensor, 1)) || throw(ArgumentError(
            "the stencil names a grid node twice: every direction needs at least 2w + 1 nodes"))
        return new{DT, typeof(tensor), typeof(Pi), typeof(Pj)}(tensor, Pi, Pj)
    end
end

function Base.size(rt::ReducedTensor)
    (size(rt.projection_i, 2), size(rt.projection_j, 2), size(rt.tensor, 3))
end

function Base.getindex(rt::ReducedTensor{DT}, i::Int, j::Int, k::Int) where {DT}
    checkbounds(rt, i, j, k)
    T = rt.tensor
    x = zero(DT)
    for m in stencil_indices(T, k)
        for n in stencil_indices(T, m)
            x += T[m, n, k] * rt.projection_i[m, i] * rt.projection_j[n, j]
        end
    end
    return x
end
