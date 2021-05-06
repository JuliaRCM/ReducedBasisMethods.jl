
import Base: getindex, NamedTuple


"""

"""
abstract type ParameterSampler end


"""

"""
struct Parameter{DT <: Number}
    name::Symbol
    minimum::DT
    maximum::DT
    samples::Vector{DT}

    function Parameter(name::Symbol, minimum::DT, maximum::DT, samples::AbstractVector{DT}) where {DT}
        @assert minimum ≤ maximum
        @assert all([minimum ≤ s for s in samples])
        @assert all([maximum ≥ s for s in samples])
        new{DT}(name, minimum, maximum, sort(unique(samples)))
    end
end

Parameter(name, minimum::DT, maximum::DT) where {DT} = Parameter(name, minimum, maximum, Vector{DT}())
Parameter(name, minimum::DT, maximum::DT, n::Int) where {DT} = Parameter(name, minimum, maximum, LinRange(minimum, maximum, n))
Parameter(name, samples::AbstractVector) = Parameter(name, minimum(samples), maximum(samples), samples)

Base.hash(p::Parameter, h::UInt) = hash(p.name, hash(p.minimum, hash(p.maximum, hash(p.samples, h))))

Base.:(==)(p1::Parameter, p2::Parameter) = (
                                p1.name    == p2.name
                             && p1.minimum == p2.minimum
                             && p1.maximum == p2.maximum
                             && p1.samples == p2.samples)

function show(io::IO, p::Parameter)
    println(io, "Parameter $(p.name) with ")
    println(io, "   minimum = ", p.minimum)
    println(io, "   maximum = ", p.maximum)
    println(io, "   samples = ")
    show(io, p.samples)
end

Base.length(p::Parameter) = length(p.samples)
Base.size(p::Parameter) = size(p.samples)


function Base.NamedTuple(parameters::Vararg{Parameter{DT}}) where {DT}
    names = Tuple(p.name for p in parameters)
    NamedTuple{names}(parameters)
end


function parameter_grid(parameters::Vararg{Parameter{DT},N}) where {DT,N}
    # get all parameter index combinations
    inds = CartesianIndices(zeros([length(p) for p in parameters]...))[:]

    # generate sample matrix
    [parameters[i].samples[inds[j][i]] for i in 1:N, j in eachindex(inds)]
end



"""
ParameterSpace collects all parameters as well as samples in the space.
"""
struct ParameterSpace{DT <: Number, PT <: NamedTuple}
    parameters::PT
    samples::Matrix{DT}

    function ParameterSpace(parameters::PT, samples::AbstractMatrix{DT}) where {DT, PT}
        @assert length(parameters) == size(samples,1)
        new{DT,PT}(parameters, samples)
    end
end

function ParameterSpace(parameters::Vararg{Parameter{DT}}) where {DT}
    ParameterSpace(NamedTuple(parameters...), parameter_grid(parameters...))
end

# function ParameterSpace(parameters::NamedTuple, sampler::ParameterSampler) end

Base.:(==)(ps1::ParameterSpace, ps2::ParameterSpace) = (
                        ps1.parameters == ps2.parameters
                     && ps1.samples    == ps2.samples)

@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i) = getindex(ps.samples, i)
@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i, j) = getindex(ps.samples, i, j)

# @inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i::Union{Int,CartesianIndex}) = getindex(ps.samples, i)
# @inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, I::Vararg{Int}) = getindex(ps.samples, I...)
# @inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i::Colon, j::Colon) = getindex(ps.samples, axes(ps.samples,1), axes(ps.samples,2))
# @inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i, j::Colon) = getindex(ps.samples, i, axes(ps.samples,2))
# @inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i::Colon, j) = getindex(ps.samples, axes(ps.samples,1), j)

function key_index(nt::NamedTuple, k::Symbol)
    @assert haskey(nt, k)
    ntkeys = keys(nt)
    for i in eachindex(ntkeys)
        if ntkeys[i] == k
            return i
        end
    end
end

@inline Base.@propagate_inbounds function Base.getindex(ps::ParameterSpace, p::Symbol)
    i = key_index(ps.parameters, p)
    getindex(ps.samples, i)
end

@inline Base.@propagate_inbounds function Base.getindex(ps::ParameterSpace, p::Symbol, j)
    i = key_index(ps.parameters, p)
    getindex(ps.samples, i, j)
end

Base.size(ps::ParameterSpace) = size(ps.samples)
Base.size(ps::ParameterSpace, d) = size(ps.samples, d)

