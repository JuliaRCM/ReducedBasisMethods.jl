
import Base: collect, getindex, length, maximum, minimum, size, NamedTuple


AbstractSample{DT} = Union{Nothing, AbstractVector{DT}}

_sort(s::Nothing) = s
_sort(s::AbstractVector) = sort(unique(s))

function key_index(nt::NamedTuple, k::Symbol)
    @assert haskey(nt, k)
    ntkeys = keys(nt)
    for i in eachindex(ntkeys)
        if ntkeys[i] == k
            return i
        end
    end
end


"""

"""
struct Parameter{DT <: Number, ST <: AbstractSample{DT}}
    name::Symbol
    minimum::DT
    maximum::DT
    samples::ST

    function Parameter(name::Symbol, minimum::DT, maximum::DT, samples::ST) where {DT, ST <: AbstractSample{DT}}
        @assert minimum ≤ maximum
        if typeof(samples) <: AbstractVector
            @assert all([minimum ≤ s for s in samples])
            @assert all([maximum ≥ s for s in samples])
        end
        new{DT,ST}(name, minimum, maximum, _sort(samples))
    end
end

Parameter(name, minimum::DT, maximum::DT) where {DT} = Parameter(name, minimum, maximum, nothing)
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

Base.maximum(p::Parameter) = p.maximum
Base.minimum(p::Parameter) = p.minimum
Base.collect(p::Parameter) = p.samples

Base.length(p::Parameter{DT,ST}) where {DT, ST <: Nothing} = 0
Base.length(p::Parameter{DT,ST}) where {DT, ST <: AbstractVector} = length(p.samples)

Base.size(p::Parameter{DT,ST}) where {DT, ST <: Nothing} = (0,)
Base.size(p::Parameter{DT,ST}) where {DT, ST <: AbstractVector} = size(p.samples)

hassamples(p::Parameter{DT,ST}) where {DT, ST <: Nothing} = false
hassamples(p::Parameter{DT,ST}) where {DT, ST <: AbstractVector} = length(p.samples) > 0


function Base.NamedTuple(parameters::Vararg{Parameter{DT}}) where {DT}
    names = Tuple(p.name for p in parameters)
    NamedTuple{names}(parameters)
end


"""

"""
abstract type ParameterSampler end

"""

"""
function sample(ps::ParameterSampler, parameters::NamedTuple)
    sample(ps, values(parameters)...)
end


"""

"""
struct CartesianParameterSampler <: ParameterSampler end


function sample(::CartesianParameterSampler, parameters::Vararg{Parameter,N}) where {N}
    # make sure all parameters have a sample vector
    for p in parameters
        @assert hassamples(p)
    end

    # get all parameter index combinations
    inds = CartesianIndices(zeros([length(p) for p in parameters]...))[:]

    # generate sample matrix
    [Tuple((parameters[i].samples[inds[j][i]] for i in 1:N)) for j in eachindex(inds)]
end


"""
ParameterSpace collects all parameters of a system as well as samples in the parameter space.
"""
struct ParameterSpace{PT <: NamedTuple, ST <: Vector{<:Tuple}}
    parameters::PT
    samples::ST

    function ParameterSpace(parameters::PT, samples::ST) where {PT,ST}
        for s in samples
            @assert length(parameters) == length(s)
        end
        new{PT,ST}(parameters, samples)
    end
end

function ParameterSpace(sampler::ParameterSampler, parameters::Vararg{Parameter})
    ParameterSpace(NamedTuple(parameters...), sample(sampler, parameters...))
end

function ParameterSpace(parameters::Vararg{Parameter})
    ParameterSpace(CartesianParameterSampler(), parameters...)
end

function ParameterSpace(parameters::NamedTuple)
    ParameterSpace(values(parameters)...)
end

Base.:(==)(ps1::ParameterSpace, ps2::ParameterSpace) = (
                        ps1.parameters == ps2.parameters
                     && ps1.samples    == ps2.samples)


(ps::ParameterSpace)(i::Union{Int,CartesianIndex}) = NamedTuple{keys(ps.parameters)}(ps.samples[i])


Base.collect(ps::ParameterSpace) = vcat(transpose.(collect.(ps.samples))...)
Base.length(ps::ParameterSpace) = length(ps.samples)
Base.ndims(ps::ParameterSpace) = length(ps.parameters)
Base.size(ps::ParameterSpace) = (length(ps.samples), length(ps.parameters))
Base.size(ps::ParameterSpace, d) = size(ps)[d]

@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i) = collect(ps.samples[i])
@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i, j) = ps.samples[i][j]
@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, ::Colon) = collect(ps)
@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, ::Colon, ::Colon) = collect(ps)
@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, i::Union{Int,CartesianIndex}, ::Colon) = ps[i]
@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, ::Colon, j::Union{Int,CartesianIndex}) = [s[j] for s in ps.samples]

@inline Base.@propagate_inbounds function Base.getindex(ps::ParameterSpace, p::Symbol)
    j = key_index(ps.parameters, p)
    [s[j] for s in ps.samples]
end

@inline Base.@propagate_inbounds function Base.getindex(ps::ParameterSpace, i, p::Symbol)
    j = key_index(ps.parameters, p)
    ps[i,j]
end
