
import Base: collect, getindex, length, ndims, size

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
