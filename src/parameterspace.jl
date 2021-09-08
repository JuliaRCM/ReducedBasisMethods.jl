
import Base: collect, eachindex, getindex, length, ndims, size

"""
ParameterSpace collects all parameters of a system as well as samples in the parameter space.
"""
struct ParameterSpace{PT <: NamedTuple, ST <: Table}
    parameters::PT
    samples::ST

    function ParameterSpace(parameters::PT, samples::ST) where {PT,ST}
        # for s in samples
        #     @assert length(parameters) == length(s)
        # end
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


Base.collect(ps::ParameterSpace) = collect(ps.samples)
Base.eachindex(ps::ParameterSpace) = eachindex(ps.samples)
Base.length(ps::ParameterSpace) = length(ps.samples)
Base.ndims(ps::ParameterSpace) = length(ps.parameters)
Base.size(ps::ParameterSpace) = (length(ps.samples), length(ps.parameters))
Base.size(ps::ParameterSpace, d) = size(ps)[d]

@inline Base.@propagate_inbounds Base.getindex(ps::ParameterSpace, args...) = getindex(ps.samples, args...)


"""
save parameterspace
"""
function h5save(fpath::AbstractString, ps::ParameterSpace)
    h5open(fpath, "r+") do file
        g = create_group(file, "parameterspace")
        cols = columns(ps.samples)
        for key in keys(cols)
            g[string(key)] = cols[key]
        end
    end
end


function h5load(fpath::AbstractString, ::Type{ParameterSpace})
    h5open(fpath, "r") do file
        # g = create_group(file, "parameterspace")
        # for (key,value) in columns(ps.samples)
        #     g[key] = value
        # end
    end
end


