
import Base: collect, length, maximum, minimum, size, NamedTuple


AbstractSample{DT} = Union{Nothing, AbstractVector{DT}}

_sort(s::Nothing) = s
_sort(s::AbstractVector) = sort(unique(s))


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

Base.collect(p::Parameter) = p.samples
Base.maximum(p::Parameter) = p.maximum
Base.minimum(p::Parameter) = p.minimum

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
