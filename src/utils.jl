
function key_index(nt::NamedTuple, k::Symbol)
    @assert haskey(nt, k)
    ntkeys = keys(nt)
    for i in eachindex(ntkeys)
        if ntkeys[i] == k
            return i
        end
    end
end
