
include("config.jl")

mutable struct ParallelDict{K,V} <: AbstractDict{K,V}
    subdicts::Vector{Dict{K,V}}
    locks::Vector{ReentrantLock}

    function ParallelDict{K,V}(num_subdicts::Int) where {V} where {K}
        subdicts = Vector{Dict{K,V}}(undef, num_subdicts)
        locks = Vector{ReentrantLock}(undef, num_subdicts)
        for subdict_index = 1:num_subdicts
            subdicts[subdict_index] = Dict{K,V}()
            locks[subdict_index] = ReentrantLock()
        end
        new(subdicts, locks)
    end
end

ParallelDict{K,V}() where {V} where {K} =
    ParallelDict{K,V}(USE_PARALLEL_HASH_MAP ? Threads.nthreads() * 2 : 1)

subdict_index(key, num_subdicts::Int) = hash((key, nothing)) % num_subdicts + 1

function Base.haskey(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        haskey(dict.subdicts[i], key)
    end
end

function Base.get(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        get(dict.subdicts[i], key)
    end
end

function Base.getindex(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        getindex(dict.subdicts[i], key)
    end
end

function Base.get!(dict::ParallelDict, key, default)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        get!(dict.subdicts[i], key, default)
    end
end

function Base.get!(f::Function, dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        get!(f, dict.subdicts[i], key)
    end
end

function Base.setindex!(dict::ParallelDict, value, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        setindex!(dict.subdicts[i], value, key)
    end
end

function Base.getkey(dict::ParallelDict, key, default)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        getkey(dict.subdicts[i], key, default)
    end
end

function Base.delete!(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        delete!(dict.subdicts[i], key)
    end
end

function Base.pop!(dict::ParallelDict, key)
    i = subdict_index(key, length(dict.subdicts))
    lock(dict.locks[i]) do
        pop!(dict.subdicts[i], key)
    end
end

function Base.length(dict::ParallelDict)
    len = 0
    for i in eachindex(dict.subdicts)
        len += lock(dict.locks[i]) do
            length(dict.subdicts[i])
        end
    end
    len
end

# ITERATION IS NOT THREAD SAFE!
function Base.iterate(dict::ParallelDict, state = ())
    if state !== ()
        y = iterate(Base.tail(state)...)
        y !== nothing && return (y[1], (state[1], state[2], y[2]))
    end
    x = (state === () ? iterate(dict.subdicts) : iterate(dict.subdicts, state[1]))
    x === nothing && return nothing
    y = iterate(x[1])
    while y === nothing
        x = iterate(dict.subdicts, x[2])
        x === nothing && return nothing
        y = iterate(x[1])
    end
    return y[1], (x[2], x[1], y[2])
end

Base.IteratorSize(dict::ParallelDict) = Base.HasLength()
Base.IteratorEltype(dict::ParallelDict) = Base.IteratorEltype(dict.subdicts[1])
Base.eltype(dict::ParallelDict) = Base.eltype(dict.subdicts[1])
