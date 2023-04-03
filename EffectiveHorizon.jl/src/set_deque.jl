using Random

include("config.jl")

"""
A deque that acts like a set in that there can only be one of each object in the
queue at a time.
"""
struct SetDeque{T}
    queue::Deque{T}
    set::Set{T}

    SetDeque{T}() where {T} = new{T}(Deque{T}(), Set{T}())
end

Base.isempty(queue::SetDeque) = isempty(queue.queue)
Base.length(queue::SetDeque) = length(queue.queue)

function Base.push!(queue::SetDeque{T}, x::T) where {T}
    if !(x in queue.set)
        push!(queue.queue, x)
        push!(queue.set, x)
    end
end

function Base.pushfirst!(queue::SetDeque{T}, x::T) where {T}
    if !(x in queue.set)
        pushfirst!(queue.queue, x)
        push!(queue.set, x)
    end
end

function Base.pop!(queue::SetDeque{T})::T where {T}
    x = pop!(queue.queue)
    delete!(queue.set, x)
    x
end

function Base.popfirst!(queue::SetDeque{T})::T where {T}
    x = popfirst!(queue.queue)
    delete!(queue.set, x)
    x
end

Base.first(queue::SetDeque) = first(queue.queue)
Base.last(queue::SetDeque) = last(queue.queue)

struct ParallelSetDeque{T}
    subqueues::Vector{SetDeque{T}}
    locks::Vector{ReentrantLock}

    function ParallelSetDeque{T}(num_subqueues::Int) where {T}
        subqueues = Vector{SetDeque{T}}(undef, num_subqueues)
        locks = Vector{ReentrantLock}(undef, num_subqueues)
        for subqueue_index = 1:num_subqueues
            subqueues[subqueue_index] = SetDeque{T}()
            locks[subqueue_index] = ReentrantLock()
        end
        new(subqueues, locks)
    end
end
ParallelSetDeque{T}() where {T} =
    ParallelSetDeque{T}(USE_PARALLEL_HASH_MAP ? Threads.nthreads() * 2 : 1)

subqueue_index(key, num_subqueues::Int) = hash((key, nothing)) % num_subqueues + 1

function Base.isempty(queue::ParallelSetDeque)
    for i in randperm(length(queue.subqueues))
        subqueue_empty = lock(queue.locks[i]) do
            isempty(queue.subqueues[i])
        end
        if !subqueue_empty
            return false
        end
    end
    return true
end

function Base.length(queue::ParallelSetDeque)
    len = 0
    for i in randperm(length(queue.subqueues))
        len += lock(queue.locks[i]) do
            length(queue.subqueues[i])
        end
    end
    return len
end

function Base.push!(queue::ParallelSetDeque, x)
    i = subqueue_index(x, length(queue.subqueues))
    lock(queue.locks[i]) do
        push!(queue.subqueues[i], x)
    end
end


function Base.pushfirst!(queue::ParallelSetDeque, x)
    i = subqueue_index(x, length(queue.subqueues))
    lock(queue.locks[i]) do
        pushfirst!(queue.subqueues[i], x)
    end
end

function Base.pop!(queue::ParallelSetDeque)
    for i in randperm(length(queue.subqueues))
        element = lock(queue.locks[i]) do
            isempty(queue.subqueues[i]) ? nothing : pop!(queue.subqueues[i])
        end
        if element !== nothing
            return element
        end
    end
    throw(ArgumentError("Queue is empty!"))
end

function Base.popfirst!(queue::ParallelSetDeque)
    for i in randperm(length(queue.subqueues))
        element = lock(queue.locks[i]) do
            isempty(queue.subqueues[i]) ? nothing : popfirst!(queue.subqueues[i])
        end
        if element !== nothing
            return element
        end
    end
    throw(ArgumentError("Queue is empty!"))
end
