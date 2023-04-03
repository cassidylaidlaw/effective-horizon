
using ProgressBars
using DataStructures


mutable struct ValueDistribution
    values::Vector{Float64}
    probs::Vector{Float64}
    function ValueDistribution()
        new(zeros(Float64, 100), zeros(Float64, 100))
    end
end

function Base.empty!(dist::ValueDistribution)
    dist.values .= 0
    dist.probs .= 0
end

function add!(dist::ValueDistribution, value::Float64, prob::Float64)
    if prob == 0
        return
    end
    dist_index = 1
    values_length = length(dist.values)
    while dist_index <= values_length
        value_matches =
            @inbounds abs(dist.values[dist_index] - value) < REWARD_PRECISION
        @inbounds if (value_matches || dist.probs[dist_index] == 0)
            @assert dist.probs[dist_index] > 0 || dist.values[dist_index] == 0
            break
        end
        dist_index += 1
    end
    if dist_index > values_length
        resize!(dist, values_length * 2)
    end
    @inbounds dist.values[dist_index] = value
    @inbounds dist.probs[dist_index] += prob
end

function resize!(dist::ValueDistribution, new_size::Int)
    # Double length.
    old_values = dist.values
    old_probs = dist.probs
    old_size = length(old_values)
    @assert new_size >= old_size
    dist.values = zeros(Float64, new_size)
    dist.probs = zeros(Float64, new_size)
    @inbounds dist.values[1:old_size] .= old_values
    @inbounds dist.probs[1:old_size] .= old_probs
end

function Base.length(dist::ValueDistribution)
    dist_index = 1
    while dist_index <= length(dist.values)
        if dist.probs[dist_index] == 0
            break
        end
        dist_index += 1
    end
    dist_index - 1
end

function distributional_value_iteration(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int,
    channel::Channel;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
)
    num_states, num_actions = size(transitions)
    transitions = permutedims(transitions, (2, 1))
    rewards = permutedims(rewards, (2, 1))

    # Value distribution is stored as a matrix of size
    #     num_state x max_num_returns x 2
    # where value_dist[1, :, state] are returns and value_dist[2, :, state] are
    # the corresponding probabilities.
    terminal_value_dists = zeros(Float64, 2, 1, num_states)
    # At the end of the MDP, all states have 0 return with probability 1.
    terminal_value_dists[2, 1, :] .= 1.0
    next_value_dists = terminal_value_dists

    current_dists = Vector{ValueDistribution}(undef, num_states)
    Threads.@threads for state = 1:num_states
        current_dists[state] = ValueDistribution()
    end

    for timestep = horizon:-1:1
        max_num_returns = Threads.Atomic{Int}(1)
        next_max_num_returns = size(next_value_dists)[2]
        Threads.@threads for state = 1:num_states
            dist = @inbounds current_dists[state]
            empty!(dist)
            for action = 1:num_actions
                # Transition matrix is 0-indexed, Julia is 1-indexed.
                next_state = @inbounds transitions[action, state] + 1
                reward = @inbounds rewards[action, state]
                for dist_index = 1:next_max_num_returns
                    next_value = @inbounds next_value_dists[1, dist_index, next_state]
                    prob = @inbounds next_value_dists[2, dist_index, next_state]
                    if prob > 0 && exploration_policy === nothing
                        # Make sure we don't get float underflow.
                        @assert prob / num_actions > 0
                    end
                    if exploration_policy === nothing
                        action_prob = 1 / num_actions
                    else
                        action_prob = exploration_policy[timestep, state, action]
                    end
                    if prob * action_prob > 0
                        add!(dist, next_value + reward, prob * action_prob)
                    end
                end
            end
            Threads.atomic_max!(max_num_returns, length(dist))
        end

        timestep_value_dists = zeros(Float64, 2, max_num_returns[], num_states)
        Threads.@threads for state = 1:num_states
            dist::ValueDistribution = current_dists[state]
            @inbounds for dist_index = 1:length(dist)
                timestep_value_dists[1, dist_index, state] = dist.values[dist_index]
                timestep_value_dists[2, dist_index, state] = dist.probs[dist_index]
            end
        end

        put!(channel, timestep_value_dists)
        next_value_dists = timestep_value_dists
    end
end
