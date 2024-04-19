
using ProgressBars
using DataStructures


mutable struct TimestepStateDictArray{T,N,M} <: AbstractArray{T,N}
    shape::NTuple{N,Int}
    dict::ParallelDict{Tuple{Int,Int},Array{T,M}}
    default_entry::Array{T,M}

    function TimestepStateDictArray{T,N,M}(
        default,
        shape...,
    ) where {T} where {N} where {M}
        @assert M == N - 2
        dict = ParallelDict{Tuple{Int,Int},Array{T,M}}()
        default_entry = fill(default, shape[3:end]...)
        new(shape, dict, default_entry)
    end
end

Base.size(arr::TimestepStateDictArray) = arr.shape

function Base.getindex(arr::TimestepStateDictArray, timestep, state, I...)
    @boundscheck checkbounds(arr, timestep, state, I...)
    key = (timestep, state)
    value = get(arr.dict, key, arr.default_entry)
    getindex(value, I...)
end

function Base.setindex!(arr::TimestepStateDictArray, v, timestep, state, I...)
    @boundscheck checkbounds(arr, timestep, state, I...)
    key = (timestep, state)
    value = get!(arr.dict, key) do
        copy(arr.default_entry)
    end
    setindex!(value, v, I...)
end

Base.maximum(arr::TimestepStateDictArray) = maximum(maximum(v) for (k, v) in arr.dict)


struct ValueIterationResults
    exploration_qs::AbstractArray{Float64,3}
    exploration_values::AbstractArray{Float64,2}
    optimal_qs::AbstractArray{Float64,3}
    optimal_values::AbstractArray{Float64,2}
    worst_qs::AbstractArray{Float64,3}
    worst_values::AbstractArray{Float64,2}
    visitable_states::AbstractVector{AbstractVector{Int}}
end

function sticky_transitions_and_rewards(
    transitions,
    rewards,
    num_actions,
    timestep,
    state,
    action,
    repeat_action_probability,
)::Vector{Tuple{Float32,Int,Reward}}
    if repeat_action_probability == 0
        return [(1, transitions[state, action] + 1, rewards[state, action])]
    else
        underlying_state = (state - 1) ÷ num_actions + 1
        prev_action = (state - 1) % num_actions + 1
        next_state = transitions[underlying_state, action] * num_actions + action
        reward = rewards[underlying_state, action]
        next_state_sticky =
            transitions[underlying_state, prev_action] * num_actions + prev_action
        reward_sticky = rewards[underlying_state, prev_action]
        if timestep == 1
            return [(1, next_state, reward)]
        else
            return [
                (1 - repeat_action_probability, next_state, reward),
                (repeat_action_probability, next_state_sticky, reward_sticky),
            ]
        end
    end
end

function value_iteration(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int;
    exploration_policy::Union{Nothing,AbstractArray{Float32,3}} = nothing,
    show_progress = true,
    repeat_action_probability::Float32 = Float32(0),
    only_compute_exploration = false,
)
    num_states, num_actions = size(transitions)

    if repeat_action_probability > 0
        num_states = num_states * num_actions
    end

    visitable_states = AbstractVector{Int}[]
    current_visitable_states = Set{Int}()
    push!(current_visitable_states, 1)
    timesteps = 1:horizon
    if show_progress
        timesteps = ProgressBar(timesteps)
    end
    for timestep in timesteps
        next_visitable_states = Set{Int}()
        for state in current_visitable_states
            underlying_state =
                repeat_action_probability > 0 ? (state - 1) ÷ num_actions + 1 : state
            for action = 1:num_actions
                if (
                    !only_compute_exploration ||
                    exploration_policy === nothing ||
                    exploration_policy[timestep, underlying_state, action] > 0
                )
                    for (transition_prob, next_state, reward) in
                        sticky_transitions_and_rewards(
                        transitions,
                        rewards,
                        num_actions,
                        timestep,
                        state,
                        action,
                        repeat_action_probability,
                    )
                        push!(next_visitable_states, next_state)
                    end
                end
            end
        end
        push!(visitable_states, collect(current_visitable_states))
        current_visitable_states = next_visitable_states
    end

    exploration_qs =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    exploration_values = TimestepStateDictArray{Float64,2,0}(NaN, horizon, num_states)
    optimal_qs =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    optimal_values = TimestepStateDictArray{Float64,2,0}(NaN, horizon, num_states)
    worst_qs =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    worst_values = TimestepStateDictArray{Float64,2,0}(NaN, horizon, num_states)

    timesteps = horizon:-1:1
    if show_progress
        timesteps = ProgressBar(timesteps)
    end

    for timestep in timesteps
        Threads.@threads for state in visitable_states[timestep]
            underlying_state =
                repeat_action_probability > 0 ? (state - 1) ÷ num_actions + 1 : state
            for action = 1:num_actions
                exploration_qs[timestep, state, action] = 0
                optimal_qs[timestep, state, action] = 0
                worst_qs[timestep, state, action] = 0
                for (transition_prob, next_state, reward) in
                    sticky_transitions_and_rewards(
                    transitions,
                    rewards,
                    num_actions,
                    timestep,
                    state,
                    action,
                    repeat_action_probability,
                )
                    exploration_qs[timestep, state, action] += transition_prob * reward
                    optimal_qs[timestep, state, action] += transition_prob * reward
                    worst_qs[timestep, state, action] += transition_prob * reward
                    if timestep < horizon
                        exploration_qs[timestep, state, action] +=
                            transition_prob * exploration_values[timestep+1, next_state]
                        optimal_qs[timestep, state, action] +=
                            transition_prob * optimal_values[timestep+1, next_state]
                        worst_qs[timestep, state, action] +=
                            transition_prob * worst_values[timestep+1, next_state]
                    end
                end
            end
            optimal_value = maximum(optimal_qs[timestep, state, :])
            worst_value = minimum(worst_qs[timestep, state, :])
            if exploration_policy === nothing
                exploration_value =
                    sum(exploration_qs[timestep, state, :]) / num_actions
            else
                exploration_value = 0
                for action = 1:num_actions
                    if exploration_policy[timestep, underlying_state, action] > 0
                        exploration_value +=
                            exploration_qs[timestep, state, action] *
                            exploration_policy[timestep, underlying_state, action]
                    end
                end
            end
            optimal_values[timestep, state] = optimal_value
            worst_values[timestep, state] = worst_value
            # Make sure exploration value is between worst and optimal values since
            # occasionally floating point error leads to that not being true.
            if !isnan(optimal_value)
                exploration_value = min(optimal_value, exploration_value)
            end
            if !isnan(worst_value)
                exploration_value = max(worst_value, exploration_value)
            end
            exploration_values[timestep, state] = exploration_value
            # Make sure we're not getting any NaNs, which would indicate a bug.
            @assert !isnan(exploration_values[timestep, state])
            if !only_compute_exploration
                @assert !isnan(optimal_values[timestep, state])
                @assert !isnan(worst_values[timestep, state])
            end
        end
    end

    return ValueIterationResults(
        exploration_qs,
        exploration_values,
        optimal_qs,
        optimal_values,
        worst_qs,
        worst_values,
        visitable_states,
    )
end

function calculate_minimum_k(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
    start_with_rewards::Bool = false,
    repeat_action_probability::Float32 = Float32(0),
)
    num_states, num_actions = size(transitions)
    if repeat_action_probability > 0
        num_states = num_states * num_actions
    end

    vi = value_iteration(
        transitions,
        rewards,
        horizon,
        exploration_policy = exploration_policy;
        repeat_action_probability = repeat_action_probability,
    )
    if start_with_rewards
        current_qs =
            TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
        for timestep = 1:horizon
            for state in vi.visitable_states[timestep]
                for action = 1:num_actions
                    current_qs[timestep, state, action] = 0
                    for (transition_prob, next_state, reward) in
                        sticky_transitions_and_rewards(
                        transitions,
                        rewards,
                        num_actions,
                        timestep,
                        state,
                        action,
                        repeat_action_probability,
                    )
                        current_qs[timestep, state, action] += transition_prob * reward
                    end
                end
            end
        end
    else
        current_qs = vi.exploration_qs
    end
    k = 1
    while true
        # Check if this value of k works.
        k_works = Threads.Atomic{Bool}(true)
        states_can_be_visited =
            TimestepStateDictArray{Bool,2,0}(false, horizon, num_states)
        states_can_be_visited[1, 1] = true
        timesteps_iter = ProgressBar(1:horizon)
        set_description(timesteps_iter, "Trying k = $(k)")
        for timestep in timesteps_iter
            Threads.@threads for state in vi.visitable_states[timestep]
                if states_can_be_visited[timestep, state]
                    max_q = -Inf64
                    for action = 1:num_actions
                        max_q = max(max_q, current_qs[timestep, state, action])
                    end
                    for action = 1:num_actions
                        if current_qs[timestep, state, action] >= max_q
                            # If we get here, it's possible to take this action.
                            if (
                                vi.optimal_qs[timestep, state, action] <
                                vi.optimal_values[timestep, state] - REWARD_PRECISION
                            )
                                k_works[] = false
                            end
                            if timestep < horizon
                                for (transition_prob, next_state, reward) in
                                    sticky_transitions_and_rewards(
                                    transitions,
                                    rewards,
                                    num_actions,
                                    timestep,
                                    state,
                                    action,
                                    repeat_action_probability,
                                )
                                    @assert transition_prob > 0
                                    states_can_be_visited[timestep+1, next_state] = true
                                end
                            end
                        end
                    end
                end
            end
            if !k_works[]
                break
            end
        end

        if k_works[]
            return k
        end

        # Run a Bellman backup.
        for timestep in ProgressBar(1:horizon-1)
            Threads.@threads for state in vi.visitable_states[timestep]
                for action = 1:num_actions
                    current_qs[timestep, state, action] = 0
                    for (transition_prob, next_state, reward) in
                        sticky_transitions_and_rewards(
                        transitions,
                        rewards,
                        num_actions,
                        timestep,
                        state,
                        action,
                        repeat_action_probability,
                    )
                        max_next_q = -Inf64
                        for action = 1:num_actions
                            next_q = current_qs[timestep+1, next_state, action]
                            max_next_q = max(max_next_q, next_q)
                        end
                        current_qs[timestep, state, action] +=
                            transition_prob * (reward + max_next_q)
                    end
                end
            end
        end
        k += 1
    end
end


function calculate_greedy_returns(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int,
    max_k::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
    start_with_rewards::Bool = false,
    repeat_action_probability::Float32 = Float32(0),
)
    num_states, num_actions = size(transitions)
    if repeat_action_probability > 0
        num_states = num_states * num_actions
    end

    vi = value_iteration(
        transitions,
        rewards,
        horizon,
        exploration_policy = exploration_policy;
        repeat_action_probability = repeat_action_probability,
    )
    if start_with_rewards
        current_qs =
            TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
        for timestep = 1:horizon
            for state in vi.visitable_states[timestep]
                for action = 1:num_actions
                    current_qs[timestep, state, action] = 0
                    for (transition_prob, next_state, reward) in
                        sticky_transitions_and_rewards(
                        transitions,
                        rewards,
                        num_actions,
                        timestep,
                        state,
                        action,
                        repeat_action_probability,
                    )
                        current_qs[timestep, state, action] += transition_prob * reward
                    end
                end
            end
        end
    else
        current_qs = vi.exploration_qs
    end

    greedy_returns = Array{Float32}(undef, max_k)

    worst_greedy_qs =
        TimestepStateDictArray{Float64,3,1}(NaN, horizon, num_states, num_actions)
    worst_greedy_values = TimestepStateDictArray{Float64,2,0}(NaN, horizon, num_states)

    for k = 1:max_k
        # Check if this value of k works.
        k_works = Threads.Atomic{Bool}(true)
        states_can_be_visited =
            TimestepStateDictArray{Bool,2,0}(false, horizon, num_states)
        states_can_be_visited[1, 1] = true
        timesteps_iter = ProgressBar(horizon:-1:1)
        set_description(timesteps_iter, "Calculating greedy return for k = $(k)")
        for timestep in timesteps_iter
            Threads.@threads for state in vi.visitable_states[timestep]
                max_q = -Inf64
                for action = 1:num_actions
                    max_q = max(max_q, current_qs[timestep, state, action])
                    worst_greedy_qs[timestep, state, action] = 0
                    for (transition_prob, next_state, reward) in
                        sticky_transitions_and_rewards(
                        transitions,
                        rewards,
                        num_actions,
                        timestep,
                        state,
                        action,
                        repeat_action_probability,
                    )
                        worst_greedy_qs[timestep, state, action] +=
                            transition_prob * reward
                        if timestep < horizon
                            worst_greedy_qs[timestep, state, action] +=
                                transition_prob *
                                worst_greedy_values[timestep+1, next_state]
                        end
                    end
                end
                worst_greedy_value = Inf64
                for action = 1:num_actions
                    if current_qs[timestep, state, action] >= max_q
                        # This is a possible greedy action.
                        worst_greedy_value = min(
                            worst_greedy_value,
                            worst_greedy_qs[timestep, state, action],
                        )
                    end
                end
                worst_greedy_values[timestep, state] = worst_greedy_value
                # Make sure we're not getting any NaNs, which would indicate a bug.
                @assert !isnan(worst_greedy_value)
            end
        end

        worst_greedy_return = worst_greedy_values[1, 1]
        println("Greedy return = $worst_greedy_return")
        greedy_returns[k] = worst_greedy_return

        if k == max_k
            break
        end

        # Run a Bellman backup.
        for timestep in ProgressBar(1:horizon-1)
            Threads.@threads for state in vi.visitable_states[timestep]
                for action = 1:num_actions
                    current_qs[timestep, state, action] = 0
                    for (transition_prob, next_state, reward) in
                        sticky_transitions_and_rewards(
                        transitions,
                        rewards,
                        num_actions,
                        timestep,
                        state,
                        action,
                        repeat_action_probability,
                    )
                        max_next_q = -Inf64
                        for action = 1:num_actions
                            next_q = current_qs[timestep+1, next_state, action]
                            max_next_q = max(max_next_q, next_q)
                        end
                        current_qs[timestep, state, action] +=
                            transition_prob * (reward + max_next_q)
                    end
                end
            end
        end
        k += 1
    end

    return greedy_returns
end
