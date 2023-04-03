
using ProgressBars
using DataStructures


struct ValueIterationResults
    exploration_qs::Array{Float64,3}
    exploration_values::Array{Float64,2}
    optimal_qs::Array{Float64,3}
    optimal_values::Array{Float64,2}
    worst_qs::Array{Float64,3}
    worst_values::Array{Float64,2}
end

function value_iteration(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
)
    num_states, num_actions = size(transitions)

    exploration_qs = zeros(Float64, horizon, num_states, num_actions)
    exploration_values = zeros(Float64, horizon + 1, num_states)
    optimal_qs = zeros(Float64, horizon, num_states, num_actions)
    optimal_values = zeros(Float64, horizon + 1, num_states)
    worst_qs = zeros(Float64, horizon, num_states, num_actions)
    worst_values = zeros(Float64, horizon + 1, num_states)

    for timestep in ProgressBar(horizon:-1:1)
        Threads.@threads for state = 1:num_states
            for action = 1:num_actions
                next_state = transitions[state, action] + 1
                reward = rewards[state, action]
                exploration_qs[timestep, state, action] =
                    reward + exploration_values[timestep+1, next_state]
                optimal_qs[timestep, state, action] =
                    reward + optimal_values[timestep+1, next_state]
                worst_qs[timestep, state, action] =
                    reward + worst_values[timestep+1, next_state]
            end
            optimal_value = maximum(optimal_qs[timestep, state, :])
            worst_value = minimum(worst_qs[timestep, state, :])
            if exploration_policy === nothing
                exploration_value =
                    sum(exploration_qs[timestep, state, :]) / num_actions
            else
                exploration_value = 0
                for action = 1:num_actions
                    exploration_value +=
                        exploration_qs[timestep, state, action] *
                        exploration_policy[timestep, state, action]
                end
            end
            optimal_values[timestep, state] = optimal_value
            worst_values[timestep, state] = worst_value
            # Make sure exploration value is between worst and optimal values since
            # occasionally floating point error leads to that not being true.
            exploration_values[timestep, state] =
                min(optimal_value, max(worst_value, exploration_value))
        end
    end

    return ValueIterationResults(
        exploration_qs,
        exploration_values[1:horizon, :],
        optimal_qs,
        optimal_values[1:horizon, :],
        worst_qs,
        worst_values[1:horizon, :],
    )
end

function calculate_minimum_k(
    transitions::Matrix{Int},
    rewards::Matrix{Float32},
    horizon::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
    start_with_rewards::Bool = false,
)
    num_states, num_actions = size(transitions)
    vi = value_iteration(
        transitions,
        rewards,
        horizon,
        exploration_policy = exploration_policy,
    )
    if start_with_rewards
        current_qs = Array{Float64}(undef, horizon, num_states, num_actions)
        for timestep = 1:horizon
            current_qs[timestep, :, :] .= rewards
        end
    else
        current_qs = vi.exploration_qs
    end
    states_can_be_visited = zeros(Bool, horizon, num_states)
    k = 1
    while true
        # Check if this value of k works.
        k_works = Threads.Atomic{Bool}(true)
        states_can_be_visited .= false
        states_can_be_visited[1, 1] = true
        timesteps_iter = ProgressBar(1:horizon)
        set_description(timesteps_iter, "Trying k = $(k)")
        for timestep in timesteps_iter
            Threads.@threads for state = 1:num_states
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
                            next_state = transitions[state, action] + 1
                            if timestep < horizon
                                states_can_be_visited[timestep+1, next_state] = true
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
            Threads.@threads for state = 1:num_states
                for action = 1:num_actions
                    next_state = transitions[state, action] + 1
                    max_next_q = -Inf64
                    for action = 1:num_actions
                        next_q = current_qs[timestep+1, next_state, action]
                        max_next_q = max(max_next_q, next_q)
                    end
                    current_qs[timestep, state, action] =
                        rewards[state, action] + max_next_q
                end
            end
        end
        k += 1
    end
end
