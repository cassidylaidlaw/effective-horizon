

struct MinOccupancyResults
    min_state_occupancy::Float64
    min_sum_state_occupancy::Float64
    min_state_action_occupancy::Float64
    min_sum_state_action_occupancy::Float64
end


function calculate_min_occupancy(
    transitions::Matrix{Int},
    horizon::Int;
    exploration_policy::Union{Nothing,Array{Float32,3}} = nothing,
)::MinOccupancyResults
    num_states, num_actions = size(transitions)

    state_occupancies = zeros(Float64, horizon, num_states)
    state_action_occupancies = zeros(Float64, horizon, num_states, num_actions)

    state_occupancies[1, 1] = 1

    locks = Vector{ReentrantLock}(undef, Threads.nthreads() * 4)
    for lock_index = 1:length(locks)
        locks[lock_index] = ReentrantLock()
    end

    for timestep in ProgressBar(1:horizon)
        Threads.@threads for state = 1:num_states
            state_occupancy = state_occupancies[timestep, state]
            for action = 1:num_actions
                if exploration_policy === nothing
                    action_prob = 1 / num_actions
                else
                    action_prob = exploration_policy[timestep, state, action]
                end
                state_action_occupancy = action_prob * state_occupancy
                if state_occupancy > 0 && state_action_occupancy == 0
                    println("Warning: underflow")
                end
                state_action_occupancies[timestep, state, action] =
                    state_action_occupancy

                if timestep < horizon
                    next_state = transitions[state, action] + 1
                    lock_index = next_state % length(locks) + 1
                    lock(locks[lock_index]) do
                        state_occupancies[timestep+1, next_state] +=
                            state_action_occupancies[timestep, state, action]
                    end
                end
            end
        end
    end

    max_state_occupancies = maximum(state_occupancies, dims = 1)
    sum_state_occupancies = sum(state_occupancies, dims = 1)
    max_state_action_occupancies = maximum(state_action_occupancies, dims = 1)
    sum_state_action_occupancies = sum(state_action_occupancies, dims = 1)

    # Ignore states or state-action pairs with zero occupancy measure.
    for state = 1:num_states
        if max_state_occupancies[1, state] == 0
            max_state_occupancies[1, state] = 1
        end
        if sum_state_occupancies[1, state] == 0
            sum_state_occupancies[1, state] = 1
        end
        for action = 1:num_actions
            if max_state_action_occupancies[1, state, action] == 0
                max_state_action_occupancies[1, state, action] = 1
            end
            if sum_state_action_occupancies[1, state, action] == 0
                sum_state_action_occupancies[1, state, action] = 1
            end
        end
    end

    results = MinOccupancyResults(
        minimum(@view max_state_occupancies[1, 1:num_states-1]),
        minimum(@view sum_state_occupancies[1, 1:num_states-1]),
        minimum(@view max_state_action_occupancies[1, 1:num_states-1, :]),
        minimum(@view sum_state_action_occupancies[1, 1:num_states-1, :]),
    )
    if !(
        results.min_state_occupancy > 0 &&
        results.min_sum_state_occupancy > 0 &&
        results.min_state_action_occupancy > 0 &&
        results.min_sum_state_action_occupancy > 0
    )
        println("Warning: some occupancy measures are zero")
    end

    return results
end
