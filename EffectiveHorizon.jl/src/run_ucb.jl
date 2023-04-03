
using ArgParse
using EffectiveHorizon
using JSON
using Statistics
using ProgressBars


mutable struct UCBResults
    actions::Vector{Int}
    final_return::Reward
    total_timesteps::Int
end

mutable struct MultipleUCBResults
    actions::Matrix{Int}
    final_returns::Vector{Reward}
    sample_complexities::Vector{Float32}
    median_final_return::Reward
    median_sample_complexity::Float32
end

MultipleUCBResults(actions, final_returns, sample_complexities) = MultipleUCBResults(
    actions,
    final_returns,
    sample_complexities,
    median(final_returns),
    median(sample_complexities),
)


function get_previous_states(transitions)
    # For each state, get all states that can lead to it.
    num_states, num_actions = size(transitions)
    done_state = num_states
    num_prev_states = zeros(Int32, num_states - 1)
    for state = 1:num_states-1
        for action = 1:num_actions
            next_state = transitions[state, action] + 1
            if next_state != done_state
                num_prev_states[next_state] += 1
            end
        end
    end
    previous_states_buffer = Vector{Int32}(undef, (num_states - 1) * num_actions)
    previous_states_buffer .= -1
    previous_states = Vector{AbstractVector{Int32}}(undef, num_states - 1)
    buffer_position = 1
    for state = 1:num_states-1
        new_buffer_position = buffer_position + num_prev_states[state]
        previous_states[state] =
            @view previous_states_buffer[buffer_position:new_buffer_position-1]
        buffer_position = new_buffer_position
    end
    previous_states_indices = zeros(Int32, num_states - 1)
    for state in ProgressBar(1:num_states-1)
        for action = 1:num_actions
            next_state = transitions[state, action] + 1
            if next_state != done_state
                next_state_previous_states = previous_states[next_state]
                previous_states_indices[next_state] += 1
                previous_states_index = previous_states_indices[next_state]
                @assert next_state_previous_states[previous_states_index] == -1
                next_state_previous_states[previous_states_index] = state
            end
        end
    end
    @assert all(previous_states_indices .== num_prev_states)
    previous_states
end


function run_ucb(
    transitions,
    rewards,
    horizon,
    max_timesteps,
    optimal_return,
    prev_states,
)::UCBResults
    num_states, num_actions = size(transitions)
    done_state = num_states
    max_reward = maximum(rewards)

    total_timesteps = 0

    learned_transitions = Array{Int32}(undef, num_states, num_actions)
    learned_rewards = Array{Reward}(undef, num_states, num_actions)

    # We start with all state-action pairs having max reward and all states absorbing.
    learned_rewards .= max_reward
    for state = 1:num_states
        learned_transitions[state, :] .= state - 1
    end
    # We assume that we already know the done state has 0 reward.
    learned_rewards[num_states, :] .= 0

    # Initial value function is easy to calculate.
    current_vs = Array{Reward}(undef, horizon, num_states)
    current_qs = Array{Reward}(undef, horizon, num_states, num_actions)
    current_vs[:, num_states] .= 0
    current_qs[:, num_states, :] .= 0
    for t = 1:horizon
        current_vs[t, 1:num_states-1] .= (horizon - t + 1) * max_reward
        current_qs[t, 1:num_states-1, :] .= (horizon - t + 1) * max_reward
    end

    while true
        episode_return = 0
        episode_states = Vector{Int32}(undef, horizon)
        episode_actions = Vector{Int32}(undef, horizon)
        episode_len = horizon
        state = 1
        for t = 1:horizon
            # Choose an action based on the current learned Q values.
            num_optimal_actions = 0
            optimal_actions = Vector{Int32}(undef, num_actions)
            optimal_q = maximum(current_qs[t, state, :])
            for action = 1:num_actions
                if current_qs[t, state, action] >= optimal_q - REWARD_PRECISION
                    num_optimal_actions += 1
                    optimal_actions[num_optimal_actions] = action
                end
            end
            action = optimal_actions[rand(1:num_optimal_actions)]

            # Actually interact with the environment to take the action--
            # this should be the only place where we use rewards/transitions.
            reward = rewards[state, action]
            next_state = transitions[state, action] + 1
            total_timesteps += 1

            # Update environment model.
            learned_rewards[state, action] = reward
            learned_transitions[state, action] = next_state - 1

            episode_return += reward
            episode_actions[t] = action
            episode_states[t] = state
            state = next_state

            if state == done_state
                # We've reached a terminal state, so end the episode here.
                episode_len = t
                episode_states = episode_states[1:episode_len]
                episode_actions = episode_actions[1:episode_len]
                break
            end
        end

        # Check if we solved the MDP or reached max sample complexity.
        if episode_return >= optimal_return - REWARD_PRECISION ||
           total_timesteps >= max_timesteps
            return UCBResults(episode_actions, episode_return, total_timesteps)
        end

        # Update Q and V.
        states_to_update = Set{Int32}()
        for t = episode_len:-1:1
            if episode_states[t] != done_state
                push!(states_to_update, episode_states[t])
            end
            prev_states_to_update = Set{Int32}()
            for state in states_to_update
                for action = 1:num_actions
                    current_qs[t, state, action] = learned_rewards[state, action]
                    if t < horizon
                        next_state = learned_transitions[state, action] + 1
                        current_qs[t, state, action] += current_vs[t+1, next_state]
                    end
                end
                old_v = current_vs[t, state]
                new_v = maximum(current_qs[t, state, :])
                if new_v != old_v
                    current_vs[t, state] = new_v
                    for prev_state in prev_states[state]
                        @assert prev_state != -1
                        push!(prev_states_to_update, prev_state)
                    end
                end
            end
            states_to_update = prev_states_to_update
        end
    end
end


function run_ucb_multiple(
    transitions,
    rewards,
    horizon,
    max_timesteps,
    optimal_return,
)::MultipleUCBResults
    println("Calculating previous states...")
    prev_states = get_previous_states(transitions)

    println("Running UCB algorithm...")
    actions = Matrix{Int}(undef, num_runs, horizon)
    actions .= -1
    final_returns = Vector{Reward}(undef, num_runs)
    sample_complexities = Vector{Float32}(undef, num_runs)
    Threads.@threads for run_index in ProgressBar(1:num_runs)
        run_results = run_ucb(
            transitions,
            rewards,
            horizon,
            max_timesteps,
            optimal_return,
            prev_states,
        )
        actions[run_index, 1:length(run_results.actions)] .= run_results.actions
        final_returns[run_index] = run_results.final_return
        sample_complexity = run_results.total_timesteps
        if run_results.final_return < optimal_return - REWARD_PRECISION
            sample_complexity = Inf32
        end
        sample_complexities[run_index] = sample_complexity
    end
    MultipleUCBResults(actions, final_returns, sample_complexities)
end


function to_dict(s)
    return Dict(String(key) => getfield(s, key) for key in propertynames(s))
end


if abspath(PROGRAM_FILE) == @__FILE__
    arg_parse_settings = ArgParseSettings()
    @add_arg_table! arg_parse_settings begin
        "--mdp", "-m"
        help = "MDP file (in NPZ format)"
        arg_type = String
        required = true
        "--out", "-o"
        help = "output filename"
        arg_type = String
        required = true
        "--horizon"
        help = "maximum episode length"
        arg_type = Int
        default = 5
        "--optimal_return"
        help = "optimal return for MDP"
        arg_type = Float32
        "--num_runs"
        help = "number of runs of algorithm to calculate median final return"
        arg_type = Int
        default = 25
        "--max_sample_complexity"
        help = "max number of timesteps to run in the environment for a single run"
        arg_type = Int
        default = 5000000
    end
    args = parse_args(arg_parse_settings)

    # Load MDP into transitions and rewards matrices.
    println("Loading MDP from $(args["mdp"])...")
    (transitions, rewards) = load_mdp(args["mdp"])
    horizon = args["horizon"]
    optimal_return = args["optimal_return"]

    num_runs = args["num_runs"]
    max_sample_complexity = args["max_sample_complexity"]

    results = run_ucb_multiple(
        transitions,
        rewards,
        horizon,
        max_sample_complexity,
        optimal_return,
    )
    results_dict = to_dict(results)

    # Save output.
    out_fname = args["out"]
    println("Saving results to $(out_fname)...")
    open(out_fname, "w") do out_file
        JSON.print(out_file, results_dict)
    end
end
