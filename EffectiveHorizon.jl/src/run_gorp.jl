
using ArgParse
using EffectiveHorizon
using JSON
using Statistics
using NPZ


mutable struct GORPResults
    actions::Vector{Int}
    final_return::Reward
    total_timesteps::Int
    mean_rewards::Vector{Reward}
end

mutable struct MultipleGORPResults
    actions::Matrix{Int}
    mean_rewards::Matrix{Reward}
    final_returns::Vector{Reward}
    sample_complexities::Vector{Float32}
    median_final_return::Reward
    median_sample_complexity::Float32
    timesteps_per_iteration::Int
end

MultipleGORPResults(
    actions,
    mean_rewards,
    final_returns,
    sample_complexities,
    timesteps_per_iteration,
) = MultipleGORPResults(
    actions,
    mean_rewards,
    final_returns,
    sample_complexities,
    median(final_returns),
    median(sample_complexities),
    timesteps_per_iteration,
)


struct ActionSequencePolicyArray <: AbstractArray{Float32,3}
    num_states::Int
    num_actions::Int
    actions::Vector{Int}
end

Base.size(arr::ActionSequencePolicyArray) =
    (length(arr.actions), num_states, num_actions)

function Base.getindex(arr::ActionSequencePolicyArray, timestep, state, action)
    @boundscheck checkbounds(arr, timestep, state, action)
    if action == arr.actions[timestep]
        return 1
    else
        return 0
    end
end


function action_seq_to_index(action_seq::Vector{Int}, num_actions::Int)
    # Calculate as zero-indexed, then convert.
    index = 0
    for action in action_seq
        index = index * num_actions + (action - 1)
    end
    index + 1
end

function index_to_action_seq(index::Int, num_actions::Int, k::Int)
    index = index - 1
    action_seq = Vector{Int}(undef, k)
    for t = k:-1:1
        action_seq[t] = (index % num_actions) + 1
        index = index ÷ num_actions
    end
    action_seq
end


function run_gorp(
    transitions,
    rewards,
    horizon,
    exploration_policy::Union{Nothing,Array{Float32,3}},
    variant,
    episodes_per_iteration,
    k;
    repeat_action_probability = 0,
)::GORPResults
    num_states, num_actions = size(transitions)

    if variant == :max
        @assert k == 1
    end

    actions = Vector{Int}(undef, horizon)
    mean_rewards = Vector{Reward}(undef, horizon)
    total_timesteps = 0

    for iteration = 1:horizon
        action_seq_returns =
            Matrix{Reward}(undef, num_actions^k, episodes_per_iteration)
        for action_seq_index = 1:num_actions^k
            action_seq = index_to_action_seq(action_seq_index, num_actions, k)
            for episode_index = 1:episodes_per_iteration
                state = 1
                prev_action::Union{Nothing,Int} = nothing
                t = 1
                episode_return::Reward = 0
                while state != num_states && t <= horizon
                    if t < iteration
                        action = actions[t]
                    elseif t < iteration + k
                        action = action_seq[t-iteration+1]
                    else
                        if exploration_policy === nothing
                            action = rand(1:num_actions)
                        else
                            action = num_actions
                            z = rand(Float32)
                            p_sum = 0
                            for dist_action = 1:num_actions
                                p_sum += exploration_policy[t, state, dist_action]
                                if z < p_sum
                                    action = dist_action
                                    break
                                end
                            end
                            if action == num_actions
                                @assert p_sum >= 1 - 1e-4
                            end
                        end
                    end

                    if prev_action !== nothing
                        if rand(Float32) < repeat_action_probability
                            action = prev_action
                        end
                    end
                    episode_return += rewards[state, action]
                    state = transitions[state, action] + 1
                    t += 1
                    total_timesteps += 1
                    prev_action = action
                end
                action_seq_returns[action_seq_index, episode_index] = episode_return
            end
        end

        if variant == :mean
            action_seq_qs = mean(action_seq_returns, dims = 2)[:, 1]
        elseif variant == :max
            action_seq_qs = maximum(action_seq_returns, dims = 2)[:, 1]
        end

        _, best_action_seq_index = findmax(action_seq_qs)
        best_action_seq = index_to_action_seq(best_action_seq_index, num_actions, k)
        actions[iteration] = best_action_seq[1]
        mean_rewards[iteration] = mean(action_seq_returns)
    end

    # Calculate final return.
    if repeat_action_probability == 0
        state = 1
        final_return::Reward = 0
        for t = 1:horizon
            action = actions[t]
            final_return += rewards[state, action]
            state = transitions[state, action] + 1
        end
    else
        final_policy = ActionSequencePolicyArray(num_states, num_actions, actions)
        vi = value_iteration(
            transitions,
            rewards,
            horizon;
            exploration_policy = final_policy,
            show_progress = false,
            repeat_action_probability = repeat_action_probability,
            only_compute_exploration = true,
        )
        final_return = vi.exploration_values[1, 1]
    end

    GORPResults(actions .- 1, final_return, total_timesteps, mean_rewards)
end


function run_gorp_multiple(
    transitions,
    rewards,
    horizon,
    exploration_policy::Union{Nothing,Array{Float32,3}},
    variant,
    episodes_per_iteration,
    k,
    num_runs,
    optimal_return::Reward;
    repeat_action_probability = 0,
)::MultipleGORPResults
    println("Trying n=$(episodes_per_iteration)...")
    num_states, num_actions = size(transitions)
    actions = Matrix{Int}(undef, num_runs, horizon)
    mean_rewards = Matrix{Reward}(undef, num_runs, horizon)
    final_returns = Vector{Reward}(undef, num_runs)
    sample_complexities = Vector{Float32}(undef, num_runs)
    timesteps_per_iteration = episodes_per_iteration * num_actions^k * horizon

    Threads.@threads for run_index = 1:num_runs
        run_results = run_gorp(
            transitions,
            rewards,
            horizon,
            exploration_policy,
            variant,
            episodes_per_iteration,
            k;
            repeat_action_probability = repeat_action_probability,
        )
        actions[run_index, :] .= run_results.actions
        mean_rewards[run_index, :] .= run_results.mean_rewards
        final_returns[run_index] = run_results.final_return
        sample_complexity = run_results.total_timesteps
        if run_results.final_return < optimal_return
            sample_complexity = Inf32
        end
        sample_complexities[run_index] = sample_complexity
    end
    MultipleGORPResults(
        actions,
        mean_rewards,
        final_returns,
        sample_complexities,
        timesteps_per_iteration,
    )
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
        required = true
        "--variant"
        help = "variant of gorp algorithm (mean or max)"
        arg_type = Symbol
        default = :mean
        "--num_runs"
        help = "number of runs of algorithm to calculate median final return"
        arg_type = Int
        default = 25
        "--max_sample_complexity"
        help = "max number of timesteps to run in the environment for a single run"
        arg_type = Int
        default = 5000000
        "--k"
        help = "value of k to use for gorp-mean algorithm"
        arg_type = Int
        default = 1
        "--exploration_policy"
        help = "use an exploration policy other than the random one"
        arg_type = String
        required = false
        "--repeat_action_probability"
        help = "repeat action probability for sticky actions"
        arg_type = Float32
        default = zero(Float32)
    end
    args = parse_args(arg_parse_settings)

    # Load MDP into transitions and rewards matrices.
    println("Loading MDP from $(args["mdp"])...")
    (transitions, rewards) = load_mdp(args["mdp"])
    num_states, num_actions = size(transitions)
    horizon = args["horizon"]
    optimal_return = args["optimal_return"]

    # Load exploration policy if used.
    exploration_policy = nothing
    if args["exploration_policy"] !== nothing
        println("Loading exploration policy from $(args["exploration_policy"])...")
        exploration_policy = npzread(args["exploration_policy"])
    end

    variant = args["variant"]
    num_runs = args["num_runs"]
    max_sample_complexity = args["max_sample_complexity"]
    k = args["k"]
    repeat_action_probability = args["repeat_action_probability"]

    min_n = 1
    max_n = Int(floor(max_sample_complexity / ((horizon^2) * (num_actions^k))))
    @assert max_n >= min_n
    min_results = run_gorp_multiple(
        transitions,
        rewards,
        horizon,
        exploration_policy,
        variant,
        min_n,
        k,
        num_runs,
        optimal_return;
        repeat_action_probability = repeat_action_probability,
    )
    max_results = run_gorp_multiple(
        transitions,
        rewards,
        horizon,
        exploration_policy,
        variant,
        max_n,
        k,
        num_runs,
        optimal_return;
        repeat_action_probability = repeat_action_probability,
    )

    while max_n - min_n > 1
        global max_n
        global min_n
        global max_results
        global min_results

        if max_results.median_sample_complexity == Inf32
            break
        end

        mindpoint_n = (max_n + min_n) ÷ 2
        midpoint_results = run_gorp_multiple(
            transitions,
            rewards,
            horizon,
            exploration_policy,
            variant,
            mindpoint_n,
            k,
            num_runs,
            optimal_return;
            repeat_action_probability = repeat_action_probability,
        )

        if midpoint_results.median_sample_complexity == Inf32
            min_n = mindpoint_n
            min_results = midpoint_results
        else
            max_n = mindpoint_n
            max_results = midpoint_results
        end
    end

    final_results = max_results
    results_dict = to_dict(final_results)

    # Save output.
    out_fname = args["out"]
    println("Saving results to $(out_fname)...")
    open(out_fname, "w") do out_file
        JSON.print(out_file, results_dict)
    end
end
