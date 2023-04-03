
using ArgParse
using EffectiveHorizon
using JSON
using NPZ

if abspath(PROGRAM_FILE) == @__FILE__
    arg_parse_settings = ArgParseSettings()
    @add_arg_table! arg_parse_settings begin
        "--mdp", "-m"
        help = "MDP file (in NPZ format)"
        arg_type = String
        required = true
        "--out", "-o"
        help = "output file"
        arg_type = String
        required = false
        "--horizon"
        help = "maximum episode length"
        arg_type = Int
        default = 5
        "--max_k"
        help = "maximum value of k"
        arg_type = Int
        default = -1
        required = false
        "--use_value_dists"
        help = "use Berry-Esseen bounds (requires distributional value iteration)"
        action = :store_true
        "--exploration_policy"
        help = "use an exploration policy other than the random one"
        arg_type = String
        required = false
    end
    args = parse_args(arg_parse_settings)

    # Load MDP into transitions and rewards matrices.
    println("Loading MDP from $(args["mdp"])...")
    (transitions, rewards) = load_mdp(args["mdp"])
    num_states, num_actions = size(transitions)
    horizon = args["horizon"]

    # If max_k is not specified, use the maximum value of k such that A^k ≤ 1000.
    max_k = Int(args["max_k"])
    if max_k == -1
        max_k = Int(floor(log(num_actions, 1000)))
    end

    # Rescale rewards.
    min_positive_reward = 1
    for reward in rewards
        if reward > 0 && reward < min_positive_reward
            global min_positive_reward
            min_positive_reward = reward
        end
    end
    rewards .*= round(1 / min_positive_reward)

    # Load exploration policy if used.
    exploration_policy = nothing
    if args["exploration_policy"] !== nothing
        println("Loading exploration policy from $(args["exploration_policy"])...")
        exploration_policy = npzread(args["exploration_policy"])
    end

    println("Running value iteration...")
    vi = value_iteration(
        transitions,
        rewards,
        horizon;
        exploration_policy = exploration_policy,
    )

    if args["exploration_policy"] !== nothing
        exploration_policy_return = vi.exploration_values[1, 1]
        println(
            "Optimal return = $(vi.optimal_values[1, 1]), " *
            "exploration policy return = $(vi.exploration_values[1, 1])",
        )
    end

    moments = nothing
    if args["use_value_dists"]
        println("Loading value distributions...")
        moments = load_moments(args["mdp"], num_states, horizon)
    end

    mdp = MDPForAnalysis(transitions, rewards, horizon, vi, moments)
    k_results = Dict{Int,Dict{String,Any}}()
    for k = 1:max_k
        println("[k=$(k)] calculating sample complexity")
        m_max::BigInt = BigInt(10)^100
        m_min::BigInt = 0
        failure_prob = compute_failure_prob(mdp, k, m_max)
        println("  failure probability <= $(failure_prob)")
        if failure_prob <= 0.5
            while m_max > m_min + 1 && (m_max - m_min) / m_max > 1e-2
                m_middle::BigInt = floor((m_max * (m_min + 1))^0.5)
                failure_prob = compute_failure_prob(mdp, k, m_middle)
                println("  failure probability <= $(failure_prob)")
                if failure_prob <= 0.5
                    m_max = m_middle
                else
                    m_min = m_middle
                end
            end
            m = m_max
            sample_complexity = (num_actions^k) * m * (horizon^2)
            effective_horizon = k + log(num_actions, m)
        else
            effective_horizon = Inf
            sample_complexity = Inf
        end
        k_results[k] = Dict(
            "effective_horizon" => effective_horizon,
            "sample_complexity" => sample_complexity,
        )
    end

    results = Dict(
        "k_results" => k_results,
        "effective_horizon" => minimum(
            k_result["effective_horizon"] for k_result in values(k_results)
        ),
        "sample_complexity" => minimum(
            k_result["sample_complexity"] for k_result in values(k_results)
        ),
    )

    json_results = JSON.json(results)
    println("Results: $(json_results)")

    # Save output.
    if !isnothing(args["out"])
        out_fname = args["out"]
    else
        out_fname = splitext(args["mdp"])[1] * "_gorp_bounds.json"
    end
    println("Saving results to $(out_fname)...")
    open(out_fname, "w") do out_file
        write(out_file, json_results)
    end
end
