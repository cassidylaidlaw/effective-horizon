
using ArgParse
using EffectiveHorizon
using JSON
using NPZ


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
        help = "output directory"
        arg_type = String
        required = false
        "--horizon"
        help = "maximum episode length"
        arg_type = Int
        default = 5
        "--repeat_action_probability"
        help = "probability of repeating the previous action"
        arg_type = Float32
        default = Float32(0.25)
        "--max_k"
        help = "maximum value of k to calculate greedy return for"
        arg_type = Int
        default = 5
    end
    args = parse_args(arg_parse_settings)
    horizon = args["horizon"]
    repeat_action_probability = args["repeat_action_probability"]

    if !isnothing(args["out"])
        out_fname = args["out"]
    else
        out_fname =
            splitext(args["mdp"])[1] *
            "_analyzed_sticky_$(repeat_action_probability).json"
    end

    # Load MDP into transitions and rewards matrices.
    println("Loading MDP from $(args["mdp"])...")
    (transitions, rewards) = load_mdp(args["mdp"])

    results = Dict()

    if isfile(out_fname)
        # Read any existing results so we don't have to recalculate them.
        open(out_fname, "r") do out_file
            global results
            results = JSON.parse(read(out_file, String))
        end
        println("Existing results: $(JSON.json(results))")
    end

    if !haskey(results, "optimal_return") ||
       !haskey(results, "random_return") ||
       !haskey(results, "worst_return") ||
       !haskey(results, "num_states")
        vi = value_iteration(
            transitions,
            rewards,
            horizon;
            repeat_action_probability = repeat_action_probability,
        )
        results["optimal_return"] = vi.optimal_values[1, 1]
        results["random_return"] = vi.exploration_values[1, 1]
        results["worst_return"] = vi.worst_values[1, 1]

        # Calculate number of states.
        println("Calculating number of states...")
        states = Set{Int}()
        for timestep in 1:horizon
            union!(states, vi.visitable_states[timestep])
        end
        results["num_states"] = length(states)
    end

    if !haskey(results, "min_k")
        # Calculate minimum value of k such that the assumption holds.
        println("Calculating minimum k...")
        results["min_k"] = calculate_minimum_k(
            transitions,
            rewards,
            horizon;
            repeat_action_probability = repeat_action_probability,
        )
    end

    if !haskey(results, "epw")
        # Calculate effective planning window (EPW).
        println("Calculating EPW...")
        results["epw"] = calculate_minimum_k(
            transitions,
            rewards,
            horizon;
            repeat_action_probability = repeat_action_probability,
            start_with_rewards = true,
        )
    end

    if !haskey(results, "greedy_returns") ||
       length(results["greedy_returns"]) < args["max_k"]
        # Calculate greedy returns.
        println("Calculating greedy returns...")
        results["greedy_returns"] = calculate_greedy_returns(
            transitions,
            rewards,
            horizon,
            args["max_k"];
            repeat_action_probability = repeat_action_probability,
        )
    end

    # Save output.
    json_results = JSON.json(results)
    println("Results: $(json_results)")

    # Save output.
    println("Saving results to $(out_fname)...")
    open(out_fname, "w") do out_file
        write(out_file, json_results)
    end
end
