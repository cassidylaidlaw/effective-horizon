
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
        "--exploration_policy"
        help = "use an exploration policy other than the random one"
        arg_type = String
        required = false
    end
    args = parse_args(arg_parse_settings)

    if !isnothing(args["out"])
        out_fname = args["out"]
    else
        out_fname = splitext(args["mdp"])[1] * "_analyzed.json"
    end

    # Load MDP into transitions and rewards matrices.
    println("Loading MDP from $(args["mdp"])...")
    (transitions, rewards) = load_mdp(args["mdp"])
    horizon = args["horizon"]

    # Load exploration policy if used.
    exploration_policy = nothing
    if args["exploration_policy"] !== nothing
        println("Loading exploration policy from $(args["exploration_policy"])...")
        exploration_policy = npzread(args["exploration_policy"])
    end

    out_dict = Dict{String,AbstractArray}()

    # Distributional value iteration: we calculate the distribution of returns under
    # the random policy at each state.
    if !isfile(splitext(out_fname)[1] * "_value_dists_1.npy")
        println("Running distributional value iteration...")
        for (timestep, value_dist) in zip(
            horizon:-1:1,
            Channel(
                (channel) -> distributional_value_iteration(
                    transitions,
                    rewards,
                    horizon,
                    channel;
                    exploration_policy = exploration_policy,
                ),
            ),
        )
            value_dist_fname = splitext(out_fname)[1] * "_value_dists_$(timestep).npy"
            println("Saving $(value_dist_fname)...")
            open(value_dist_fname, "w") do value_dist_file
                NPZ.npzwritearray(value_dist_file, value_dist)
            end
        end
    end

    results = Dict()

    if isfile(out_fname)
        # Read any existing results so we don't have to recalculate them.
        open(out_fname, "r") do out_file
            global results
            results = JSON.parse(read(out_file, String))
        end
        println("Existing results: $(JSON.json(results))")
    end

    if !haskey(results, "min_k")
        # Calculate minimum value of k such that the assumption holds.
        println("Calculating minimum k...")
        results["min_k"] = calculate_minimum_k(
            transitions,
            rewards,
            horizon;
            exploration_policy = exploration_policy,
        )
    end

    if !(
        haskey(results, "effective_horizon") &&
        haskey(results, "effective_horizon_results")
    )
        # Calculate simple bounds on effective horizon.
        println("Calculating effective horizon...")
        effective_horizon_results = compute_simple_effective_horizon(
            transitions,
            rewards,
            horizon;
            exploration_policy = exploration_policy,
        )
        results["effective_horizon"] = effective_horizon_results.effective_horizon
        results["effective_horizon_results"] = to_dict(effective_horizon_results)
    end

    if !haskey(results, "epw")
        # Calculate effective planning window (EPW).
        println("Calculating EPW...")
        results["epw"] = calculate_minimum_k(
            transitions,
            rewards,
            horizon;
            exploration_policy = exploration_policy,
            start_with_rewards = true,
        )
    end

    if !(haskey(results, "min_occupancy_results"))
        println("Calculating minimum state occupancy measure...")
        min_occupancy_results = calculate_min_occupancy(
            transitions,
            horizon;
            exploration_policy = exploration_policy,
        )
        results["min_occupancy_results"] = to_dict(min_occupancy_results)
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
