
using EffectiveHorizon
using ArgParse

if abspath(PROGRAM_FILE) == @__FILE__
    arg_parse_settings = ArgParseSettings()
    @add_arg_table! arg_parse_settings begin
        "--rom", "-r"
        help = "Atari ROM file"
        arg_type = String
        required = false
        "--env_name", "-e"
        help = "ProcGen/MiniGrid env name"
        arg_type = String
        required = false
        "--minigrid"
        help = "use MiniGrid env (as opposed to ProcGen)"
        action = :store_true
        "--out", "-o"
        help = "output directory"
        arg_type = String
        required = true
        "--uncompressed", "-u"
        help = "don't compress the output"
        action = :store_true
        "--save_screens"
        help = "save screen images"
        action = :store_true
        "--save_states"
        help = "save serialized states"
        action = :store_true
        "--horizon"
        help = "maximum episode length"
        arg_type = Int
        default = 5
        "--frameskip"
        help = "ALE frameskip"
        arg_type = Int
        default = 5
        "--done_on_reward"
        help = "end episode on nonzero reward"
        action = :store_true
        "--done_on_life_lost"
        help = "end episode upon losing a life"
        action = :store_true
        "--no_done_reward"
        help = "reward to give if the episode doesn't terminate (generally negative)"
        arg_type = Float32
        default = Float32(0)
        "--noops_after_horizon"
        help = "take this number of additional NOOPs after the horizon has been reached"
        arg_type = Int
        default = 0
        "--level"
        help = "ProcGen level number"
        arg_type = Int
        default = 0
        "--distribution_mode"
        help = "ProcGen level variant (easy, hard, extreme, memory, or exploration)"
        arg_type = String
        default = "easy"
    end
    args = parse_args(arg_parse_settings)

    if !isnothing(args["rom"])
        config = AtariMDPConfig(
            rom_file = args["rom"],
            horizon = args["horizon"],
            done_on_reward = args["done_on_reward"],
            done_on_life_lost = args["done_on_life_lost"],
            no_done_reward = args["no_done_reward"],
            noops_after_horizon = args["noops_after_horizon"],
            frameskip = args["frameskip"],
            log_dir = args["out"],
        )
    elseif !isnothing(args["env_name"]) && args["minigrid"]
        config = MiniGridMDPConfig(
            env_name = args["env_name"],
            horizon = args["horizon"],
            done_on_reward = args["done_on_reward"],
            no_done_reward = args["no_done_reward"],
            noops_after_horizon = args["noops_after_horizon"],
            frameskip = args["frameskip"],
            log_dir = args["out"],
        )
    elseif !isnothing(args["env_name"])
        config = ProcgenMDPConfig(
            env_name = args["env_name"],
            distribution_mode = args["distribution_mode"],
            level = args["level"],
            horizon = args["horizon"],
            done_on_reward = args["done_on_reward"],
            no_done_reward = args["no_done_reward"],
            noops_after_horizon = args["noops_after_horizon"],
            frameskip = args["frameskip"],
            log_dir = args["out"],
        )
    end

    mdp = construct_mdp(config)

    compress = !args["uncompressed"]
    save_screens = args["save_screens"]
    save_states = args["save_states"]

    println("Consolidating (ignoring screen)...")
    consolidated_mdp = MDP(mdp)
    consolidate_completely!(consolidated_mdp; ignore_screen = true)
    save_fname = joinpath(args["out"], "consolidated_ignore_screen")
    println("Saving to $(save_fname)...")
    save(
        consolidated_mdp,
        save_fname;
        compress = compress,
        save_screens = save_screens,
        save_states = save_states,
    )

    println("Consolidating (using screen)...")
    consolidated_mdp = MDP(mdp)
    consolidate_completely!(consolidated_mdp)
    save_fname = joinpath(args["out"], "consolidated")
    println("Saving to $(save_fname)...")
    save(
        consolidated_mdp,
        save_fname;
        compress = compress,
        save_screens = save_screens,
        save_states = save_states,
    )

    save_fname = joinpath(args["out"], "mdp")
    println("Saving to $(save_fname)...")
    save(
        mdp,
        save_fname;
        compress = compress,
        save_screens = save_screens,
        save_states = save_states,
    )
end
