
using JSON
using PyCall

Base.@kwdef struct ProcgenMDPConfig <: MDPConfig
    horizon::Int
    log_dir::Union{String,Nothing} = nothing
    done_on_reward::Bool = false
    no_done_reward::Reward = 0
    noops_after_horizon::Int = 0
    frameskip::Int = 5
    num_workers::Int = Threads.nthreads()

    env_name::String
    level::Int = 0
    distribution_mode::String
end

include("libenv.jl")
libenv = ""

mutable struct ProcgenEnv <: Env
    lib_path::String
    config::ProcgenMDPConfig
    options::libenv_options
    keep_alive::Vector
    env::Ptr{libenv_env}
    ob_buffer::Vector{UInt8}
    ac_buffer::Vector{Int32}
    info_buffers::Vector{Vector}
    reward_buffer::Vector{Float32}
    first_buffer::Vector{UInt8}
    buffers::libenv_buffers
    done::Bool

    function ProcgenEnv(config::ProcgenMDPConfig)
        env = new()
        env.config = config
        env.keep_alive = Vector{Any}(undef, 0)

        py_env = pyimport("procgen").ProcgenGym3Env(1, config.env_name)
        global libenv = py_env._lib_path

        level = config.level
        distribution_mode_str = config.distribution_mode
        if distribution_mode_str == "exploration"
            level = pyimport("procgen.env").EXPLORATION_LEVEL_SEEDS[config.env_name]
            distribution_mode_str = "hard"
        end
        distribution_mode =
            pyimport("procgen.env").DISTRIBUTION_MODE_DICT[distribution_mode_str]

        option_array = [
            make_libenv_option!("env_name", config.env_name, env.keep_alive),
            make_libenv_option!("num_levels", 1, env.keep_alive),
            make_libenv_option!("start_level", level, env.keep_alive),
            make_libenv_option!("num_actions", 15, env.keep_alive),
            make_libenv_option!("rand_seed", 0, env.keep_alive),
            make_libenv_option!("distribution_mode", distribution_mode, env.keep_alive),
            make_libenv_option!("render_human", UInt8(false), env.keep_alive),
            make_libenv_option!("center_agent", UInt8(true), env.keep_alive),
            make_libenv_option!("use_generated_assets", UInt8(false), env.keep_alive),
            make_libenv_option!("use_monochrome_assets", UInt8(false), env.keep_alive),
            make_libenv_option!("restrict_themes", UInt8(false), env.keep_alive),
            make_libenv_option!("use_backgrounds", UInt8(true), env.keep_alive),
            make_libenv_option!("paint_vel_info", UInt8(false), env.keep_alive),
        ]
        push!(env.keep_alive, option_array)

        env.options = libenv_options(pointer(option_array), length(option_array))

        env.env = libenv_make(1, env.options)

        # Create buffers.
        env.ob_buffer =
            Vector{UInt8}(undef, 3 * get_screen_width(env) * get_screen_height(env))
        ob_buffers = [pointer(env.ob_buffer)]
        push!(env.keep_alive, ob_buffers)

        env.ac_buffer = Vector{Int32}(undef, 1)
        ac_buffers = [pointer(env.ac_buffer)]
        push!(env.keep_alive, ac_buffers)

        env.info_buffers =
            [Vector{Int32}(undef, 1), Vector{Int32}(undef, 1), Vector{UInt8}(undef, 1)]
        env.reward_buffer = Vector{Cfloat}(undef, 1)
        env.first_buffer = Vector{UInt8}(undef, 1)

        env.buffers = libenv_buffers(
            pointer(ob_buffers),
            pointer(env.reward_buffer),
            pointer(env.first_buffer),
            pointer(env.info_buffers),
            pointer(ac_buffers),
        )
        buffers_array = [env.buffers]
        push!(env.keep_alive, buffers_array)
        libenv_set_buffers(env.env, pointer(buffers_array))

        env.done = false

        finalizer(env) do env
            libenv_close(env.env)
        end
    end
end

function get_env(config::ProcgenMDPConfig)::Env
    ProcgenEnv(config)
end

function get_actions(env::ProcgenEnv)
    num_actions = Dict(
        "bigfish" => 9,
        "bossfight" => 10,
        "caveflyer" => 10,
        "chaser" => 9,
        "climber" => 9,
        "coinrun" => 9,
        "dodgeball" => 10,
        "fruitbot" => 9,
        "heist" => 9,
        "jumper" => 9,
        "leaper" => 9,
        "maze" => 9,
        "miner" => 9,
        "ninja" => 13,
        "plunder" => 10,
        "starpilot" => 11,
    )[env.config.env_name]
    0:num_actions-1
end

function get_state(env::ProcgenEnv)::Vector{UInt8}
    MAX_STATE_SIZE = 2^20
    buffer = Vector{UInt8}(undef, MAX_STATE_SIZE)
    state_len = libenv_get_state(env.env, 0, buffer, length(buffer))
    buffer[1:state_len]
end

function get_state(env::ProcgenEnv, buffer::Vector{UInt8})::Int
    libenv_get_state(env.env, 0, buffer, length(buffer))
end

function get_screen(env::ProcgenEnv)::Vector{UInt8}
    Vector{UInt8}(env.ob_buffer)
end

function get_screen(env::ProcgenEnv, buffer::Vector{UInt8})::Int
    buffer[1:length(env.ob_buffer)] .= env.ob_buffer
    length(env.ob_buffer)
end

function get_screen_width(env::ProcgenEnv)::Int
    64
end

function get_screen_height(env::ProcgenEnv)::Int
    64
end

function set_state!(env::ProcgenEnv, state::Vector{UInt8}, state_len::Integer)
    libenv_set_state(env.env, 0, state, state_len)
    env.done = false
end

function set_state!(
    env::ProcgenEnv,
    state::SubArray{UInt8,1,Vector{UInt8},Tuple{UnitRange{UInt64}},true},
)
    set_state!(env, state.parent, length(state))
end

function is_done(env::ProcgenEnv)::Bool
    env.done
end

function step(env::ProcgenEnv, action::Int)::Reward
    if env.done
        return 0
    end

    env.ac_buffer[1] = action
    libenv_act(env.env)
    libenv_observe(env.env)
    if env.first_buffer[1] != 0
        env.done = true
    end
    env.reward_buffer[1]
end
