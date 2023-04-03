
using JSON
using PyCall

Base.@kwdef struct MiniGridMDPConfig <: MDPConfig
    horizon::Int
    log_dir::Union{String,Nothing} = nothing
    done_on_reward::Bool = false
    no_done_reward::Reward = 0
    noops_after_horizon::Int = 0
    frameskip::Int = 1
    num_workers::Int = Threads.nthreads()

    env_name::String
end

mutable struct MiniGridEnv <: Env
    env::Any
    done::Bool
    obs::Vector{UInt8}
    config::MiniGridMDPConfig

    function MiniGridEnv(config::MiniGridMDPConfig)
        env = new()

        # Fix for https://github.com/JuliaPy/PyCall.jl/issues/973
        os = pyimport("os")
        pyimport("sys").setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)

        pyimport("effective_horizon.envs.minigrid")
        ray_registry = pyimport("ray.tune.registry")
        env.env =
            ray_registry._global_registry.get(ray_registry.ENV_CREATOR, config.env_name)(
                Dict(),
            )
        env.obs, info = env.env.reset()

        env.done = false
        env
    end
end

function get_env(config::MiniGridMDPConfig)::Env
    MiniGridEnv(config)
end

function get_actions(env::MiniGridEnv)
    0:env.env.action_space.n-1
end

function get_state(env::MiniGridEnv)::Vector{UInt8}
    Vector{UInt8}(env.env.get_state())
end

function get_screen(env::MiniGridEnv)::Vector{UInt8}
    env.obs
end

function get_screen_width(env::MiniGridEnv)::Int
    ceil(Int, length(env.obs) / 3)
end

function get_screen_height(env::MiniGridEnv)::Int
    1
end

function set_state!(env::MiniGridEnv, state::AbstractVector{UInt8})
    bytes_state = pybytes(Vector{UInt8}(state))
    env.env.set_state(bytes_state)
    env.done = false
end

function is_done(env::MiniGridEnv)::Bool
    env.done
end

function step(env::MiniGridEnv, action::Int)::Reward
    if env.done
        return 0
    end

    env.obs, reward, env.done, truncated, info = env.env.step(action)
    reward
end
