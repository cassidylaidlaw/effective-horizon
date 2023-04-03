
using ArcadeLearningEnvironment


Base.@kwdef struct AtariMDPConfig <: MDPConfig
    horizon::Int
    log_dir::Union{String,Nothing} = nothing
    done_on_reward::Bool = false
    no_done_reward::Reward = 0
    noops_after_horizon::Int = 0
    frameskip::Int = 5
    num_workers::Int = Threads.nthreads()

    rom_file::String
    done_on_life_lost::Bool
end


mutable struct AtariEnv <: Env
    ale::ALEPtr
    lives::Int
    config::AtariMDPConfig
    life_lost::Bool

    function AtariEnv(config::AtariMDPConfig)
        env = new()

        env.ale = ALE_new()
        setLoggerMode!(:error)
        setInt(env.ale, "random_seed", 0)
        setInt(env.ale, "system_random_seed", 4753849)
        setFloat(env.ale, "repeat_action_probability", 0)
        loadROM(env.ale, config.rom_file)
        reset_game(env.ale)

        env.config = config
        env.lives = lives(env.ale)
        env.life_lost = false

        finalizer(env) do env
            ALE_del(env.ale)
        end
    end
end

function get_env(config::AtariMDPConfig)::Env
    AtariEnv(config)
end

function get_actions(env::AtariEnv)
    getMinimalActionSet(env.ale)
end

function get_state(env::AtariEnv)::Vector{UInt8}
    ale_state = cloneSystemState(env.ale)
    state = encodeState(ale_state)
    deleteState(ale_state)
    reinterpret(UInt8, state)
end


function encodeState!(state::ArcadeLearningEnvironment.ALEStatePtr, buf::Array{UInt8})
    ccall(
        (:encodeState, ArcadeLearningEnvironment.libale_c),
        Cvoid,
        (ArcadeLearningEnvironment.ALEStatePtr, Ptr{Cchar}, Cint),
        state,
        buf,
        length(buf),
    )
    buf
end

function get_state(env::AtariEnv, buffer::Vector{UInt8})::Int
    ale_state = cloneSystemState(env.ale)
    encoded_state_len = ArcadeLearningEnvironment.encodeStateLen(ale_state)
    encodeState!(ale_state, buffer)
    deleteState(ale_state)
    encoded_state_len
end

function get_screen(env::AtariEnv)::Vector{UInt8}
    screen_size = 3 * get_screen_width(env) * get_screen_height(env)
    buffer = Vector{UInt8}(undef, screen_size)
    @assert get_screen(env.ale, buffer) == screen_size
    buffer
end

function get_screen(env::AtariEnv, buffer::Vector{UInt8})::Int
    ArcadeLearningEnvironment.getScreenRGB!(env.ale, buffer)
    3 * get_screen_width(env) * get_screen_height(env)
end


function get_screen_width(env::AtariEnv)::Int
    getScreenWidth(env.ale)
end

function get_screen_height(env::AtariEnv)::Int
    getScreenHeight(env.ale)
end

function set_state!(env::AtariEnv, state::Vector{UInt8})
    set_state!(env, state, length(state))
end

function set_state!(
    env::AtariEnv,
    state::SubArray{UInt8,1,Vector{UInt8},Tuple{UnitRange{UInt64}},true},
)
    set_state!(env, state.parent, length(state))
end

function set_state!(env::AtariEnv, state::AbstractVector{UInt8}, state_length::Integer)
    ale_state = ccall(
        (:decodeState, ArcadeLearningEnvironment.libale_c),
        Ptr{Cvoid},
        (Ptr{Cchar}, Cint),
        state,
        state_length,
    )
    restoreSystemState(env.ale, ale_state)
    deleteState(ale_state)

    env.lives = lives(env.ale)
    env.life_lost = false
end

function is_done(env::AtariEnv)::Bool
    game_over(env.ale) || (env.life_lost && env.config.done_on_life_lost)
end

function step(env::AtariEnv, action::Int)::Reward
    reward = act(env.ale, action)
    if lives(env.ale) < env.lives
        env.life_lost = true
    end
    env.lives = lives(env.ale)
    reward
end
