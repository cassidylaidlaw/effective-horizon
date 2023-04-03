module EffectiveHorizon

using DataStructures: Deque
using TensorBoardLogger, Logging
import Base.@kwdef

export TerminalState,
    NonTerminalState,
    State,
    StateInfo,
    MDP,
    num_actions,
    num_states,
    get_state_info,
    AtariMDPConfig,
    ProcgenMDPConfig,
    MiniGridMDPConfig,
    construct_mdp,
    set_unknown_states_terminal!,
    consolidate,
    consolidate_completely,
    consolidate_completely!,
    save,
    load_mdp,
    value_iteration,
    calculate_minimum_k,
    distributional_value_iteration,
    calculate_min_occupancy,
    load_moments,
    is_mdp_binomial,
    MDPForAnalysis,
    compute_failure_prob,
    compute_simple_effective_horizon,
    Reward,
    REWARD_PRECISION


const Reward = Float32
const REWARD_PRECISION = 1e-4

abstract type MDPConfig end

include("env.jl")
include("atari.jl")
include("procgen.jl")
include("minigrid.jl")
include("set_deque.jl")
include("parallel_dict.jl")
include("compression.jl")
include("config.jl")
include("value_iteration.jl")
include("distributional_value_iteration.jl")
include("occupancy_measures.jl")
include("gorp_bounds.jl")


"""
If a compressed serialized state is longer than this length then a new compression
dictionary will be created.
"""
const MAX_COMPRESSED_STATE_LENGTH = 500

ScreenImage = Vector{UInt8}

struct TerminalState end

mutable struct NonTerminalState
    compressed_serialized_state::AbstractVector{UInt8}
    uncompressed_size::UInt
    ddict::ZstdDDict
    compressed_screen_image::ScreenImage
end

"""
Contains preallocated space for various data needed by one exploration worker, to
avoid destroying the heap with repeated allocations.
"""
mutable struct WorkerContext
    compression_context::CompressionContext
    state_encoding_buffer::Vector{UInt8}
    state::NonTerminalState
    function WorkerContext()
        state = NonTerminalState(
            Vector{UInt8}(undef, 0),
            0,
            DUMMY_DDICT,
            Vector{UInt8}(undef, 0),
        )
        new(CompressionContext(), zeros(Int8, MAX_DATA_LEN), state)
    end
end

Base.hash(state::NonTerminalState, h::UInt) =
    hash(state.ddict, hash(state.compressed_serialized_state, h))
# hash(state.compressed_screen_image, h)
Base.:(==)(state_a::NonTerminalState, state_b::NonTerminalState) = (
    state_a.ddict === state_b.ddict &&
    state_a.compressed_serialized_state == state_b.compressed_serialized_state
    # state_a.compressed_screen_image == state_b.compressed_screen_image
)


function bytes_to_bits(bytes::Vector{UInt8})::BitArray{1}
    bits = falses(8 * length(bytes))
    bits_bytes = reinterpret(UInt8, bits.chunks)
    bits_bytes[1:length(bytes)] .= bytes
    return bits
end

function difference(
    state_a::NonTerminalState,
    state_b::NonTerminalState,
    context_a::CompressionContext,
    context_b::CompressionContext;
    compare_bits = false,
)
    differing_indices = Vector{Int}()
    serialized_state_a = decompress(
        state_a.compressed_serialized_state,
        state_a.ddict,
        state_a.uncompressed_size,
        context_a,
    )
    serialized_state_b = decompress(
        state_b.compressed_serialized_state,
        state_b.ddict,
        state_b.uncompressed_size,
        context_b,
    )

    if compare_bits
        serialized_state_a = bytes_to_bits(serialized_state_a)
        serialized_state_b = bytes_to_bits(serialized_state_b)
    end

    for i = 1:max(length(serialized_state_a), length(serialized_state_b))
        if i > length(serialized_state_a) || i > length(serialized_state_b)
            push!(differing_indices, i)
        elseif serialized_state_a[i] != serialized_state_b[i]
            push!(differing_indices, i)
        end
    end

    differing_indices
end

const State = Union{TerminalState,NonTerminalState}

mutable struct StateInfo
    transitions::Vector{Union{State,Nothing}}
    rewards::Vector{Reward}

    "Minimum timestep at which this state can be reached (0 for start state)."
    min_time::Int

    lock::ReentrantLock
end

Base.hash(state_info::StateInfo, h::UInt) =
    Base.hash((state_info.transitions, state_info.rewards))
Base.:(==)(state_info_a::StateInfo, state_info_b::StateInfo) = (
    state_info_a.transitions == state_info_b.transitions &&
    state_info_a.rewards == state_info_b.rewards
)

contains_transition(state_info::StateInfo, action_index::Int)::Bool =
    state_info.transitions[action_index] isa State
function record_transition!(
    state_info::StateInfo,
    action_index::Int,
    next_state::State,
    reward::Reward,
    time::Int,
)
    state_info.transitions[action_index] = next_state
    state_info.rewards[action_index] = reward
    if time < state_info.min_time
        state_info.min_time = time
    end
end
more_to_explore(state_info::StateInfo)::Bool =
    any(next_state isa Nothing for next_state in state_info.transitions)

mutable struct MDP
    states::AbstractDict{NonTerminalState,NonTerminalState}
    screen_images::AbstractDict{ScreenImage,ScreenImage}
    state_infos::AbstractDict{NonTerminalState,StateInfo}
    actions::Vector{Int}
    screen_width::Int
    screen_height::Int
    start_state::NonTerminalState

    "Used by zstd to compress serialized states."
    state_compression_cdicts::Vector{ZstdCDict}
    state_compression_ddicts::Vector{ZstdDDict}
    @atomic num_dicts::Int
    dict_lock::ReentrantLock

    function MDP(env::Env)
        mdp = new()

        # Use start state to make compression dict.
        serialized_start_state = get_state(env)
        mdp.state_compression_cdicts = [ZstdCDict(serialized_start_state)]
        mdp.state_compression_ddicts = [ZstdDDict(serialized_start_state)]
        @atomic mdp.num_dicts = 1
        mdp.dict_lock = ReentrantLock()

        mdp.actions = get_actions(env)
        mdp.screen_width = get_screen_width(env)
        mdp.screen_height = get_screen_height(env)

        mdp.states = ParallelDict{NonTerminalState,NonTerminalState}()
        mdp.screen_images = ParallelDict{ScreenImage,ScreenImage}()
        mdp.start_state = NonTerminalState(env, mdp, WorkerContext())
        mdp.state_infos = ParallelDict{NonTerminalState,StateInfo}()
        mdp.state_infos[mdp.start_state] = StateInfo(mdp, 0)

        mdp
    end

    """
    Clone old MDP to a new one, potentially mapping all states through a function.
    """
    function MDP(old_mdp::MDP; state_mapping = state -> state)
        new_mdp = new()

        new_mdp.state_compression_cdicts = old_mdp.state_compression_cdicts[1:end]
        new_mdp.state_compression_ddicts = old_mdp.state_compression_ddicts[1:end]
        new_mdp.actions = old_mdp.actions[1:end]
        new_mdp.screen_width = old_mdp.screen_width
        new_mdp.screen_height = old_mdp.screen_height
        new_mdp.start_state = state_mapping(old_mdp.start_state)
        new_mdp.screen_images = old_mdp.screen_images

        new_mdp.states = ParallelDict{NonTerminalState,NonTerminalState}()
        new_mdp.state_infos = ParallelDict{NonTerminalState,StateInfo}()

        Threads.@threads for subdict in old_mdp.state_infos.subdicts
            for (old_state, old_state_info) in subdict
                new_state = get_state!(new_mdp, state_mapping(old_state))
                get!(new_mdp.state_infos, new_state) do
                    new_state_info = StateInfo(new_mdp, old_state_info.min_time)
                    new_state_info.rewards = old_state_info.rewards[1:end]
                    for action_index in eachindex(new_mdp.actions)
                        old_next_state = old_state_info.transitions[action_index]
                        if old_next_state === nothing
                            new_next_state = nothing
                        elseif old_next_state isa TerminalState
                            new_next_state = TerminalState()
                        else
                            new_next_state =
                                get_state!(new_mdp, state_mapping(old_next_state))
                        end
                        new_state_info.transitions[action_index] = new_next_state
                    end
                    new_state_info
                end
            end
        end

        new_mdp
    end
end

num_actions(mdp::MDP) = length(mdp.actions)
num_states(mdp::MDP) = length(mdp.state_infos)
"""
Given a state, gets the singleton instance of that state that should be used for all
future tasks. That way, we don't use up extra memory with multiple instances
representing the same state.
"""
get_state!(mdp::MDP, state::NonTerminalState)::NonTerminalState =
    get!(mdp.states, state, state)
has_state(mdp::MDP, state::NonTerminalState) = haskey(mdp.states, state)
"""
Same idea, but for a compressed screen image.
"""
get_screen_image!(mdp::MDP, compressed_screen_image::ScreenImage)::ScreenImage =
    get!(mdp.screen_images, compressed_screen_image, compressed_screen_image)
get_state_info(mdp::MDP, state::NonTerminalState)::StateInfo = mdp.state_infos[state]
function get_state_info!(
    mdp::MDP,
    state::NonTerminalState,
    default_min_time::Int,
)::StateInfo
    get!(mdp.state_infos, state) do
        StateInfo(mdp, default_min_time)
    end
end

function try_compress(encoded_state, mdp::MDP, context::WorkerContext)
    local compressed_serialized_state
    ddict::Union{ZstdDDict,Nothing} = nothing
    for dict_index = 1:mdp.num_dicts
        compressed_serialized_state = compress(
            encoded_state,
            mdp.state_compression_cdicts[dict_index],
            context.compression_context;
            return_view = true,
        )
        if length(compressed_serialized_state) < MAX_COMPRESSED_STATE_LENGTH
            ddict = mdp.state_compression_ddicts[dict_index]
            break
        end
    end
    (compressed_serialized_state, ddict)
end


function NonTerminalState(env::Env, mdp::MDP, context::WorkerContext)
    if AVOID_MALLOC
        encoded_state_len = get_state(env, context.state_encoding_buffer)

        encoded_state = @view context.state_encoding_buffer[1:encoded_state_len]
    else
        encoded_state = get_state(env)
    end

    compressed_serialized_state, ddict = try_compress(encoded_state, mdp, context)
    if ddict === nothing
        compressed_serialized_state, ddict = lock(mdp.dict_lock) do
            # First, check to see if another thread has added a good dictionary in
            # the mean time.
            compressed_serialized_state, ddict =
                try_compress(encoded_state, mdp, context)
            if ddict === nothing
                # No existing dictionary was able to compress this state small enough,
                # so make a new one.
                cdict = ZstdCDict(encoded_state)
                ddict = ZstdDDict(encoded_state)
                push!(mdp.state_compression_cdicts, cdict)
                push!(mdp.state_compression_ddicts, ddict)
                @atomic mdp.num_dicts += 1
                compressed_serialized_state = compress(
                    encoded_state,
                    cdict,
                    context.compression_context;
                    return_view = true,
                )
            end
            (compressed_serialized_state, ddict)
        end
    end

    if AVOID_MALLOC
        # First, make a "fake" state to see if it's already in the state set. This avoids
        # allocating memory if this state already exists.
        # TODO: if we're comparing based on ram or screen, need to add it here
        context.state.compressed_serialized_state = compressed_serialized_state
        context.state.ddict = ddict
        if has_state(mdp, context.state)
            return get_state!(mdp, context.state)
        end

        compressed_serialized_state = Vector{UInt8}(compressed_serialized_state)

        # Compress screen.
        screen_size = 3 * mdp.screen_width * mdp.screen_height
        @assert screen_size <= MAX_DATA_LEN
        screen_size = get_screen(env, context.state_encoding_buffer)
        screen_data = @view context.state_encoding_buffer[1:screen_size]
        compressed_screen_image = compress(screen_data, context.compression_context)
    else
        screen_size = 3 * mdp.screen_width * mdp.screen_height
        screen_data = get_screen(env)
        compressed_screen_image = compress(screen_data, context.compression_context)
    end

    state = NonTerminalState(
        compressed_serialized_state,
        length(encoded_state),
        ddict,
        get_screen_image!(mdp, compressed_screen_image),
    )
    get_state!(mdp, state)  # Convert to singleton version.
end

function StateInfo(mdp::MDP, min_time::Int)
    transitions = Array{Union{State,Nothing}}(undef, num_actions(mdp))
    fill!(transitions, nothing)
    rewards = zeros(Reward, num_actions(mdp))
    StateInfo(transitions, rewards, min_time, ReentrantLock())
end

function log(mdp::MDP, state_queue, logger::TBLogger; force_all = false)
    step = TensorBoardLogger.step(logger)

    print(" "^120 * "\r")
    print("$(length(state_queue)) queued states  \t")
    print("$(num_states(mdp)) total states  \t")
    bytes_per_state = trunc(Int, Base.gc_live_bytes() / num_states(mdp))
    print(
        "$(Base.format_bytes(Base.gc_live_bytes())) memory used ",
        "($(Base.format_bytes(bytes_per_state)) per state)\r",
    )

    # with_logger(logger) do
    #     @info "mdp" num_states = num_states(mdp)
    #     @info "mdp" num_screens = length(mdp.screen_images)
    #     @info "mdp" num_cdicts = length(mdp.state_compression_cdicts)
    #     @info "state_queue" length = length(state_queue) log_step_increment = 0
    #     @info "memory" gc_live_bytes = Base.gc_live_bytes() log_step_increment = 0
    #     if step % 100 == 0 || force_all
    #         @info "memory" rss =
    #             (parse(Int, strip(read(`ps -o rss= $(getpid())`, String))) * 1024) log_step_increment =
    #             0
    #     end
    #     if (
    #         ((round(log2(step)) == log2(step) && step >= 100) || force_all)
    #         # Don't do this past 100000 states because it gets way too slow.
    #         &&
    #         num_states(mdp) < (AVOID_MALLOC ? 100000 : 100)
    #     )
    #         mdp_queue_size = Base.summarysize((mdp, state_queue))
    #         bytes_per_state = trunc(Int, mdp_queue_size / num_states(mdp))
    #         non_mdp_mem = Base.gc_live_bytes() - mdp_queue_size
    #         @info "memory" mdp_queue = mdp_queue_size log_step_increment = 0
    #         @info "memory" per_state = bytes_per_state log_step_increment = 0
    #         @info "memory" other = non_mdp_mem log_step_increment = 0
    #     end
    # end
end

function explore_mdp!(
    config::MDPConfig,
    env::Env,
    mdp::MDP,
    state_queue,
    context::WorkerContext;
    logger::Union{Nothing,TBLogger} = nothing,
)
    while true
        local state::NonTerminalState
        local state_info::StateInfo

        queue_retries = 10
        while queue_retries > 0
            try
                state = popfirst!(state_queue)
                break
            catch error
                if error isa ArgumentError
                    # Queue must be empty.
                    queue_retries -= 1
                    sleep(0.1)
                else
                    rethrow()
                end
            end
        end
        if queue_retries == 0
            break
        end

        state_info = get_state_info(mdp, state)

        serialized_state = decompress(
            state.compressed_serialized_state,
            state.ddict,
            state.uncompressed_size,
            context.compression_context,
        )
        set_state!(env, @view serialized_state[1:state.uncompressed_size])
        t::Int = state_info.min_time

        exploring::Bool = true
        while exploring
            state_to_queue::Union{NonTerminalState,Nothing} = nothing

            lock(state_info.lock)
            try
                @assert t < config.horizon
                exploring = false
                local next_state::State
                local next_t::Int
                for action_index in eachindex(mdp.actions)
                    if contains_transition(state_info, action_index)
                        continue
                    end
                    action = mdp.actions[action_index]

                    reward::Reward = 0
                    for repeat = 1:config.frameskip
                        reward += step(env, action)
                    end

                    next_t = t + 1
                    if next_t == config.horizon
                        for repeat = 1:config.noops_after_horizon*config.frameskip
                            reward += step(env, 0)
                        end
                    end

                    done::Bool = is_done(env)
                    if config.done_on_reward && !iszero(reward)
                        done = true
                    end

                    if done
                        next_state = TerminalState()
                    else
                        next_state = NonTerminalState(env, mdp, context)
                    end

                    record_transition!(state_info, action_index, next_state, reward, t)
                    if next_t < config.horizon && !done
                        exploring = true
                    end

                    break
                end

                if more_to_explore(state_info)
                    state_to_queue = state
                end

                if exploring
                    state = next_state
                    t = next_t
                end
            finally
                unlock(state_info.lock)
            end

            if exploring
                state_info = get_state_info!(mdp, state, t)
            end

            if state_to_queue isa NonTerminalState
                push!(state_queue, state_to_queue)
            end
        end

        if logger !== nothing
            log(mdp, state_queue, logger)
        end
    end
    if logger !== nothing
        log(mdp, state_queue, logger; force_all = true)
        println()
    end
end

function construct_mdp(config::MDPConfig)::MDP
    env = get_env(config)

    mdp = MDP(env)
    state_queue = ParallelSetDeque{NonTerminalState}()
    push!(state_queue, mdp.start_state)

    if config.log_dir === nothing
        logger = nothing
    else
        logger = TBLogger(config.log_dir)
    end

    # Setup envs before starting parallel tasks since trying to reset all the ALEs
    # in parallel sometimes leads to crash.
    println("Setting up $(config.num_workers) envs...")
    envs::Vector{Env} = []
    worker_contexts::Vector{WorkerContext} = []
    for task_index = 1:config.num_workers
        push!(envs, get_env(config))
        push!(worker_contexts, WorkerContext())
    end

    first_thread_done = false

    Threads.@threads for task_index = 1:config.num_workers
        # After starting the first thread, wait until there are plenty of states in
        # the queue to start the next.
        if task_index != 1
            while true
                if length(state_queue) >= config.num_workers * 2 || first_thread_done
                    break
                end
                sleep(0.01)
            end
        end
        explore_mdp!(
            config,
            envs[task_index],
            mdp,
            state_queue,
            worker_contexts[task_index];
            logger = task_index == 1 ? logger : nothing,
        )
        if task_index == 1
            first_thread_done = true
        end
    end

    set_unknown_states_terminal!(mdp; no_done_reward = config.no_done_reward)
    mdp
end

function set_unknown_states_terminal!(mdp::MDP; no_done_reward::Reward = 0)
    Threads.@threads for subdict in mdp.state_infos.subdicts
        for (state, state_info) in subdict
            for action_index in eachindex(state_info.transitions)
                if !haskey(mdp.state_infos, state_info.transitions[action_index])
                    if state_info.transitions[action_index] != TerminalState()
                        state_info.rewards[action_index] += no_done_reward
                    end
                    state_info.transitions[action_index] = TerminalState()
                end
            end
        end
    end
end

function consolidate(mdp::MDP; return_state_mapping = false)
    # States are consolidated if they have the same
    #  - transitions for each action
    #  - rewards for each action
    #  - and screen image
    consolidated_states = ParallelDict{Tuple{Array{UInt8},StateInfo},NonTerminalState}()
    state_mapping = ParallelDict{NonTerminalState,NonTerminalState}()
    Threads.@threads for subdict in mdp.state_infos.subdicts
        for (state, state_info) in subdict
            state_key = (state.compressed_screen_image, state_info)
            new_state = get!(consolidated_states, state_key, state)
            state_mapping[state] = new_state
        end
    end

    consolidated_mdp = MDP(mdp; state_mapping = state -> state_mapping[state])
    if return_state_mapping
        return (consolidated_mdp, state_mapping)
    else
        return consolidated_mdp
    end
end

function consolidate_completely(mdp::MDP; return_state_mapping = false)
    state_mapping = Dict{NonTerminalState,NonTerminalState}()
    while true
        prev_num_states = num_states(mdp)
        mdp, new_state_mapping = consolidate(mdp; return_state_mapping = true)
        if return_state_mapping
            merge!(state_mapping, new_state_mapping)
        end

        print(" "^120 * "\r")
        print("$(num_states(mdp)) total states  \t")
        print("$(Base.format_bytes(Base.gc_live_bytes())) memory used")
        print("\r")

        if num_states(mdp) == prev_num_states
            println()
            if return_state_mapping
                return mdp, state_mapping
            else
                return mdp
            end
        end
    end
end

function consolidate_completely!(
    mdp::MDP;
    ignore_screen = false,
    return_state_mapping = false,
)
    consolidated_states = ParallelDict{Tuple{Array{UInt8},StateInfo},NonTerminalState}()
    empty_screen = Vector{UInt8}(undef, 0)
    state_mapping = ParallelDict{NonTerminalState,NonTerminalState}()
    Threads.@threads for subdict in mdp.state_infos.subdicts
        for (state, state_info) in subdict
            screen = ignore_screen ? empty_screen : state.compressed_screen_image
            state_key = (screen, state_info)
            new_state = get!(consolidated_states, state_key, state)
            state_mapping[state] = new_state
        end
    end

    while true
        mdp_changed = false

        Threads.@threads for subdict in mdp.state_infos.subdicts
            for (old_state, state_info) in subdict
                new_state = state_mapping[old_state]
                if new_state != old_state
                    delete!(subdict, old_state)
                    mdp_changed = true
                else
                    changed = false
                    new_transitions = state_info.transitions[1:end]
                    for action_index in eachindex(mdp.actions)
                        old_next_state = state_info.transitions[action_index]
                        if old_next_state === nothing
                            new_next_state = nothing
                        elseif old_next_state isa TerminalState
                            new_next_state = TerminalState()
                        else
                            new_next_state = state_mapping[old_next_state]
                        end
                        if new_next_state != old_next_state
                            new_transitions[action_index] = new_next_state
                            changed = true
                        end
                    end

                    if changed
                        new_state_info = StateInfo(
                            new_transitions,
                            state_info.rewards,
                            state_info.min_time,
                            ReentrantLock(),
                        )
                        subdict[old_state] = new_state_info
                        screen =
                            ignore_screen ? empty_screen :
                            old_state.compressed_screen_image
                        state_key = (screen, new_state_info)
                        new_state = get!(consolidated_states, state_key, old_state)
                        state_mapping[old_state] = new_state
                        mdp_changed = true
                    end
                end
            end
        end

        print(" "^120 * "\r")
        print("$(num_states(mdp)) total states  \t")
        print("$(Base.format_bytes(Base.gc_live_bytes())) memory used")
        print("\r")

        if !mdp_changed
            println()

            while true
                old_start_state = mdp.start_state
                new_start_state = state_mapping[old_start_state]
                if new_start_state == old_start_state
                    break
                else
                    mdp.start_state = new_start_state
                end
            end

            if return_state_mapping
                return state_mapping
            else
                return mdp
            end
        end
    end
end

include("saving.jl")

end # module
