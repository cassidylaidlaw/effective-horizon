
using ZipFile
using NPZ


const global NPY_MAGIC_STRING = b"\x93NUMPY\x01\x00"

function npy_header(descr, shape::Tuple{Vararg{Int}}; fortran_order = true)
    fortran_order_python = fortran_order ? "True" : "False"
    header = (
        "{'descr': '$(descr)', 'fortran_order': $(fortran_order_python), " *
        "'shape': ($(join(map(string, shape), ", ")),)}"
    )
    return vcat(
        NPY_MAGIC_STRING,
        [(length(header) % 256) % UInt8, (length(header) ÷ 255) % UInt8],
        Base.codeunits(header),
    )
end



function get_state_indices(mdp::MDP)
    state_indices = ParallelDict{NonTerminalState,Int}()
    state_index = Threads.Atomic{Int}(1)
    Threads.@threads for subdict in mdp.state_infos.subdicts
        for (state, _) in subdict
            if state == mdp.start_state
                state_indices[state] = 0
            else
                state_indices[state] = Threads.atomic_add!(state_index, 1)
            end
        end
    end
    state_indices
end

function get_array_representation(
    mdp::MDP,
    state_indices::AbstractDict{NonTerminalState,Int},
)
    transitions_array = Matrix{Int}(undef, num_states(mdp), num_actions(mdp))
    rewards_array = Matrix{Reward}(undef, num_states(mdp), num_actions(mdp))
    Threads.@threads for subdict in mdp.state_infos.subdicts
        for (state, state_info) in subdict
            state_index = state_indices[state]
            for action_index in eachindex(mdp.actions)
                next_state = state_info.transitions[action_index]
                if next_state === nothing
                    throw(ArgumentError("Not all transitions have been explored."))
                elseif next_state isa TerminalState
                    next_state_index = -1
                else
                    next_state_index = state_indices[next_state]
                end
                transitions_array[state_index+1, action_index] = next_state_index
                rewards_array[state_index+1, action_index] =
                    state_info.rewards[action_index]
            end
        end
    end
    (transitions_array, rewards_array)
end

function write_screens(mdp::MDP, screens_file::IO)
    screen_indices = Dict{ScreenImage,Int}()
    screen_size = 3 * mdp.screen_height * mdp.screen_width
    screen_index = 0
    write(
        screens_file,
        npy_header(
            "<u1",
            (length(mdp.screen_images), mdp.screen_height, mdp.screen_width, 3);
            fortran_order = false,
        ),
    )
    context = CompressionContext()
    for (compressed_screen_image, _) in mdp.screen_images
        screen_image = decompress(compressed_screen_image, screen_size, context)
        write(screens_file, @view screen_image[1:screen_size])
        screen_indices[compressed_screen_image] = screen_index
        screen_index += 1
    end
    screen_indices
end

function get_screen_mapping_array(
    mdp::MDP,
    state_indices::AbstractDict{NonTerminalState,Int},
    screen_indices::AbstractDict{ScreenImage,Int},
)
    screen_mapping_array = Vector{Int}(undef, num_states(mdp))
    for (state, state_index) in state_indices
        screen_mapping_array[state_index+1] =
            screen_indices[state.compressed_screen_image]
    end
    screen_mapping_array
end

function save(mdp::MDP, save_path; compress = true, save_screens = false)
    state_indices = get_state_indices(mdp)
    transitions_array, rewards_array = get_array_representation(mdp, state_indices)

    if compress
        zip_writer = ZipFile.Writer("$(save_path).npz")
    else
        mkpath(save_path)
    end

    if compress
        transitions_file =
            ZipFile.addfile(zip_writer, "transitions.npy", method = ZipFile.Deflate)
    else
        transitions_file = open(joinpath(save_path, "transitions.npy"), "w")
    end
    write(
        transitions_file,
        npy_header("<i$(sizeof(Int))", (num_states(mdp), num_actions(mdp))),
    )
    write(transitions_file, reinterpret(UInt8, vec(transitions_array)))
    close(transitions_file)

    if compress
        rewards_file =
            ZipFile.addfile(zip_writer, "rewards.npy", method = ZipFile.Deflate)
    else
        rewards_file = open(joinpath(save_path, "rewards.npy"), "w")
    end
    write(
        rewards_file,
        npy_header("<f$(sizeof(Reward))", (num_states(mdp), num_actions(mdp))),
    )
    write(rewards_file, reinterpret(UInt8, vec(rewards_array)))
    close(rewards_file)

    if save_screens
        if compress
            screens_file =
                ZipFile.addfile(zip_writer, "screens.npy", method = ZipFile.Deflate)
        else
            screens_file = open(joinpath(save_path, "screens.npy"), "w")
        end
        screen_indices = write_screens(mdp, screens_file)
        close(screens_file)

        screen_mapping_array =
            get_screen_mapping_array(mdp, state_indices, screen_indices)
        if compress
            screen_mapping_file = ZipFile.addfile(
                zip_writer,
                "screen_mapping.npy",
                method = ZipFile.Deflate,
            )
        else
            screen_mapping_file = open(joinpath(save_path, "screen_mapping.npy"), "w")
        end
        write(screen_mapping_file, npy_header("<i$(sizeof(Int))", (num_states(mdp),)))
        write(screen_mapping_file, reinterpret(UInt8, screen_mapping_array))
        close(screen_mapping_file)
    end

    if compress
        close(zip_writer)
    end

    return nothing
end

function load_mdp(mdp_fname)
    local transitions, rewards
    mdp_zip = ZipFile.Reader(mdp_fname)
    for file in mdp_zip.files
        if file.name === "transitions.npy"
            transitions = NPZ.npzreadarray(file)
        elseif file.name === "rewards.npy"
            rewards = NPZ.npzreadarray(file)
        end
    end
    close(mdp_zip)

    num_states, num_actions = size(transitions)
    done_state = num_states
    num_states += 1
    transitions = cat(transitions, zeros(eltype(transitions), 1, num_actions); dims = 1)
    transitions[transitions.==-1] .= done_state
    transitions[done_state+1, :] .= done_state
    rewards = cat(rewards, zeros(eltype(rewards), 1, num_actions); dims = 1)
    @assert eltype(rewards) <: Float32
    (transitions, rewards)
end
