
abstract type Env end

function get_actions(env::Env)::Vector{Int}
    throw("unimplemented")
end

function get_state(env::Env)::Vector{UInt8}
    throw("unimplemented")
end

function get_state(env::Env, buffer::Vector{UInt8})::Int
    state = get_state(env)
    buffer[1:length(state)] .= state
    length(state)
end

function get_screen(env::Env)::Vector{UInt8}
    throw("unimplemented")
end

function get_screen(env::Env, buffer::Vector{UInt8})::Int
    screen = get_state(env)
    buffer[1:length(screen)] .= screen
    length(screen)
end


function get_screen_width(env::Env)::Int
    throw("unimplemented")
end


function get_screen_height(env::Env)::Int
    throw("unimplemented")
end

function set_state!(env::Env, state::AbstractVector{UInt8})
    throw("unimplemented")
end

function is_done(env::Env)::Bool
    throw("unimplemented")
end

function step(env::Env, action::Int)::Reward
    throw("unimplemented")
end
