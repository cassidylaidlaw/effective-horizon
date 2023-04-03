
using CEnum

const libenv_env = Cvoid

@cenum libenv_dtype::UInt32 begin
    LIBENV_DTYPE_UNUSED = 0
    LIBENV_DTYPE_UINT8 = 1
    LIBENV_DTYPE_INT32 = 2
    LIBENV_DTYPE_FLOAT32 = 3
end

struct libenv_value
    data::NTuple{4,UInt8}
end

function Base.getproperty(x::Ptr{libenv_value}, f::Symbol)
    f === :uint8 && return Ptr{UInt8}(x + 0)
    f === :int32 && return Ptr{Int32}(x + 0)
    f === :float32 && return Ptr{Cfloat}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::libenv_value, f::Symbol)
    r = Ref{libenv_value}(x)
    ptr = Base.unsafe_convert(Ptr{libenv_value}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{libenv_value}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

@cenum libenv_scalar_type::UInt32 begin
    LIBENV_SCALAR_TYPE_UNUSED = 0
    LIBENV_SCALAR_TYPE_REAL = 1
    LIBENV_SCALAR_TYPE_DISCRETE = 2
end

@cenum libenv_space_name::UInt32 begin
    LIBENV_SPACE_UNUSED = 0
    LIBENV_SPACE_OBSERVATION = 1
    LIBENV_SPACE_ACTION = 2
    LIBENV_SPACE_INFO = 3
end

struct libenv_tensortype
    data::NTuple{212,UInt8}
end

function Base.getproperty(x::Ptr{libenv_tensortype}, f::Symbol)
    f === :name && return Ptr{NTuple{128,Cchar}}(x + 0)
    f === :scalar_type && return Ptr{libenv_scalar_type}(x + 128)
    f === :dtype && return Ptr{libenv_dtype}(x + 132)
    f === :shape && return Ptr{NTuple{16,Cint}}(x + 136)
    f === :ndim && return Ptr{Cint}(x + 200)
    f === :low && return Ptr{libenv_value}(x + 204)
    f === :high && return Ptr{libenv_value}(x + 208)
    return getfield(x, f)
end

function Base.getproperty(x::libenv_tensortype, f::Symbol)
    r = Ref{libenv_tensortype}(x)
    ptr = Base.unsafe_convert(Ptr{libenv_tensortype}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{libenv_tensortype}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct libenv_option
    name::NTuple{128,Cchar}
    dtype::libenv_dtype
    count::Cint
    data::Ptr{Cvoid}
end

libenv_option(name::String, dtype::libenv_dtype, count::Integer, data) = libenv_option(
    ntuple(i -> i <= ncodeunits(name) ? codeunit(name, i) : 0x0, 128),
    dtype,
    count,
    data,
)

function make_libenv_option!(
    name::String,
    dtype::libenv_dtype,
    data::Vector,
    keep_alive::Vector,
)
    push!(keep_alive, data)
    libenv_option(name, dtype, length(data), pointer(data))
end

make_libenv_option!(name::String, data::Vector{UInt8}, keep_alive::Vector) =
    make_libenv_option!(name, LIBENV_DTYPE_UINT8, data, keep_alive)

make_libenv_option!(name::String, data::String, keep_alive::Vector) =
    make_libenv_option!(name, LIBENV_DTYPE_UINT8, Vector{UInt8}(data), keep_alive)

make_libenv_option!(name::String, data::Integer, keep_alive::Vector) =
    make_libenv_option!(name, LIBENV_DTYPE_INT32, Int32[data], keep_alive)

make_libenv_option!(name::String, data::Float32, keep_alive::Vector) =
    make_libenv_option!(name, LIBENV_DTYPE_FLOAT32, Float32[data], keep_alive)

make_libenv_option!(name::String, data::UInt8, keep_alive::Vector) =
    make_libenv_option!(name, LIBENV_DTYPE_UINT8, UInt8[data], keep_alive)

struct libenv_options
    items::Ptr{libenv_option}
    count::Cint
end

struct libenv_buffers
    ob::Ptr{Ptr{Cvoid}}
    rew::Ptr{Cfloat}
    first::Ptr{UInt8}
    info::Ptr{Ptr{Cvoid}}
    ac::Ptr{Ptr{Cvoid}}
end

# no prototype is found for this function at libenv.h:146:16, please use with caution
function libenv_version()
    ccall((:libenv_version, libenv), Cint, ())
end

function libenv_make(num, options)
    ccall((:libenv_make, libenv), Ptr{libenv_env}, (Cint, libenv_options), num, options)
end

function libenv_get_tensortypes(handle, name, types)
    ccall(
        (:libenv_get_tensortypes, libenv),
        Cint,
        (Ptr{libenv_env}, libenv_space_name, Ptr{libenv_tensortype}),
        handle,
        name,
        types,
    )
end

function libenv_set_buffers(handle, bufs)
    ccall(
        (:libenv_set_buffers, libenv),
        Cvoid,
        (Ptr{libenv_env}, Ptr{libenv_buffers}),
        handle,
        bufs,
    )
end

function libenv_observe(handle)
    ccall((:libenv_observe, libenv), Cvoid, (Ptr{libenv_env},), handle)
end

function libenv_act(handle)
    ccall((:libenv_act, libenv), Cvoid, (Ptr{libenv_env},), handle)
end

function libenv_close(handle)
    ccall((:libenv_close, libenv), Cvoid, (Ptr{libenv_env},), handle)
end

function libenv_get_state(handle, env_idx, data, length)
    ccall(
        (:get_state, libenv),
        Cint,
        (Ptr{libenv_env}, Cint, Ptr{Cchar}, Cint),
        handle,
        env_idx,
        data,
        length,
    )
end

function libenv_set_state(handle, env_idx, data, length)
    ccall(
        (:set_state, libenv),
        Cint,
        (Ptr{libenv_env}, Cint, Ptr{Cchar}, Cint),
        handle,
        env_idx,
        data,
        length,
    )
end

# Skipping MacroDefinition: LIBENV_API __attribute__ ( ( __visibility__ ( "default" ) ) )

const LIBENV_MAX_NAME_LEN = 128

const LIBENV_MAX_NDIM = 16

const LIBENV_VERSION = 1
