
using CodecZstd.LibZstd

const DEFAULT_COMPRESSION_LEVEL = 3
const MAX_DATA_LEN = 2000000

include("config.jl")

mutable struct ZstdCDict
    ptr::Ptr{ZSTD_CDict}
end

mutable struct CompressionContext
    cctx::Ptr{ZSTD_CCtx}
    dctx::Ptr{ZSTD_DCtx}
    buffer::Vector{UInt8}

    function CompressionContext()
        cctx = ZSTD_createCCtx()
        dctx = ZSTD_createDCtx()
        buffer = zeros(UInt8, ZSTD_compressBound(MAX_DATA_LEN))

        context = new(cctx, dctx, buffer)
        finalizer(context) do context
            ZSTD_freeCCtx(context.cctx)
            ZSTD_freeDCtx(context.dctx)
        end
    end
end


function ZstdCDict(
    dict_buffer::AbstractVector{UInt8};
    level::Integer = DEFAULT_COMPRESSION_LEVEL,
)
    cdict = ZstdCDict(ZSTD_createCDict(dict_buffer, length(dict_buffer), level))
    finalizer(cdict) do cdict
        ZSTD_freeCDict(cdict.ptr)
    end
    cdict
end

function compress(
    data,
    cdict::ZstdCDict,
    context::CompressionContext;
    return_view = false,
)::AbstractVector{UInt8}
    if AVOID_MALLOC
        @assert length(data) <= MAX_DATA_LEN
        csize = ZSTD_compress_usingCDict(
            context.cctx,
            context.buffer,
            length(context.buffer),
            data,
            length(data),
            cdict.ptr,
        )
        if return_view
            return @view context.buffer[1:csize]
        else
            return context.buffer[1:csize]
        end
    else
        buff = Vector{UInt8}(undef, ZSTD_compressBound(length(data)))
        cctx = ZSTD_createCCtx()
        csize = ZSTD_compress_usingCDict(
            cctx,
            buff,
            length(buff),
            data,
            length(data),
            cdict.ptr,
        )
        ZSTD_freeCCtx(cctx)
        buff[1:csize]
    end
end

function compress(data, context::CompressionContext)::AbstractVector{UInt8}
    if AVOID_MALLOC
        @assert length(data) <= MAX_DATA_LEN
        csize = ZSTD_compressCCtx(
            context.cctx,
            context.buffer,
            length(context.buffer),
            data,
            length(data),
            DEFAULT_COMPRESSION_LEVEL,
        )
        context.buffer[1:csize]
    else
        buff = Vector{UInt8}(undef, ZSTD_compressBound(length(data)))
        cctx = ZSTD_createCCtx()
        csize = ZSTD_compressCCtx(
            cctx,
            buff,
            length(buff),
            data,
            length(data),
            DEFAULT_COMPRESSION_LEVEL,
        )
        ZSTD_freeCCtx(cctx)
        buff[1:csize]
    end
end

mutable struct ZstdDDict
    ptr::Ptr{ZSTD_DDict}
end

function ZstdDDict(dict_buffer::AbstractVector{UInt8})
    ddict = ZstdDDict(ZSTD_createDDict(dict_buffer, length(dict_buffer)))
    ddict
    finalizer(ddict) do ddict
        ZSTD_freeDDict(ddict.ptr)
    end
end

function decompress(
    compressed_data::Vector{UInt8},
    ddict::ZstdDDict,
    uncompressed_size::Integer,
    context::CompressionContext,
)::AbstractVector{UInt8}
    if AVOID_MALLOC
        @assert uncompressed_size <= MAX_DATA_LEN
        context.buffer .= 0
        ZSTD_decompress_usingDDict(
            context.dctx,
            context.buffer,
            length(context.buffer),
            compressed_data,
            length(compressed_data),
            ddict.ptr,
        )
        context.buffer
    else
        data = Vector{UInt8}(undef, uncompressed_size)
        dctx = ZSTD_createDCtx()
        ZSTD_decompress_usingDDict(
            dctx,
            data,
            length(data),
            compressed_data,
            length(compressed_data),
            ddict.ptr,
        )
        ZSTD_freeDCtx(dctx)
        data
    end
end

function decompress(
    compressed_data::Vector{UInt8},
    uncompressed_size::Integer,
    context::CompressionContext,
)::AbstractVector{UInt8}
    if AVOID_MALLOC
        @assert uncompressed_size <= MAX_DATA_LEN
        context.buffer .= 0
        ZSTD_decompressDCtx(
            context.dctx,
            context.buffer,
            length(context.buffer),
            compressed_data,
            length(compressed_data),
        )
        context.buffer
    else
        data = Vector{UInt8}(undef, uncompressed_size)
        dctx = ZSTD_createDCtx()
        ZSTD_decompressDCtx(
            dctx,
            data,
            length(data),
            compressed_data,
            length(compressed_data),
        )
        ZSTD_freeDCtx(dctx)
        data
    end
end

const DUMMY_CDICT = ZstdCDict(b"")
const DUMMY_DDICT = ZstdDDict(b"")
