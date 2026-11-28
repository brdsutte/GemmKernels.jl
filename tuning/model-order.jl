# A model-guided search order for the tuner (GK_STRATEGY=model): see tuning/MODEL-ORDER.md.
#
# This file has no dependencies beyond Base, so that it can also be used without a GPU (e.g.
# by tuning/replay-order.jl). It uses no measured times: only the contraction, the code base's
# rules for which configurations are valid and how they copy data, and properties of the GPU.

module ModelOrder

export GPUInfo, ada_6000, ordered_candidates, estimate

# ------------------------------------------------------------------------------------------
# The GPU

Base.@kwdef struct GPUInfo
    sms::Int = 142                      # multiprocessors
    smem_per_sm::Int = 102400           # bytes of shared memory per SM
    smem_per_block::Int = 101376        # bytes of shared memory a block can use (opt-in)
    regs_per_sm::Int = 65536
    max_threads_per_sm::Int = 1536
    max_blocks_per_sm::Int = 24
    async::Bool = true                  # cp.async (sm_80 or later)
    max_tuple_splat::Int = 32           # GPUCompiler's limit on accumulator fragments per warp
    # generic throughput ratios of an Ada/Ampere SM, per cycle
    tensor_macs::Float64 = 256.0        # FP16 multiply-adds with FP32 accumulation
    global_bytes::Float64 = 14.0        # share of the L2 bandwidth
    shared_bytes::Float64 = 128.0
end

const ada_6000 = GPUInfo()

# ------------------------------------------------------------------------------------------
# Padding, as configs/configs.jl's pad_extents (copied, so that this file stands alone)

function two_factors(n::Integer)
    r = 0
    while n % 2 == 0
        r += 1
        n ÷= 2
    end
    r
end

increase_two_factor(n) = (m = 2^(two_factors(n) + 1); cld(n, m) * m)

function power_two_sequence(start, multiple)
    s = [start]
    while s[end] % multiple != 0
        push!(s, increase_two_factor(s[end]))
    end
    s
end

function minimal_padding(extents, multiple)
    res = collect(Iterators.filter(x -> prod(x) % multiple == 0,
                                   Iterators.product(power_two_sequence.(extents, multiple)...)))
    _, idx = findmin(prod, res)
    res[idx]
end

function pad_extents(extents, modes, multiples)
    padded = [extents...]
    for (i1, i2, multiple) in ((1, 2, multiples.M), (3, 2, multiples.K), (1, 3, multiples.N))
        sel = intersect(modes[i1], modes[i2])
        isempty(sel) && continue
        padded[sel] .= minimal_padding(padded[sel], multiple)
    end
    padded
end

# ------------------------------------------------------------------------------------------
# The contraction

struct Problem
    name::String
    extents::Vector{Int}
    modes::Vector{Vector{Int}}       # D, A, B
    Mmodes::Vector{Int}
    Nmodes::Vector{Int}
    Kmodes::Vector{Int}
end

function Problem(name::AbstractString, extents)
    modes = [parse.(Int, split(m, '.')) for m in split(name, '-')]
    D, A, B = modes
    Problem(String(name), collect(Int, extents), modes, intersect(D, A), intersect(D, B), intersect(A, B))
end

# ------------------------------------------------------------------------------------------
# How the GETT layouts copy a tensor (src/tensors/gett.jl, the override path of
# createGETTContractionPlan, and src/config.jl's copy tile heuristics)

# tensor positions of the modes, in the given order
positions(tensor_modes, ordered_modes) = [findfirst(==(m), tensor_modes) for m in ordered_modes]

# the leading run of memory-adjacent modes of a dimension (positions in a column-major
# tensor with these extents), as GETTLayout.copy_tile_crossing computes it
function leading_run(pos, ext)
    stride(p) = prod(@view ext[1:p-1]; init = 1)
    run = ext[pos[1]]
    for j in 2:length(pos)
        stride(pos[j]) == stride(pos[j-1]) * ext[pos[j-1]] || break
        run *= ext[pos[j]]
    end
    run
end

# Efficiency of copying one operand: returns (efficiency, valid). `dims` are the tensor
# positions of the modes of the copied (vector) dimension in layout order, `n` the elements
# of a thread's copy, `wv` the elements a warp covers along that dimension.
function copy_efficiency(dims, ext, unpadded_ext, n, wv, partial::Bool; elsize = 2)
    run = leading_run(dims, ext)
    n > 1 && run % n != 0 && return (0.0, false)      # would cross a mode boundary
    strided = dims[1] != 1
    strided && return (0.25, true)
    seg = min(run, wv) * elsize
    misaligned = (run * elsize) % 32 != 0
    eff = seg / (32 * cld(seg, 32) + (misaligned ? 16 : 0))
    if partial && (unpadded_ext[dims[1]] * elsize) % 32 != 0
        eff *= 0.9
    end
    (min(eff, 1.0), true)
end

# adjacent_elements(num, (d1, d2), col_major): the elements along d1 and d2
adjacent(num, d1, d2, col_major) = col_major ? (min(num, d1), num ÷ min(num, d1)) :
                                               (num ÷ min(num, d2), min(num, d2))

# ------------------------------------------------------------------------------------------
# Kernels

kernel_stages(k) = k == "pipelined_ng" ? 2 :
                   startswith(k, "pipelined_async_") ? parse(Int, last(split(k, '_'))) : 1
is_async(k) = startswith(k, "pipelined_async_")
double_buffered(k) = k in ("pipelined", "pipelined_ng") || startswith(k, "pipelined_async_db_")
overlaps(k) = k != "singlestage"
needed_warps(k) = k == "singlestage" ? 16 : startswith(k, "pipelined_async_db_") ? 4 : 8
# how many K-tiles ahead a kernel loads (to hide the latency of global memory), and its
# barriers per K-tile
prefetch_depth(k) = k == "singlestage" ? 1 : is_async(k) ? kernel_stages(k) - 1 : 2
barriers(k) = k == "singlestage" ? 2 : k == "pipelined" ? 3 : k == "pipelined_ng" ? 2 : 1
const GLOBAL_LATENCY = 600.0    # cycles, an L2 hit
const BARRIER_CYCLES = 40.0

# ------------------------------------------------------------------------------------------
# The structural part of a configuration, and its estimate apart from the layout

struct Structure
    BM::Int; BN::Int; BK::Int; WM::Int; WN::Int
    OPM::Int; OPN::Int; OPK::Int
    kernel::String; split_k::Int; indexing::String; edge::String
end

# validity as the tuner's pre-check decides it (tuning/wmma-contraction.jl), for the parts
# that do not depend on the layout
function structure_valid(s::Structure, p::Problem, gpu::GPUInfo, padded)
    warps = s.WM * s.WN
    (s.BM * s.BK < 128 || s.BK * s.BN < 128 || s.BK < 8) && return false
    (s.BM < 2 * s.OPM || s.BN < 2 * s.OPN || s.BK < s.OPK) && return false
    (s.BM * s.BK < 32 * warps || s.BK * s.BN < 32 * warps || s.BM * s.BN < 32 * warps) && return false
    wm, wn = s.BM ÷ s.WM, s.BN ÷ s.WN
    (wm == 0 || wn == 0 || wm % s.OPM != 0 || wn % s.OPN != 0) && return false
    (s.kernel == "pipelined" && s.BK < 2 * s.OPK) && return false
    (is_async(s.kernel) && !gpu.async) && return false
    if s.split_k != 1
        s.kernel in ("pipelined", "pipelined_ng") && return false
        k_tiles = prod(padded[p.Kmodes]) ÷ s.BK
        s.split_k > k_tiles && return false
    end
    s.indexing == "lean32" && any(m -> prod(padded[m]) > typemax(Int32), p.modes) && return false
    shared_bytes(s) > gpu.smem_per_block && return false
    (wm ÷ s.OPM) * (wn ÷ s.OPN) > gpu.max_tuple_splat && return false
    registers(s) > 255 && return false
    registers(s) * 32 * warps > gpu.regs_per_sm && return false
    true
end

# the pre-check's lower bound of the kernel's shared memory, with the padding of the shared
# layouts (8 elements per column)
shared_bytes(s::Structure) = max(4 * s.BM * s.BN,
                                 kernel_stages(s.kernel) * 2 * ((s.BM + 8) * s.BK + s.BK * (s.BN + 8)))

# registers per thread: accumulators, fragments of A and B (twice when double-buffered), the
# pipelined kernels' staging registers for the next K-tile, and a base for addresses etc.
function registers(s::Structure)
    warps = s.WM * s.WN
    wm, wn = s.BM ÷ s.WM, s.BN ÷ s.WN
    acc = wm * wn ÷ 32
    frags = (wm * s.OPK + wn * s.OPK) ÷ 64
    double_buffered(s.kernel) && (frags *= 2)
    staging = s.kernel in ("pipelined", "pipelined_ng") ? (s.BM * s.BK + s.BK * s.BN) ÷ (2 * 32 * warps) : 0
    r = acc + frags + staging + 40
    8 * cld(r, 8)
end

function resident_blocks(s::Structure, gpu::GPUInfo)
    threads = 32 * s.WM * s.WN
    by_smem = gpu.smem_per_sm ÷ (shared_bytes(s) + 1024)
    by_regs = gpu.regs_per_sm ÷ (registers(s) * threads)
    by_threads = gpu.max_threads_per_sm ÷ threads
    max(0, min(by_smem, by_regs, by_threads, gpu.max_blocks_per_sm))
end

# the estimated time of a configuration, given its copy efficiencies (A, B, D)
function estimate(s::Structure, p::Problem, gpu::GPUInfo, padded, effA, effB, effD)
    warps = s.WM * s.WN
    M = prod(padded[p.Mmodes]); N = prod(padded[p.Nmodes]); K = prod(padded[p.Kmodes])
    tiles = cld(M, s.BM) * cld(N, s.BN) * s.split_k
    per_sm = cld(tiles, gpu.sms)

    k_part = cld(cld(K, s.split_k), s.BK) * s.BK
    compute = s.BM * s.BN / gpu.tensor_macs
    shared = warps * 2 * (s.BM ÷ s.WM + s.BN ÷ s.WN) / gpu.shared_bytes
    globl = (2 * s.BM / effA + 2 * s.BN / effB) / gpu.global_bytes
    core = max(compute, shared)

    r = resident_blocks(s, gpu)
    r == 0 && return Inf
    r_eff = min(r, per_sm)
    # per K-tile: the larger of the tensor-core, shared-memory and global-memory time (for
    # singlestage, which does not overlap its own copies, plus the part other resident blocks
    # cannot hide), and of the latency of global memory, as far as loading ahead and other
    # resident blocks hide it; plus the barriers
    work = s.BK * max(core, globl)
    overlaps(s.kernel) || (work += s.BK * min(core, globl) / r_eff)
    tile = max(work, GLOBAL_LATENCY / (prefetch_depth(s.kernel) * r_eff)) + barriers(s.kernel) * BARRIER_CYCLES
    block = (k_part ÷ s.BK) * tile
    block += s.BK * globl + GLOBAL_LATENCY                          # prologue: the first K-tile
    block += 2 * s.BM * s.BN / effD / gpu.global_bytes              # epilogue: storing D
    s.split_k > 1 && (block += 8 * s.BM * s.BN / gpu.global_bytes)  # partial tiles' round trip

    hiding = min(1.0, r_eff * warps / needed_warps(s.kernel))
    overhead = s.indexing == "default" ? 1.10 : 1.0
    per_sm * block / hiding * overhead
end

swizzle_factor(str) = (n = parse(Int, last(split(str, '-'))); 1 + 0.005 * log2(n))

# the estimated time of a structure with a layout (Inf if the layout is invalid)
function layout_estimate(s::Structure, p::Problem, gpu::GPUInfo, padded, a_col, b_col, d_col, pM, pN, pK)
    D, A, B = p.modes
    BM, BN, BK = s.BM, s.BN, s.BK
    warps = s.WM * s.WN
    nA = min(8, BM * BK ÷ (32 * warps)); nB = min(8, BK * BN ÷ (32 * warps)); nC = min(8, BM * BN ÷ (32 * warps))
    partial = s.edge == "partial"
    Mtot = prod(padded[p.Mmodes]); Ntot = prod(padded[p.Nmodes])
    # A: (M, K); B: (K, N); D: (M, N)
    tA = adjacent(nA, BM, BK, a_col); wA = adjacent(32 * nA, BM, BK, a_col)
    effA, okA = a_col ? copy_efficiency(positions(A, pM), padded, p.extents, tA[1], wA[1], partial) :
                        copy_efficiency(positions(A, pK), padded, p.extents, tA[2], wA[2], partial)
    okA || return Inf
    tB = adjacent(nB, BK, BN, b_col); wB = adjacent(32 * nB, BK, BN, b_col)
    effB, okB = b_col ? copy_efficiency(positions(B, pK), padded, p.extents, tB[1], wB[1], partial) :
                        copy_efficiency(positions(B, pN), padded, p.extents, tB[2], wB[2], partial)
    okB || return Inf
    tD = adjacent(nC, BM, BN, d_col); wD = adjacent(32 * nC, BM, BN, d_col)
    effD, okD = d_col ? copy_efficiency(positions(D, pM), padded, p.extents, tD[1], wD[1], partial) :
                        copy_efficiency(positions(D, pN), padded, p.extents, tD[2], wD[2], partial)
    okD || return Inf
    # partial edge tiles: M and N must be multiples of the copy tiles
    if partial
        (Mtot % tA[1] == 0 && Ntot % tB[2] == 0 && Mtot % tD[1] == 0 && Ntot % tD[2] == 0) || return Inf
    end

    estimate(s, p, gpu, padded, effA, effB, effD)
end

"""
    score(config; gpu = ada_6000)

The model's estimate for one configuration (a NamedTuple or row with the tuner's fields; missing
SPLIT_K, INDEXING and EDGE default to 1, "default" and "pad"), including the swizzle; Inf if the
model considers it invalid.
"""
function score(c; gpu::GPUInfo = ada_6000, problem = Problem(c.name, collect(c.extents)))
    p = problem
    get_(f, d) = hasproperty(c, f) ? getproperty(c, f) : d
    edge = get_(:EDGE, "pad")
    s = Structure(c.BLOCK_M, c.BLOCK_N, c.BLOCK_K, c.WARPS_M, c.WARPS_N, c.OP_M, c.OP_N, c.OP_K,
                  c.kernel_str, get_(:SPLIT_K, 1), get_(:INDEXING, "default"), edge)
    padded = pad_extents(p.extents, p.modes, edge == "partial" ? (M = 1, N = 1, K = s.BK) : (M = s.BM, N = s.BN, K = s.BK))
    structure_valid(s, p, gpu, padded) || return Inf
    t = layout_estimate(s, p, gpu, padded, c.is_A_col_major, c.is_B_col_major, c.is_D_col_major,
                        collect(c.PERM_M), collect(c.PERM_N), collect(c.PERM_K))
    t * swizzle_factor(c.cta_swizzle_str)
end

# ------------------------------------------------------------------------------------------
# The order

"""
    ordered_candidates(name, extents; kernels, split_k_values, indexing_values, edge_values,
                       swizzles, gpu = ada_6000, layouts_per_structure = 4, limit = 200_000,
                       valid = nothing)

The configurations of a contraction in increasing order of estimated time, as NamedTuples with
the fields of the tuner's configurations (tuning/wmma-contraction.jl). Every valid structural
part (block shape, warps, operator shape, kernel, split-K, edge, indexing) is combined with its
`layouts_per_structure` best layouts and every swizzle. `valid(config)`, if given, is an extra
validity check (the tuner passes its pre-check). At most `limit` configurations are returned.
"""
function ordered_candidates(name, extents; kernels, split_k_values, indexing_values = ["default"],
                            edge_values = ["pad"],
                            swizzles = ["horizontal-1", "horizontal-2", "horizontal-4", "horizontal-8", "horizontal-16",
                                        "vertical-1", "vertical-2", "vertical-4", "vertical-8", "vertical-16"],
                            gpu::GPUInfo = ada_6000, layouts_per_structure = 4, limit = 200_000,
                            valid = nothing, with_estimates = false)
    p = Problem(name, extents)
    D, A, B = p.modes
    perms_M = collect(permutations_of(p.Mmodes))
    perms_N = collect(permutations_of(p.Nmodes))
    perms_K = collect(permutations_of(p.Kmodes))

    candidates = Tuple{Float64, Candidate}[]
    padded_cache = Dict{Tuple{Int, Int, Int, String}, Vector{Int}}()

    for BM in 2 .^ (3:9), BN in 2 .^ (3:9), BK in 2 .^ (3:7), WM in 2 .^ (0:3), WN in 2 .^ (0:3),
        (OPM, OPN, OPK) in ((16, 16, 16), (8, 32, 16), (32, 8, 16)),
        kernel in kernels, split_k in split_k_values, indexing in indexing_values, edge in edge_values

        s = Structure(BM, BN, BK, WM, WN, OPM, OPN, OPK, kernel, split_k, indexing, edge)
        padded = get!(padded_cache, (BM, BN, BK, edge)) do
            pad_extents(p.extents, p.modes, edge == "partial" ? (M = 1, N = 1, K = BK) : (M = BM, N = BN, K = BK))
        end
        structure_valid(s, p, gpu, padded) || continue
        # the extra check (the tuner's pre-check) does not depend on the layout or the swizzle,
        # so one configuration per structure tells
        valid === nothing ||
            valid(to_config(p, Candidate(s, false, false, false, perms_M[1], perms_N[1], perms_K[1], swizzles[1]))) ||
            continue

        # the layouts: estimate each, keep the best few
        best = Tuple{Float64, Tuple}[]
        for a_col in (false, true), b_col in (false, true), d_col in (false, true),
            pM in perms_M, pN in perms_N, pK in perms_K

            t = layout_estimate(s, p, gpu, padded, a_col, b_col, d_col, pM, pN, pK)
            isfinite(t) || continue
            push!(best, (t, (a_col, b_col, d_col, pM, pN, pK)))
        end
        isempty(best) && continue
        sort!(best; by = first)

        for (t, (a_col, b_col, d_col, pM, pN, pK)) in Iterators.take(best, layouts_per_structure),
            sw in swizzles

            push!(candidates, (t * swizzle_factor(sw), Candidate(s, a_col, b_col, d_col, pM, pN, pK, sw)))
        end
    end

    ordered = [(t, to_config(p, c)) for (t, c) in diversify(candidates, limit)]
    with_estimates ? ordered : [c for (_, c) in ordered]
end

# a candidate while ordering (lighter than the tuner's NamedTuple, built only for the result)
struct Candidate
    s::Structure
    a_col::Bool; b_col::Bool; d_col::Bool
    pM::Vector{Int}; pN::Vector{Int}; pK::Vector{Int}
    sw::String
end

to_config(p::Problem, c::Candidate) =
    (; name = p.name, extents = copy(p.extents),
       BLOCK_M = c.s.BM, BLOCK_N = c.s.BN, BLOCK_K = c.s.BK, WARPS_M = c.s.WM, WARPS_N = c.s.WN,
       OP_M = c.s.OPM, OP_N = c.s.OPN, OP_K = c.s.OPK, kernel_str = c.s.kernel, SPLIT_K = c.s.split_k,
       INDEXING = c.s.indexing, EDGE = c.s.edge,
       is_A_col_major = c.a_col, is_B_col_major = c.b_col, is_D_col_major = c.d_col,
       PERM_M = c.pM, PERM_N = c.pN, PERM_K = c.pK, cta_swizzle_str = c.sw)

# Many candidates have equal or nearly equal estimates (e.g. they differ only in the number of
# stages, a swizzle or a layout that the model cannot tell apart), and trying them one after
# the other would spend the budget on near-duplicates. So candidates of the same tile shape
# (block shape, warps, operator shape) are penalised by 3% for every earlier one: the order
# visits the most promising tile shapes first and comes back to their variants later.
const REPEAT_PENALTY = 0.03

#
# Revision 1 (after inspecting the model's own order, before any measurement with it): with
# only tile shapes as groups, the first 200 configurations of TC1 and TC13 covered just 3-4
# block shapes, as the model barely tells warp and operator shapes apart (a replay against a
# sparse sample of the space cannot show this). So the penalty now counts earlier candidates
# at three levels: of the same block shape (BM, BN, BK), of the same tile shape (block shape,
# warps, operator shape) and of the same tile shape and kernel, 3% for each. The order thus
# spreads over block shapes first, then over their tile shapes and kernels.
# GK_MODEL_GROUP=tile restores revision 0.
const GROUPING = get(ENV, "GK_MODEL_GROUP", "levels")
GROUPING in ("levels", "tile") || error("GK_MODEL_GROUP must be levels or tile, not $GROUPING")

block_of(c) = (c.BLOCK_M, c.BLOCK_N, c.BLOCK_K)
group_of(c) = (c.BLOCK_M, c.BLOCK_N, c.BLOCK_K, c.WARPS_M, c.WARPS_N, c.OP_M, c.OP_N, c.OP_K)
kernel_group_of(c) = (group_of(c)..., c.kernel_str)
block_of(c::Candidate) = (c.s.BM, c.s.BN, c.s.BK)
group_of(c::Candidate) = (c.s.BM, c.s.BN, c.s.BK, c.s.WM, c.s.WN, c.s.OPM, c.s.OPN, c.s.OPK)
kernel_group_of(c::Candidate) = (group_of(c)..., c.s.kernel)

function diversify(candidates, limit)
    sort!(candidates; by = first)
    leaf = GROUPING == "tile" ? group_of : kernel_group_of
    groups = Dict{Any, Vector{Int}}()
    for (i, (_, c)) in enumerate(candidates)
        push!(get!(groups, leaf(c), Int[]), i)
    end
    # how many candidates have been taken, per group at each level
    taken = Dict{Any, Int}()
    count(key) = get(taken, key, 0)
    function penalty(c)
        GROUPING == "tile" && return count(group_of(c))
        count(block_of(c)) + count(group_of(c)) + count(kernel_group_of(c))
    end
    # a heap of (penalised estimate, group, position within the group, penalty used): the
    # penalties only grow, so a popped entry whose penalty is stale is pushed back updated
    heap = [(candidates[idx[1]][1], g, 1, 0) for (g, idx) in groups]
    heapify!(heap)
    out = similar(candidates, 0)
    while !isempty(heap) && length(out) < limit
        (_, g, k, used) = heappop!(heap)
        idx = groups[g]
        t, c = candidates[idx[k]]
        pen = penalty(c)
        if pen != used
            heappush!(heap, (t * (1 + REPEAT_PENALTY * pen), g, k, pen))
            continue
        end
        push!(out, candidates[idx[k]])
        for key in (block_of(c), group_of(c), kernel_group_of(c))
            taken[key] = count(key) + 1
        end
        if k < length(idx)
            t2, c2 = candidates[idx[k+1]]
            pen2 = penalty(c2)
            heappush!(heap, (t2 * (1 + REPEAT_PENALTY * pen2), g, k + 1, pen2))
        end
    end
    out
end

# a minimal binary heap on the first element of tuples (to avoid a dependency)
function heapify!(h)
    for i in length(h) ÷ 2:-1:1
        siftdown!(h, i)
    end
    h
end
function siftdown!(h, i)
    n = length(h)
    while true
        l = 2i; r = 2i + 1; m = i
        l <= n && h[l][1] < h[m][1] && (m = l)
        r <= n && h[r][1] < h[m][1] && (m = r)
        m == i && return
        h[i], h[m] = h[m], h[i]
        i = m
    end
end
function heappop!(h)
    top = h[1]
    h[1] = h[end]
    pop!(h)
    isempty(h) || siftdown!(h, 1)
    top
end
function heappush!(h, x)
    push!(h, x)
    i = length(h)
    while i > 1 && h[i][1] < h[i ÷ 2][1]
        h[i], h[i ÷ 2] = h[i ÷ 2], h[i]
        i ÷= 2
    end
end

# all permutations of a vector (small: at most 4 modes per dimension in the TCCG suite)
function permutations_of(v::Vector{Int})
    length(v) <= 1 && return [copy(v)]
    out = Vector{Vector{Int}}()
    for i in eachindex(v)
        rest = [v[j] for j in eachindex(v) if j != i]
        for r in permutations_of(rest)
            push!(out, [v[i]; r])
        end
    end
    out
end

end # module
