# Run a selection of GemmKernels kernels once each, small enough to run under NVIDIA's
# compute-sanitizer, which checks every memory access of the kernels it instruments:
#
#     JULIA_CUDA_MEMORY_POOL=none compute-sanitizer --tool memcheck --kernel-name kns=matmul \
#         julia +1.11 --project=tuning test/sanitize.jl
#
# Tools: memcheck (out-of-bounds and misaligned accesses), racecheck (shared-memory data
# races), synccheck (invalid barriers), initcheck (reads of uninitialised global memory).
# `--kernel-name kns=matmul` restricts the checks to GemmKernels' kernels (all are named
# matmul_*), not cuTENSOR's or CUDA.jl's. Disabling CUDA.jl's memory pool lets memcheck see
# every allocation's real bounds.
#
# This prints a line per case (whether the result is correct); compute-sanitizer prints a
# report for every error or hazard it finds, and a summary at the end.
#
# Environment: CASES=substring runs only the cases whose name contains it.

using CUDA, GemmKernels, Printf
include(joinpath(@__DIR__, "..", "configs", "configs.jl"))

const ONLY = get(ENV, "CASES", "")
const HAS_ASYNC = CUDA.capability(device()) ≥ v"8.0"

const KERNELS = [
    "singlestage" => Kernel.matmul_singlestage,
    "pipelined" => Kernel.matmul_pipelined,
    "pipelined_ng" => Kernel.matmul_pipelined_ng,
    "async_2" => Kernel.matmul_pipelined_async(2),
    "async_3" => Kernel.matmul_pipelined_async(3),
    "async_4" => Kernel.matmul_pipelined_async(4),
]
kernels(; async = true) = [kv for kv in KERNELS if !startswith(first(kv), "async") || (async && HAS_ASYNC)]

# A small instance of the contraction the GETT tests use (abc-bda-dc).
small_tc() = WMMATensorContraction(; name = "1.2.3-2.4.1-4.3", extents = [64, 64, 24, 64],
                                     data_type = Float16, compute_type = Float16,
                                     accumulate_type = Float32, zero_c = true)

const results = Pair{String, String}[]

function run_case(name, problem, is_tc, params)
    isempty(ONLY) || occursin(ONLY, name) || return
    status = try
        data = allocate_data(problem)
        initialize_data(problem, data...)
        reference = Array(calculate_reference(problem, data...))
        args = try
            is_tc ? prepare(problem, data...; params...) : prepare(problem; params...)
        finally
            GemmKernels.Tensors.OVERRIDE_do_override = false
        end
        is_tc && initialize_data(problem, data...; params...)
        result = Array(execute(problem, data...; args...))
        synchronize()
        ok = is_tc ? verify(problem, reference, result) :
                     isapprox(result, reference; rtol = sqrt(eps(Float16)))
        ok ? "correct" : "WRONG RESULT"
    catch err
        isa(err, InterruptException) && rethrow()
        isa(err, GemmKernels.ConfigError) ? "skipped (configuration error)" :
            "ERROR: " * first(split(sprint(showerror, err), '\n'))
    end
    @printf("%-60s %s\n", name, status)
    flush(stdout)
    push!(results, name => status)
end

# 1. GEMM, all kernels, plain and transposed inputs.
for (transpose_a, transpose_b) in [(false, false), (true, true)], (kname, kernel) in kernels()
    problem = WMMAMatrixMultiplication(; M = 256, N = 128, K = 256, AB_type = Float16, CD_type = Float32,
                                         transpose_a, transpose_b, zero_c = false)
    run_case("GEMM $(transpose_a ? 'T' : 'N')$(transpose_b ? 'T' : 'N'), $kname", problem, false,
             (; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                OP_M = 16, OP_N = 16, OP_K = 16, zero_c = false, kernel))
end

# 2. GETT layouts with few elements per thread (test/gett-layout.jl): vectorised and strided A.
for (kname, kernel) in kernels(; async = false), perm_m in [[2, 1], [1, 2]]
    run_case("GETT 4 elements of A per thread, PERM_M = $perm_m, $kname", small_tc(), true,
             (; BLOCK_M = 16, BLOCK_N = 128, BLOCK_K = 32, WARPS_M = 1, WARPS_N = 4,
                OP_M = 8, OP_N = 32, OP_K = 16, kernel,
                is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                PERM_M = perm_m, PERM_N = [3], PERM_K = [4], cta_swizzle = CTASwizzle.Identity))
end

# 3. Row-major D with non-square and square tiles (test/rowmajor-d.jl).
for (kname, kernel) in kernels(; async = false), (BM, BN) in [(128, 64), (64, 128), (64, 64)]
    run_case("row-major D, $(BM)x$(BN), $kname", small_tc(), true,
             (; BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                OP_M = 16, OP_N = 16, OP_K = 16, kernel,
                is_A_col_major = true, is_B_col_major = true, is_D_col_major = false,
                PERM_M = [1, 2], PERM_N = [3], PERM_K = [4], cta_swizzle = CTASwizzle.Identity))
end

# 4. The async kernels with a strided A, which they copy through registers (B with cp.async).
for (kname, kernel) in kernels(), _ in 1:1
    startswith(kname, "async") || continue
    run_case("async with A through registers, $kname", small_tc(), true,
             (; BLOCK_M = 128, BLOCK_N = 16, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 1,
                OP_M = 32, OP_N = 8, OP_K = 16, kernel,
                is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                PERM_M = [1, 2], PERM_N = [3], PERM_K = [4], cta_swizzle = CTASwizzle.Identity))
end

# 5. CTA swizzles with partial strips (test/contraction-configs.jl).
for (kname, kernel) in kernels(), swizzle in [CTASwizzle.HorizontallyTiled{4}, CTASwizzle.VerticallyTiled{4}]
    run_case("swizzle $(swizzle), $kname", small_tc(), true,
             (; BLOCK_M = 128, BLOCK_N = 16, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 1,
                OP_M = 32, OP_N = 8, OP_K = 16, kernel,
                is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                PERM_M = [2, 1], PERM_N = [3], PERM_K = [4], cta_swizzle = swizzle))
end

println()
counts = Dict{String, Int}()
for (_, status) in results
    key = startswith(status, "ERROR") ? "ERROR" : status
    counts[key] = get(counts, key, 0) + 1
end
println("Cases: ", join(["$v $k" for (k, v) in sort(collect(counts))], ", "))
