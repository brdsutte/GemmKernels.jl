using CUDA
using GemmKernels
using GemmKernels.Tensors.GETTLayout

isdefined(@__MODULE__, :Configs) || include("../configs/configs.jl")

# The vectorised copies of the GETT layouts must not cross from one tensor mode into the next
# (GETTLayout.copy_tile_crossing): such configurations give a ConfigError instead of wrong
# results.

@testset "GETT copies across mode boundaries" begin
    @testcase "copy_tile_crossing" begin
        # 8-element copies along a first dimension made of modes of 12 and 312 elements
        # (strides 97344 and 1: not adjacent in memory), and of 24 elements along the second
        crossing(modT1, strides1, tile) = GETTLayout.copy_tile_crossing(modT1, strides1, (24,), (1,), tile)
        @test crossing((12, 312), (97344, 1), (M = 8, K = 1)) == (1, 8, 12)
        @test crossing((312, 12), (1, 97344), (M = 8, K = 1)) === nothing
        @test crossing((12, 312), (97344, 1), (M = 4, K = 1)) === nothing
        @test crossing((12, 312), (97344, 1), (M = 1, K = 8)) === nothing
        # modes that are adjacent in memory form one run: 12 × 16 = 192 elements
        @test crossing((12, 16), (1, 12), (M = 8, K = 1)) === nothing
        @test crossing((12, 10), (1, 12), (M = 16, K = 1)) == (1, 16, 120)
        # the second dimension
        @test GETTLayout.copy_tile_crossing((16,), (1,), (12, 24), (16, 1000), (M = 1, K = 8)) == (2, 8, 12)
        @test GETTLayout.copy_tile_crossing((16,), (1,), (12, 24), (16, 192), (M = 1, K = 8)) === nothing
        # the lean and masked layouts are checked like the layouts they derive from
        args = (Float16, [12, 312, 24], ([1, 2], [3]), true, false)
        L = GETTLayout.createGETTLayout(args...)
        @test GETTLayout.copy_tile_crossing(L, (M = 8, K = 1)) === nothing      # modes 1 and 2 are adjacent
        args = (Float16, [12, 24, 312], ([1, 3], [2]), true, false)
        for L in (GETTLayout.createGETTLayout(args...),
                  GETTLayout.createGETTLayout(args...; indexing = :lean32),
                  GETTLayout.createGETTLayout(args...; bounds = (3744, 0)))
            @test GETTLayout.copy_tile_crossing(L, (M = 8, K = 1)) == (1, 8, 12)
        end
    end

    if CUDA.functional()
        # D = 1.2.3, A = 2.4.1: the M modes 1 and 2 are A's third and first dimension. With
        # PERM_M = [1, 2], A is copied along mode 1 (extent 12) in strided 8-element copies,
        # which would cross into mode 2; with PERM_M = [2, 1] along mode 2 (extent 312).
        problem = WMMATensorContraction(; name = "1.2.3-2.4.1-4.3", extents = [12, 312, 24, 312],
                                          data_type = Float16, compute_type = Float16,
                                          accumulate_type = Float32, zero_c = true)
        params(perm_m) = (; BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, WARPS_M = 2, WARPS_N = 2,
                            OP_M = 16, OP_N = 16, OP_K = 16, kernel = Kernel.matmul_singlestage,
                            is_A_col_major = true, is_B_col_major = true, is_D_col_major = true,
                            PERM_M = perm_m, PERM_N = [3], PERM_K = [4], cta_swizzle = CTASwizzle.Identity)

        @testcase "copies within a mode are accepted and correct" begin
            data = allocate_data(problem)
            initialize_data(problem, data...)
            reference = Array(calculate_reference(problem, data...))
            args = try
                prepare(problem, data...; params([2, 1])...)
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
            initialize_data(problem, data...; params([2, 1])...)
            @test verify(problem, reference, Array(execute(problem, data...; args...)))
        end

        @testcase "copies across a mode boundary are rejected" begin
            data = allocate_data(problem)
            err = try
                prepare(problem, data...; params([1, 2])...)
                nothing
            catch err
                err
            finally
                GemmKernels.Tensors.OVERRIDE_do_override = false
            end
            @test err isa GemmKernels.ConfigError
            @test occursin("mode boundary", sprint(showerror, err))
        end
    end
end
