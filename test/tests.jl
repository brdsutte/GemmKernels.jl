using Distributed, XUnit, Dates
using CUDA, GemmKernels
CUDA.allowscalar(false)

if myid() == 1
    using InteractiveUtils
    @info "Julia details:\n" * sprint(io->InteractiveUtils.versioninfo(io))
    @info "CUDA details:\n" * sprint(io->CUDA.versioninfo(io))
end

# Select test files with the GEMMKERNELS_TESTS environment variable:
#   unset or empty   all tests
#   new              the tests added or repaired by the fix branches and the async-copy
#                    prototype
#   a,b,...          the given files, e.g. GEMMKERNELS_TESTS=async-copy,gett-layout
all_tests = ["tiling.jl", "cta-swizzle.jl", "matmul.jl", "blas.jl", "gett-layout.jl",
             "async-copy.jl", "examples.jl", "contraction.jl", "contraction-configs.jl",
             "rowmajor-d.jl", "kernel-cache.jl", "tuning-overrides.jl", "baseline.jl",
             "pipelined-ng-stages.jl", "split-k.jl"]
new_tests = ["cta-swizzle.jl", "gett-layout.jl", "async-copy.jl", "contraction.jl",
             "contraction-configs.jl", "rowmajor-d.jl", "kernel-cache.jl", "tuning-overrides.jl",
             "baseline.jl", "pipelined-ng-stages.jl", "split-k.jl"]
test_selection = strip(get(ENV, "GEMMKERNELS_TESTS", ""))
test_files = if isempty(test_selection)
    all_tests
elseif test_selection == "new"
    new_tests
else
    [endswith(f, ".jl") ? String(f) : f * ".jl" for f in strip.(split(test_selection, ','))]
end
unknown_tests = setdiff(test_files, all_tests)
isempty(unknown_tests) ||
    error("Unknown test file(s) in GEMMKERNELS_TESTS: $(join(unknown_tests, ", ")). Available: $(join(all_tests, ", "))")
myid() == 1 && @info "Running test files: $(join(test_files, ", "))"

t0 = now()
try
    @testset runner=DistributedTestRunner() "GemmKernels.jl" begin
        for file in test_files
            include(file)
        end
    end
finally
    if myid() == 1
        t1 = now()
        elapsed = canonicalize(Dates.CompoundPeriod(t1-t0))
        println("Testing finished in $elapsed")
    end
end

