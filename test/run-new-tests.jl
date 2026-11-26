# Runs the tests added by the fix and prototype branches with the standard Test library,
# without XUnit (which does not work on Julia 1.13 and later), e.g.
#
#     julia --project=tuning test/run-new-tests.jl
#     julia --project=tuning test/run-new-tests.jl async-copy.jl
using Test, CUDA, GemmKernels

# XUnit's @testcase takes the same arguments as Test.@testset
macro testcase(args...)
    esc(Expr(:macrocall, GlobalRef(Test, Symbol("@testset")), __source__, args...))
end

files = isempty(ARGS) ? ["cta-swizzle.jl", "gett-layout.jl", "async-copy.jl",
                         "contraction.jl", "contraction-configs.jl"] : ARGS
@testset "GemmKernels (new tests)" begin
    for f in files
        include(joinpath(@__DIR__, f))
    end
end
