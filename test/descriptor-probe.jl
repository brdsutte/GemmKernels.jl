# What does GemmKernels compute for unsupported descriptors when the checks added in
# fix/contraction-api are switched off? Run from the repository root on a GPU machine:
#
#     julia +1.11 --project test/descriptor-probe.jl
#
# The checks are disabled in this Julia session only, by overwriting check_gett_support;
# no source file is changed. Expect a warning about the overwritten method.

using CUDA, GemmKernels, LinearAlgebra
using GemmKernels.Tensors, GemmKernels.Operator

GemmKernels.Tensors.check_gett_support(desc::GemmKernels.Tensors.ContractionDescriptor, β) = nothing

M = N = K = 512
A = CUDA.rand(Float16, M, K)
B = CUDA.rand(Float16, K, N)
C_transposed = CUDA.rand(Float16, N, M) .* Float16(100)
C_float32 = CUDA.rand(Float32, M, N) .* 100f0

function contract(a, c, modeC, β)
    d = CUDA.zeros(Float16, M, N)
    plan = Tensors.ContractionPlan(
        1,
        a, TensorDescriptor(a), [1, 3],
        B, TensorDescriptor(B), [3, 2],
        β,
        c, TensorDescriptor(c), modeC,
        d, TensorDescriptor(d), [1, 2];
        operator = Operator.WMMAOp{16, 16, 16, Float16, Float32},
        computeType = Float16, accumulateType = Float32,
        blockShape = (M = 128, N = 128, K = 32), warpsPerBlock = 8,
        computeWarp = (M = 32, N = 64, K = 16))
    Tensors.contraction!(plan, 1, a, B, β, c, d)
    return Array{Float32}(Array(d))
end

host(x) = Array{Float32}(Array(x))
relerr(x, y) = norm(x - y) / norm(y)
AB = host(A) * host(B)

println("== C with modes [2, 1], D with modes [1, 2]")
D = contract(A, C_transposed, [2, 1], 1)
println("relative error vs. AB + Cᵀ (the correct result): ", relerr(D, AB + permutedims(host(C_transposed))))
println("relative error vs. AB + C as stored:             ", relerr(D, AB + host(C_transposed)))
println("relative error vs. AB alone (C ignored):         ", relerr(D, AB))

println("\n== C is Float32, D is Float16")
try
    D = contract(A, C_float32, [1, 2], 1)
    println("ran; relative error vs. AB + C: ", relerr(D, AB + host(C_float32)))
catch err
    msg = sprint(showerror, err)
    println(typeof(err), ": ", first(msg, 1500))
end
