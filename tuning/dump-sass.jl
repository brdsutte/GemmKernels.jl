# Dump the PTX and SASS of the kernels of chosen configurations, for inspecting the
# generated code (e.g. the address arithmetic of the GETT layouts): by default the thesis's
# best configuration of each given TC, with each address arithmetic of the GETT layouts
# (INDEXING = default, lean32). Compiles only; runs nothing on the GPU.
#
# usage (from the repository root, on the GPU machine):
#   julia +1.11 --project=tuning tuning/dump-sass.jl <thesis configs.arrow> [TCs] [out dir]
#
# TCs defaults to 12,20,22,24; out dir to sass/. Writes tc<N>-<indexing>.sass per
# configuration, and prints its registers per thread and local memory (spills). With
# GK_DUMP_EDGE=pad,partial, each is dumped with padding and with partial edge tiles
# (tc<N>-<indexing>-<edge>.sass); with GK_DUMP_KERNELS (kernel strings as in the tuner), with
# each of those kernels instead of the configuration's own (tc<N>-<kernel>-<indexing>.sass).

using CUDA, GemmKernels, DataFrames
import Arrow

const CONFIGS = ARGS[1]
const TCS = parse.(Int, split(get(ARGS, 2, "12,20,22,24"), ','))
const OUT = get(ARGS, 3, "sass")
const EDGES = filter(!isempty, String.(strip.(split(get(ENV, "GK_DUMP_EDGE", ""), ','))))
const KERNELS = filter(!isempty, String.(strip.(split(get(ENV, "GK_DUMP_KERNELS", ""), ','))))
include(joinpath(@__DIR__, "wmma-contraction.jl"))

function nvdisasm()
    for cand in (Sys.which("nvdisasm"), "/usr/local/cuda/bin/nvdisasm")
        cand !== nothing && isfile(cand) && return cand
    end
    return nothing
end

function dump(problem, config, prefix)
    data = allocate_data(problem)
    params = create_params(config)
    args = try
        prepare(problem, data...; params...)
    catch err
        err isa GemmKernels.ConfigError || rethrow()
        println("  $prefix: rejected ($(sprint(showerror, err)))")
        return
    finally
        GemmKernels.Tensors.OVERRIDE_do_override = false
    end
    plan = args.plan.matmulPlan
    sass = try
        sprint(io -> CUDA.code_sass(io, plan.job))
    catch err
        # fall back to disassembling the compiled image
        image = plan.compiled.image
        tool = nvdisasm()
        image isa Vector{UInt8} && tool !== nothing || rethrow()
        write("$prefix.cubin", image)
        read(`$tool -c $prefix.cubin`, String)
    end
    write("$prefix.sass", sass)
    resources = try
        fun = CUDA.link(plan.job, plan.compiled)
        attrs = CUDA.attributes(fun)
        ", $(attrs[CUDA.FUNC_ATTRIBUTE_NUM_REGS]) registers, $(attrs[CUDA.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]) bytes of local memory"
    catch err
        ", registers unknown ($(sprint(showerror, err)))"
    end
    println("  $prefix: $(count(l -> occursin(r"^\s+[A-Z@]", l), split(sass, '\n'))) lines of SASS$resources")
    foreach(CUDA.unsafe_free!, data)
end

mkpath(OUT)
configs = DataFrame(Arrow.Table(CONFIGS))
delete!(ENV, "GK_PROBLEMS")
problems = generate_problems()
for tc in TCS
    problem = problems[tc]
    sel = filter(:name => ==(problem.name), configs)
    nrow(sel) == 1 || (@warn "TC$tc: $(nrow(sel)) configurations; skipped"; continue)
    thesis = NamedTuple(first(sel))
    println("TC$tc: $(repr_row(thesis))")
    for indexing in ("default", "lean32"), edge in (isempty(EDGES) ? [nothing] : EDGES),
        kernel in (isempty(KERNELS) ? [nothing] : KERNELS)
        variant = merge(thesis, (; INDEXING = indexing))
        edge === nothing || (variant = merge(variant, (; EDGE = edge)))
        kernel === nothing || (variant = merge(variant, (; kernel_str = kernel)))
        name = join(filter(!isnothing, ["tc$tc", kernel, indexing, edge]), '-')
        dump(problem, variant, joinpath(OUT, name))
    end
end
println("written to $OUT/")
