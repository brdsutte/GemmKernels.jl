# Run kernels continuously and record the GPU's SM clock, power draw and temperature while
# they run, to find out whether kernels run at different clocks (with unlocked clocks, the GPU
# lowers its clock when it reaches its power or temperature limit, and a kernel that executes
# more instructions per second may reach it sooner). Needs no root: nvidia-smi only reads.
#
# usage (on the GPU machine, with nothing else running on the GPU):
#   julia +1.11 --project=tuning tuning/clock-under-load.jl <configs.arrow> [TCs] [out.csv]
#
# For each TC (default 20): the configuration in <configs.arrow> (e.g. the thesis's best) with
# EDGE/INDEXING = pad/default, pad/lean32 and mask/lean32, then cuTENSOR, each run back to
# back for GK_LOAD_SECONDS (default 15) seconds after a pause of GK_LOAD_PAUSE (default 10)
# seconds. Prints, per variant, the time per run, the mean SM clock, power and temperature
# over its last two thirds, and the time per run in SM cycles (time × clock). The raw
# nvidia-smi samples go to <out.csv> (default clock-under-load.csv).

using CUDA, GemmKernels, DataFrames, Adapt, Dates, Printf
import Arrow
include(joinpath(@__DIR__, "wmma-contraction.jl"))

const CONFIGS = DataFrame(Arrow.Table(ARGS[1]))
const TCS = parse.(Int, split(get(ARGS, 2, "20"), ','))
const OUT = get(ARGS, 3, "clock-under-load.csv")
const SECONDS = parse(Float64, get(ENV, "GK_LOAD_SECONDS", "15"))
const PAUSE = parse(Float64, get(ENV, "GK_LOAD_PAUSE", "10"))

# sample the GPU every 100 ms in the background
const FIELDS = "timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"
smi_cmd(fields) = `nvidia-smi -i $(CUDA.deviceid(device())) --query-gpu=$fields --format=csv,noheader,nounits -lms 100`
fields = success(pipeline(`nvidia-smi --query-gpu=$FIELDS --format=csv,noheader`; stdout = devnull, stderr = devnull)) ? FIELDS :
         "timestamp,clocks.sm,power.draw,temperature.gpu"
smi_out = open(OUT, "w")
println(smi_out, fields)
smi = run(pipeline(smi_cmd(fields); stdout = smi_out), wait = false)

function samples_between(t0, t1)
    flush(smi_out)
    rows = []
    for line in eachline(OUT)
        startswith(line, "timestamp") && continue
        parts = strip.(split(line, ','))
        t = tryparse(DateTime, parts[1], dateformat"yyyy/mm/dd HH:MM:SS.sss")
        t === nothing && continue
        t0 <= t <= t1 || continue
        push!(rows, parse.(Float64, parts[2:4]))
    end
    return rows
end

# run `f` back to back for SECONDS; returns the mean time per run over the last two thirds,
# and that window
function load(f)
    f(); synchronize()
    sleep(PAUSE)
    start = time()
    times = Tuple{Float64, Float64}[]     # (end time, seconds per run)
    while time() - start < SECONDS
        n = 20
        t = CUDA.@elapsed for _ in 1:n
            f()
        end
        push!(times, (time(), t / n))
    end
    cutoff = start + SECONDS / 3
    window = filter(x -> x[1] >= cutoff, times)
    t0 = unix2datetime(cutoff) + (now() - now(UTC))     # nvidia-smi prints local time
    t1 = unix2datetime(time()) + (now() - now(UTC))
    return sum(last, window) / length(window), t0, t1
end

report(label, per_run, t0, t1) = begin
    s = samples_between(t0, t1)
    clock = isempty(s) ? NaN : sum(x -> x[1], s) / length(s)
    power = isempty(s) ? NaN : sum(x -> x[2], s) / length(s)
    temp = isempty(s) ? NaN : sum(x -> x[3], s) / length(s)
    @printf("   %-22s %9.1f μs  %6.0f MHz  %5.1f W  %4.1f °C  %8.0f k cycles  (%d samples)\n",
            label, per_run * 1e6, clock, power, temp, per_run * clock * 1e3, length(s))
end

delete!(ENV, "GK_PROBLEMS")
problems = generate_problems()
for tc in TCS
    problem = problems[tc]
    sel = filter(:name => ==(problem.name), CONFIGS)
    nrow(sel) == 1 || (@warn "TC$tc: $(nrow(sel)) configurations; skipped"; continue)
    config = NamedTuple(first(sel))
    println("TC$tc: $(repr_row(config))")
    for (edge, indexing) in (("pad", "default"), ("pad", "lean32"), ("partial", "lean32"))
        variant = merge(config, (; EDGE = edge, INDEXING = indexing))
        data = allocate_data(problem)
        initialize_data(problem, data...)
        reference = Array(calculate_reference(problem, data...))
        params = create_params(variant)
        args = try
            prepare(problem, data...; params...)
        finally
            GemmKernels.Tensors.OVERRIDE_do_override = false
        end
        initialize_data(problem, data...; params...)
        valid = verify(problem, reference, adapt(Array, execute(problem, data...; args...)))
        report("$edge, $indexing" * (valid ? "" : " (INVALID)"), load(() -> execute(problem, data...; args...))...)
        foreach(CUDA.unsafe_free!, data)
        GC.gc(true); CUDA.reclaim()
    end
    data = allocate_data(problem)
    initialize_data(problem, data...)
    baseline_args = prepare_baseline(problem, data...)
    report("cuTENSOR", load(() -> execute_baseline(problem, data...; baseline_args...))...)
    foreach(CUDA.unsafe_free!, data)
    GC.gc(true); CUDA.reclaim()
end

kill(smi)
close(smi_out)
println("nvidia-smi samples written to $OUT")
