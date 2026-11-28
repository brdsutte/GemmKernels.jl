# Compares search orders from tuning runs with equal budgets (tuning/compare-orders.sh):
# for every TC, the best time each run found after a given time, relative to the best time any
# run found for that TC, and the time each run needed to come within 5% of it.
#
# Usage: julia --project=tuning tuning/compare-orders.jl DIR [OUT.csv]
#        (DIR holds STRATEGY-tcN-rR.arrow files)

using Arrow, DataFrames, Printf, Statistics

const CHECKPOINTS = [60, 120, 300, 600, 900, 1200, 1800, 2700, 3600]

dir = ARGS[1]
out = length(ARGS) >= 2 ? ARGS[2] : joinpath(dir, "compare-orders.csv")

runs = DataFrame(strategy = String[], tc = Int[], rep = Int[], file = String[])
for f in readdir(dir)
    m = match(r"^([\w+]+)-tc(\d+)-r(\d+)\.arrow$", f)
    m === nothing && continue
    push!(runs, (m[1], parse(Int, m[2]), parse(Int, m[3]), joinpath(dir, f)))
end
isempty(runs) && error("no STRATEGY-tcN-rR.arrow files in $dir")

# best time found by each moment, per run
function curve(df)
    ok = filter(:status => ==("success"), df)
    # results saved before tune.jl recorded found_after where outcomes are decided can carry
    # -Inf (decided, but not processed when the sweep stopped): count them at the end
    last = maximum(filter(isfinite, df.found_after); init = 0.0)
    ok.found_after = [isfinite(t) ? t : last for t in ok.found_after]
    sort!(ok, :found_after)
    (ok.found_after, accumulate(min, ok.time))
end
# the time the cost model took to order the configurations, before the sweep's clock started
# (" - cost model: N configurations in order, in T s" in the run's log): added to its curve
function ordering_time(file)
    log = replace(file, r"\.arrow$" => ".log")
    isfile(log) || return 0.0
    m = match(r"cost model: \d+ configurations in order, in ([\d.]+) s", read(log, String))
    m === nothing ? 0.0 : parse(Float64, m[1])
end
shift(c, dt) = (c[1] .+ dt, c[2])

best_at(c, t) = (i = searchsortedlast(c[1], t); i == 0 ? Inf : c[2][i])
function time_to(c, target)
    i = findfirst(<=(target), c[2])
    i === nothing ? Inf : c[1][i]
end

rows = []
for g in groupby(runs, :tc; sort = true)
    tc = g.tc[1]
    curves = Dict((r.strategy, r.rep) => shift(curve(DataFrame(Arrow.Table(r.file))), ordering_time(r.file))
                  for r in eachrow(g))
    best = minimum(minimum(c[2]; init = Inf) for c in values(curves))
    isfinite(best) || continue
    spent = Dict(k => maximum(c[1]; init = 0.0) for (k, c) in curves)
    println(@sprintf("\nTC%d: best %.4f ms. Best found by time t, relative to it (1.00 = found it):", tc, 1000best))
    println("  " * rpad("run", 14) * join([@sprintf("%8s", "$(t)s") for t in CHECKPOINTS]) * "   to within 5%")
    for k in sort(collect(keys(curves)))
        c = curves[k]
        vals = [best_at(c, t) / best for t in CHECKPOINTS]
        t5 = time_to(c, 1.05 * best)
        println("  " * rpad("$(k[1]) r$(k[2])", 14) *
                join([isfinite(v) ? @sprintf("%8.2f", v) : "       -" for v in vals]) *
                (isfinite(t5) ? @sprintf("   %6.0f s", t5) : "   never"))
        push!(rows, (; tc, strategy = k[1], rep = k[2], best_ms = 1000best,
                       (Symbol("t$t") => v for (t, v) in zip(CHECKPOINTS, vals))...,
                       to_within_5pct_s = t5))
    end
end

df = DataFrame(rows)
open(out, "w") do io
    println(io, join(names(df), ','))
    for r in eachrow(df)
        println(io, join(string.(collect(r)), ','))
    end
end

# summary over TCs: geometric mean of best/best at each checkpoint, per strategy
println("\nGeometric mean over TCs of best found by time t / best of all runs:")
for g in groupby(df, :strategy; sort = true)
    # a run that found nothing yet counts as 10× the best
    vals = [exp(mean(log.(min.(g[!, Symbol("t$t")], 10.0)))) for t in CHECKPOINTS]
    println("  " * rpad(g.strategy[1], 14) * join([@sprintf("%8.3f", v) for v in vals]) *
            "   ($(nrow(g)) runs; a run that found nothing yet counts as 10.00)")
end
println("\nwritten: $out")
