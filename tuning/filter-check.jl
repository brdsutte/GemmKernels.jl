# Checks the events filter of the tuner's first measurement (GK_FILTER=events): how event times
# compare with the profiler's, per configuration, from the "Filter check" lines that the
# measurement workers log with GK_FILTER=check (unbiased: every configuration is measured both
# ways) or GK_FILTER=events (only the configurations the filter let through).
#
# A configuration is wrongly filtered out only if its event time exceeds GK_FILTER_FACTOR (1.5)
# times the best time so far while its profiler time does not: the ratio events / profiler must
# then exceed 1.5 for a configuration as fast as the best, and more for slower ones.
#
# Usage: julia tuning/filter-check.jl [LOG...]   (default: tuning/tuning.log.*)
#        GK_FILTER_CHECK_LIMIT=1.4: exit with an error if a fully measured configuration's ratio
#        exceeds it

using Printf, Statistics

files = isempty(ARGS) ? filter(f -> startswith(f, "tuning.log."), readdir(@__DIR__)) .|> (f -> joinpath(@__DIR__, f)) : ARGS
re = r"Filter check: events ([-+\d.eE]+) s, profiler ([-+\d.eE]+) s, ratio ([-+\d.eE]+)( \(slow)?"
full = Float64[]; slow = Float64[]; profiler_times = Float64[]
for f in files, line in eachline(f)
    m = match(re, line)
    m === nothing && continue
    push!(m[4] === nothing ? full : slow, parse(Float64, m[3]))
    m[4] === nothing && push!(profiler_times, parse(Float64, m[2]))
end
isempty(full) && isempty(slow) && error("no \"Filter check\" lines in $(join(files, ", "))")

function report(label, r)
    isempty(r) && return
    q(p) = quantile(r, p)
    println(@sprintf("%-46s %6d   min %.3f  1%% %.3f  median %.3f  99%% %.3f  max %.3f   >1.2: %d  >1.5: %d",
                     label, length(r), minimum(r), q(0.01), median(r), q(0.99), maximum(r),
                     count(>(1.2), r), count(>(1.5), r)))
end
println("ratio of event time (faster of 2 runs) to profiler time, per configuration:")
report("measured in full (profiler: minimum of >= 3)", full)
report("slow (profiler: a single measurement)", slow)
if !isempty(full)
    # short kernels are the most sensitive to launch overhead in the events
    short = full[profiler_times .< 50e-6]
    report("measured in full, profiler time < 50 us", short)
end

limit = get(ENV, "GK_FILTER_CHECK_LIMIT", "")
if !isempty(limit) && !isempty(full) && maximum(full) > parse(Float64, limit)
    println("FAILED: a ratio exceeds $limit")
    exit(1)
end
