# Summarises the correctness of a sweep run with GK_VERIFY=all (every configuration that ran is
# verified): the wrong results per TC, and for every setting the share of configurations with
# that setting that gave a wrong result, to show which settings the wrong results share.
#
# Usage: julia --project=tuning tuning/verify-report.jl CONFIGS.arrow [OUT_DIR]
#        (OUT_DIR, default the file's directory, receives invalid-configs.csv)

using Arrow, DataFrames, Printf

file = ARGS[1]
outdir = length(ARGS) >= 2 ? ARGS[2] : dirname(abspath(file))
df = DataFrame(Arrow.Table(file))

const RAN = ["success", "slow", "invalid_result"]
ran = df[in(RAN).(df.status), :]
bad = ran[ran.status .== "invalid_result", :]

println("Statuses over all TCs:")
for g in sort(combine(groupby(df, :status), nrow => :n), :n; rev = true) |> eachrow
    println(@sprintf("  %-30s %8d", g.status, g.n))
end

println("\nPer TC: configurations that ran (and were verified), and wrong results:")
for g in groupby(ran, :name; sort = true)
    n = nrow(g); k = count(==("invalid_result"), g.status)
    println(@sprintf("  %-32s %7d ran  %5d wrong (%.1f%%)", g.name[1], n, k, 100k / n))
end

if nrow(bad) == 0
    println("\nNo wrong results.")
else
    println("\nShare of wrong results per setting (only settings with any):")
    for col in [:kernel_str, :SPLIT_K, :INDEXING, :EDGE, :BLOCK_M, :BLOCK_N, :BLOCK_K, :WARPS_M, :WARPS_N,
                :OP_M, :is_A_col_major, :is_B_col_major, :is_D_col_major, :cta_swizzle_str]
        hasproperty(ran, col) || continue
        g = combine(groupby(ran, col), :status => (s -> count(==("invalid_result"), s)) => :wrong, nrow => :ran)
        g = g[g.wrong .> 0, :]
        isempty(g) && continue
        sort!(g, :wrong; rev = true)
        println("  $col: " * join([@sprintf("%s %d/%d (%.0f%%)", r[col], r.wrong, r.ran, 100r.wrong / r.ran) for r in eachrow(g)], ", "))
    end
    out = joinpath(outdir, "invalid-configs.csv")
    cols = filter(c -> !(c in ("status", "time", "found_after")), names(bad))
    open(out, "w") do io
        println(io, join(cols, ','))
        for r in eachrow(bad)
            println(io, join([replace(string(r[c]), ',' => ';') for c in cols], ','))
        end
    end
    println("\n$(nrow(bad)) wrong configurations written to $out")
end
