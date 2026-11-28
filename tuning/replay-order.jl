# Replays a search order against an earlier (random) sweep: how quickly would the model's
# order (tuning/model-order.jl) have found the sweep's best configurations, compared to random
# orders of the same configurations?
#
# Only the configurations the sweep tried have known outcomes, so both orders are restricted to
# them: the model's order ranks them by its estimate (with the same spreading of near-duplicates
# and every 10th configuration from a random order, as in the tuner), the random orders are
# random permutations of them. The sweep tried a uniform random sample of the space, so this
# compares the orders on a sparser copy of the space.
#
# The budget counts configurations that were compiled or measured; configurations rejected
# while preparing them (config_error...) cost next to nothing and are not counted.
#
# Usage: julia --project=tuning tuning/replay-order.jl SWEEP.arrow [SWEEP2.arrow...]
#        (GK_REPLAY_TRIALS: random orders per TC, default 200)

using Arrow, DataFrames, Random, Statistics, Printf

include(joinpath(@__DIR__, "model-order.jl"))
using .ModelOrder

const TRIALS = parse(Int, get(ENV, "GK_REPLAY_TRIALS", "200"))
const BUDGETS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]
const RANDOM_EVERY = 10

# the benchmark suite's numbering, and the TCs held out while designing the model
const SUITE = read(joinpath(@__DIR__, "..", "test", "benchmark-suite.json"), String)
const SUITE_NAMES = [m.captures[1] for m in eachmatch(r"\"parseableName\"\s*:\s*\"([^\"]+)\"", SUITE)]
tc_number(name) = something(findfirst(==(name), SUITE_NAMES), 0)
held_out(tc) = tc in 1:8 || tc in 15:18 || tc in 31:48

measured(status) = !(startswith(status, "config_error") || startswith(status, "unknown_error"))
known_time(status, time) = status in ("success", "slow") ? time : Inf

# best time found after each budget, for an order (indices into the TC's rows)
function best_after(order, times, costs)
    out = fill(Inf, length(BUDGETS))
    best = Inf; spent = 0; b = 1
    for i in order
        best = min(best, times[i])
        spent += costs[i]
        while b <= length(BUDGETS) && spent >= BUDGETS[b]
            out[b] = best; b += 1
        end
        b > length(BUDGETS) && break
    end
    out[b:end] .= best
    out
end

# budget needed to come within 5% of the best time in the sweep
function budget_to(order, times, costs, target)
    spent = 0
    for i in order
        spent += costs[i]
        times[i] <= target && return spent
    end
    typemax(Int)
end

# the model's order of the rows: diversified estimate order, every 10th from a random order,
# the rest (including what the model considers invalid) in random order
function model_order(ranked, n, rng)
    rand_order = randperm(rng, n)
    taken = falses(n)
    out = Int[]
    ri = 1; mi = 1; k = 0
    while length(out) < n
        k += 1
        if k % RANDOM_EVERY == 0 || mi > length(ranked)
            while ri <= n && taken[rand_order[ri]]; ri += 1; end
            ri > n && break
            i = rand_order[ri]
        else
            i = ranked[mi]; mi += 1
            taken[i] && continue
        end
        taken[i] = true
        push!(out, i)
    end
    out
end

function replay(df::DataFrame, label)
    println("\n== $label: $(nrow(df)) configurations, kernels $(join(unique(df.kernel_str), ", "))")
    summary = []
    for g in groupby(df, :name; sort = true)
        name = g.name[1]; tc = tc_number(name)
        p = ModelOrder.Problem(name, collect(g.extents[1]))
        rows = [(; name, extents = collect(r.extents), r.BLOCK_M, r.BLOCK_N, r.BLOCK_K, r.WARPS_M,
                   r.WARPS_N, r.OP_M, r.OP_N, r.OP_K, r.kernel_str, r.is_A_col_major,
                   r.is_B_col_major, r.is_D_col_major, PERM_M = collect(r.PERM_M),
                   PERM_N = collect(r.PERM_N), PERM_K = collect(r.PERM_K), r.cta_swizzle_str)
                for r in eachrow(g)]
        scores = [ModelOrder.score(r; problem = p) for r in rows]
        times = known_time.(g.status, g.time)
        costs = Int.(measured.(g.status))
        n = length(rows)
        best = minimum(times)
        isfinite(best) || continue
        target = 1.05 * best

        finite = findall(isfinite, scores)
        ranked = [i for (_, i) in ModelOrder.diversify([(scores[i], merge(rows[i], (; idx = i))) for i in finite], n) |>
                  x -> [(t, c.idx) for (t, c) in x]]

        # diagnostics of the validity rules
        ok = isfinite.(times)
        rejected_ok = count(ok .& .!isfinite.(scores))
        best_rejected = !isfinite(scores[argmin(times)])
        accepted_cfgerr = count((costs .== 0) .& isfinite.(scores))

        rng = MersenneTwister(1)
        M = reduce(hcat, [best_after(model_order(ranked, n, rng), times, costs) for _ in 1:TRIALS])
        R = reduce(hcat, [best_after(randperm(rng, n), times, costs) for _ in 1:TRIALS])
        mb = [budget_to(model_order(ranked, n, rng), times, costs, target) for _ in 1:TRIALS]
        rb = [budget_to(randperm(rng, n), times, costs, target) for _ in 1:TRIALS]
        actual = sortperm(g.found_after)
        A = best_after(actual, times, costs)
        ab = budget_to(actual, times, costs, target)

        # rank correlation of the estimates with the measured times
        both = findall(i -> isfinite(scores[i]) && isfinite(times[i]), 1:n)
        rho = length(both) > 2 ? cor(tiedrank(scores[both]), tiedrank(times[both])) : NaN

        println(@sprintf("\nTC%d %s%s: %d configs, %d measured, %d with a time; best %.4f ms",
                         tc, name, held_out(tc) ? " (held out)" : "", n, sum(costs), count(ok), 1000best))
        println(@sprintf("  model: rejects %d configs with a time%s; accepts %d that failed in preparation; Spearman ρ(estimate, time) = %.2f over %d",
                         rejected_ok, best_rejected ? " INCLUDING THE BEST" : "", accepted_cfgerr, rho, length(both)))
        println("  best found / best in sweep (1.00 = found it), median over $TRIALS orders [10%–90%]:")
        println("  budget  " * join([@sprintf("%11d", b) for b in BUDGETS]))
        fmt(v) = isfinite(v) ? @sprintf("%11.2f", v) : "          -"
        row(lbl, X) = println(rpad("  $lbl", 10) * join([fmt(median(X[b, :]) / best) for b in eachindex(BUDGETS)]) *
                              "\n" * " "^10 * join([isfinite(quantile(X[b, :], 0.9)) ?
                                  @sprintf("%11s", @sprintf("%.2f-%.2f", quantile(X[b, :], 0.1) / best, quantile(X[b, :], 0.9) / best)) :
                                  "          -" for b in eachindex(BUDGETS)]))
        row("model", M)
        row("random", R)
        println(rpad("  sweep", 10) * join([fmt(A[b] / best) for b in eachindex(BUDGETS)]))
        med(v) = (m = median(v); m >= typemax(Int) / 2 ? "never" : string(round(Int, m)))
        println("  measured configs to come within 5% of the best: model $(med(mb)), random $(med(rb)), the sweep's own order $(ab == typemax(Int) ? "never" : ab)")
        push!(summary, (; tc, name, held_out = held_out(tc), model = median(mb), random = median(rb),
                          model50 = median(M[findfirst(==(50), BUDGETS), :]) / best,
                          random50 = median(R[findfirst(==(50), BUDGETS), :]) / best))
    end
    println("\nSummary ($label): measured configs to within 5% (median), and best/best after 50")
    for s in summary
        println(@sprintf("  TC%-3d %-30s %-9s model %6s  random %6s   after 50: model %.2f random %.2f",
                         s.tc, s.name, s.held_out ? "held out" : "",
                         s.model >= typemax(Int) / 2 ? "never" : string(round(Int, s.model)),
                         s.random >= typemax(Int) / 2 ? "never" : string(round(Int, s.random)),
                         s.model50, s.random50))
    end
    summary
end

function tiedrank(v)
    p = sortperm(v); r = similar(v, Float64)
    i = 1
    while i <= length(v)
        j = i
        while j < length(v) && v[p[j+1]] == v[p[i]]; j += 1; end
        r[p[i:j]] .= (i + j) / 2
        i = j + 1
    end
    r
end

for f in ARGS
    replay(DataFrame(Arrow.Table(f)), basename(f))
end
