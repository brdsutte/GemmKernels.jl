# A model-guided search order for the tuner

`GK_STRATEGY=model` makes `tune.jl` try configurations in the order of a cost model instead of
in random order (`tuning/model-order.jl`). The measurements, verification and bookkeeping are
the same; only the order changes. Measurement remains a burst measurement, as before.

## Rules, and why they are allowed

The model only uses:

- the contraction: its modes and extents (as for any new contraction);
- the code base: which configurations the kernels accept (the tuner's pre-check), how the
  GETT layouts copy data (which copies are vectorised, strided, or cross a mode boundary), how
  much shared memory each kernel needs, and how a configuration pads or partially covers
  M, N and K;
- the GPU: number of SMs, shared memory and registers per SM, maximum threads and blocks per
  SM, and generic throughput ratios of an Ada/Ampere SM (below).

It does **not** use any measured time of any configuration of any contraction. Past sweeps are
only used afterwards, to evaluate the order (`tuning/replay-order.jl`).

These rules were written on 28 November 2026, after a day of profiling TCCG TC12-14 and
TC19-30 with the thesis configurations. Some of the rules (address arithmetic, partial edge
tiles, shared memory limiting the async kernels) were informed by that work. To keep the claim
about unseen contractions honest, the evaluation reports TC1-8, TC15-18 and TC31-48
separately: contractions that were not studied.

## The model

For each configuration the model estimates a time in arbitrary units; configurations are tried
in increasing order of that estimate. The estimate is the time of the busiest SM:

    time = (blocks on the busiest SM) × (cycles per block) / (latency hiding) × (instruction overhead)

1. **Validity.** Configurations the tuner's pre-check rejects, configurations whose copies
   would cross a tensor mode boundary (`GETTLayout.copy_tile_crossing`), and configurations
   whose estimated registers per thread exceed 255 are never tried first (they come last, in
   random order, as the random fallback reaches them).

2. **Work.** M, N and K are padded exactly as `prepare` pads them (`EDGE = pad`: to multiples of
   the block shape; `EDGE = partial`: only K). Tiles = ⌈M/BLOCK_M⌉ × ⌈N/BLOCK_N⌉ × SPLIT_K, and
   the blocks on the busiest SM = ⌈tiles / SMs⌉.

3. **Cycles per block**, per K-tile, times the K-tiles of the block's part, plus a prologue
   (one K-tile's copies and the latency of global memory) and the epilogue. Per K element:
   - tensor cores: BLOCK_M × BLOCK_N / 256 (multiply-adds per SM per cycle, FP16 with FP32
     accumulation);
   - global memory: 2 × (BLOCK_M + BLOCK_N) bytes / 14 bytes per cycle per SM (L2 bandwidth
     share), each operand divided by its copy efficiency (below);
   - shared memory: warps × 2 × (warp_M + warp_N) bytes / 128 bytes per cycle (fragment loads);
   - the kernel decides how these combine: `singlestage` does not overlap its own copies with
     its computation (compute + memory, of which other resident blocks hide a share), the
     pipelined and async kernels do (the maximum of the three);
   - a K-tile takes at least the latency of global memory (600 cycles, an L2 hit) divided by
     how many K-tiles the kernel loads ahead (1 for `singlestage`, 2 for the pipelined kernels,
     stages − 1 for the async kernels) and by the resident blocks;
   - each K-tile adds its barriers (40 cycles each: 2 for `singlestage` and `pipelined_ng`, 3
     for `pipelined`, 1 for the async kernels);
   - split-K adds the partial tiles' round trip through global memory (8 × BLOCK_M × BLOCK_N
     bytes per part); the epilogue adds the store of D.

4. **Copy efficiency** of A, B and D, from the GETT layout the configuration selects (`PERM_*`,
   `is_*_col_major`), exactly as `createGETTContractionPlan` decides:
   - a strided copy (the copied dimension's first mode is not the tensor's stride-1 mode) loads
     element by element: efficiency 1/4;
   - a vectorised copy has efficiency 1, reduced when the run of memory-adjacent leading modes
     is short (sectors partly used: `bytes / (32 × sectors)` for a run of that many bytes);
   - with partial edge tiles, a leading mode whose extent is not a multiple of 16 elements
     (32 bytes) costs 10% (unaligned columns).

5. **Latency hiding.** Resident blocks per SM = the minimum of the shared-memory limit (the
   kernel's shared memory, as the pre-check estimates it), the register limit (registers per
   thread estimated from accumulators, fragments, staging registers and a base of 40), the
   thread limit and the block limit, and never more than the blocks the SM actually gets.
   Latency hiding = min(1, resident warps / needed warps), with needed warps 16 for
   `singlestage`, 8 for `pipelined`, `pipelined_ng` and the async kernels, and 4 for the
   async kernels with double-buffered fragments.

6. **Instruction overhead.** `INDEXING = default` costs 10% more than `lean32` (64-bit division
   and modulo chains in the address arithmetic). CTA swizzles other than the two `-1` variants
   cost 0.5% per doubling (they only matter for L2 reuse, which the model does not estimate), so
   that the model tries them after the default swizzle of an otherwise equal configuration.

## The order

1. Every valid configuration's structural part (block shape, warps, operator shape, kernel,
   split-K, edge, indexing) is combined with every layout (`PERM_*`, `is_*_col_major`), and the
   best few layouts of each are kept; with the swizzles, all candidates are sorted by estimated
   time.
2. Many candidates have (nearly) equal estimates, e.g. when they differ only in the number of
   stages, a swizzle or a layout the model cannot tell apart. To avoid spending the budget on
   near-duplicates, candidates with the same tile shape (block shape, warps, operator shape)
   are penalised by 3% for every earlier one: the order visits the most promising tile shapes
   first and comes back to their variants later.
3. Every 10th configuration is taken from the original random order instead, as insurance
   against blind spots of the model.
4. When the model's list (at most 200,000 configurations) is exhausted, the tuner continues
   in random order (configurations already tried are skipped, as before).

## Revisions

The rules above (revision 0) were committed before any sweep results were looked at.

**Revision 1**, after inspecting the model's own order (no measurements involved), before any
GPU run with it: with tile shapes as the only groups in step 2, the first 200 configurations
of TC1 and TC13 still covered just 3-4 block shapes, because the model barely tells warp and
operator shapes apart. Step 2 now counts earlier candidates at three levels (the same block
shape, the same tile shape, the same tile shape and kernel), 3% for each: the first 200
configurations then cover 22-23 block shapes and 170-200 tile shapes. `GK_MODEL_GROUP=tile`
restores revision 0.

## Evaluation

1. **Replay** (`tuning/replay-order.jl`) against earlier random sweeps on the RTX 6000 Ada
   (old kernels only: singlestage, pipelined, pipelined_ng; default indexing, padding): both
   orders are restricted to the configurations the sweep tried, and the budget counts compiled
   or measured configurations. This compares the orders on a sparse, uniform sample of the
   space; it cannot judge the very top of the model's order in the full space (e.g. how
   diverse it is), nor the new kernels and options.
2. **GPU runs** (`tuning/compare-orders.sh`, `tuning/compare-orders.jl`): every held-out TC is
   tuned once per strategy with the same time budget, in alternating order, with the full,
   current search space; the measure is the best time found after a given time.
