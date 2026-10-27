# cuTile fused small Top-K

Physical-K2 serves logical K2; physical-K4 serves K3 and K4. The fused
producer computes distances without materializing the full distance matrix,
retains a small candidate list, recomputes all retained values in direct FP32,
and orders them by returned metric value and corpus index. K3 refines all four
physical candidates before compaction. Corpus-split direct launches merge the
split-local refined lists with the same ordering.

## Numerical/API contract

FP32 inputs use one-pass TF32 candidate scoring with TF32-consistent corpus
norms. FP16 inputs use native half MMA with FP32 accumulation; norm and retained
value arithmetic widens stored half values before multiplication/subtraction.
Both routes have **approximate membership**: refinement cannot recover omitted
neighbors, and there is no universal recall floor. Neither route is certified
or strict; TF32x3 is not included in this initial change.

Supported metrics are squared L2, Euclidean, cosine and inner product. The
returned distances/similarities are FP32. Final order is ascending distance or
descending inner product, then ascending corpus index, including ties introduced
by the Euclidean square root. Cosine follows Top1's convention: a comparison
involving a zero-norm operand has distance one. Invalid slots use index -1 and
positive infinity. Inputs must be finite with representable metric intermediates;
there is no device-wide input scan or bitwise equivalence promise to a different
reduction implementation.

`brute_force::search_params::allow_approximate` defaults to false. True permits,
but does not force, the cuTile backend. Existing filtered and
unsupported calls fall back to cuVS. Ordinary `all_neighbors::build` forwards
this option through `graph_build_params::brute_force_params::search_params`;
mutual-reachability paths are unchanged. The named all-neighbors API remains
float-only; FP16 graphs use brute-force self-search. Deprecated overloads remain
default-off. C/Python exposure of the option is a separate integration task.
Adding a search-parameter field requires rebuilding consumers; this is not an
old-header/new-library binary compatibility promise.

## Public dispatch

The public policy is capability-based, not a list of benchmark shapes.
On SM120, approximate opt-in calls can use PK2/PK4 for all positive int32-sized
query/corpus extents and feature widths, including tails and unequal query/corpus
row counts. All four supported metrics and both input types use the same rule.
There are no exact-D, exact-row-count, or self-search-only performance exceptions.
When query tiles alone underfill the GPU, a bounded corpus-split launch adds
parallelism and merges the split-local retained lists. This uses query/corpus
tile counts and SM count, not dimension/row-count benchmark exceptions.

Nonaligned feature widths use fresh zero-padded storage rounded to a 16-byte
row pitch when representable and supported by the device copy-pitch limit.
Graph calls share one packed allocation; independent search packs both inputs.
Allocation, zero-fill, copy, producer, refinement/merge and release belong to
the public call. There is no stale cross-call cache. This is a general layout
optimization, not an exact-D dispatch list; the relaxed launcher remains the
fallback when preparation limits prevent padding. Zero extension preserves the
metric, while different kernel lowering may change approximate membership.

Approximation permission, row-major query/corpus storage, one corpus partition,
an available backend, and valid launcher buffers remain required. Filtered calls,
unsupported metrics/layouts/devices, and default-off calls retain existing cuVS
routing. The launcher may refuse calls that exceed its ABI/buffer limits.

This is not a claim of universal speedup. A targeted 48-cell squared-L2 screen
covering FP32/FP16, K2/3/4, graph and unequal-row search passed output checks and
three independent timing processes on SM120. Public calls beat the existing
row-major route in every pass for 44/48 cells and both tested query layouts for
42/48 cells. Narrow dimensions and some small-query FP32 PK4 cases still lose.
The prior finite 144-cell benchmark remains historical evidence, not a universal
certificate. These results do not justify a shape-by-shape dispatch table.
Padding can substantially expand very narrow rows; allocation errors propagate
normally rather than promising automatic memory-pressure fallback.

## Compilation and launch contracts

FP32 strict/relaxed cubins use Top1's architecture metadata for SM80, SM86/89,
SM90, SM100 and SM120. Compilation/direct-launch coverage is not performance
qualification on those devices. Initial FP16 exports are SM120-only with
TM64/TN128/TK32 geometry. Strict pitch requires D divisible by4 for float or8
for half; both require16-byte base alignment. `strict` and `relaxed` label
alignment ABIs, not accuracy modes. Dtype-specific symbols and caches prevent
cross-type reuse. Producer descriptors/IDs are int32, independent of public
int32/int64 result indices. The native merge uses64-bit row/offset arithmetic
and a bounded, overflow-safe grid. Unsupported try-launch calls return false.

## Tests and reproducible screening

Tests cover retained values, ordering/ties, split boundaries, short corpora,
tails, dtype/cache transitions, guards, invalid calls and both index widths.
Independent-query tests use separate Q/X and nondefault streams. Public tests
check default-off, approximate opt-in/direct agreement where supported, and
fallback otherwise. Backend-only FP16 fixtures explicitly skip when SM120 cuTile
is unavailable; failures on a supported target remain failures. Public fallback
tests are not skipped merely because cuTile is unavailable.

Disabled representative benchmarks are separate from ordinary unit tests.
`cpp/scripts/run_fused_topk_benchmark.py` runs three fresh processes, retaining
raw warmed CUDA-event samples, process variation, sampled direct-FP32 recall,
cuVS membership overlap and retained-value checks. PK2 has12 Gaussian/clustered
cells (D128 all metrics; D32/512 squared L2); PK4 has16 cells (D128 all metrics,
K3/K4). Optional PK2 near-tie input demonstrates approximate membership. Smoke
uses PK2 D32 L2 or PK4 D128 K3 L2.

```sh
python cpp/scripts/run_fused_topk_benchmark.py \
  --binary cpp/build/gtests/NEIGHBORS_TEST --output /path/to/new/evidence-directory
```

Timings are resident public search, including internal allocations and excluding
index build/transfers. This portable runner compares the row-major baseline;
it does not establish fastest-layout performance or capture actual launch symbols.
The route-aware qualification packet additionally compares column-query cuVS,
checks CUPTI symbols outside timing and uses sampled FP32/FP64 references.
Selected-value tolerance is1e-6 +2e-5*abs(reference); membership is a separate
measurement with explicit sample denominators, not an exactness guarantee.
Repeated processes reuse inputs and do not expand independent quality coverage.
Self-excluded recall removes reference self without refilling the logical prefix.
Use the project's managed GPU queue when running these tests in the FNM-ML pilot.
