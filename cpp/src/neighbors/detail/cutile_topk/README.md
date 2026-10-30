# cuTile fused small Top-K

Physical K2 serves logical K2; physical K4 serves K3 and K4. The producer
computes a tile of distances and retains a small candidate list without
materializing the full query-by-corpus distance matrix. Retained candidates
are recomputed from the original stored coordinates in FP32 and ordered by
returned metric value and corpus index. K3 refines all four physical
candidates before compaction. Corpus-split launches merge the split-local
refined lists using the same ordering.

## Numerical/API contract

FP32 inputs use one-pass TF32 candidate scoring with corpus norms computed from
the same TF32-rounded operands as the dot product. Query-only ranking constants
are omitted: L2 ranks by half the corpus squared norm minus the dot product;
cosine ranks by the negative dot product divided by the corpus norm. This is
not the returned-distance formula. FP16 inputs use native half Tensor Core
products with FP32 accumulation; norm and retained-value arithmetic widens
stored half coordinates before multiplication/subtraction.

Both routes have **approximate membership**. Candidate scoring can omit a
neighbor that direct FP32 or FP64 ranking would retain. Refinement improves
returned values and ordering among retained IDs; it cannot recover an omitted
ID. There is no universal recall floor, exact-membership guarantee, or bitwise
equivalence promise to another reduction implementation. Certified, strict
IEEE-FP32 selection and TF32x3 modes are not included in this contribution.

Supported metrics are squared L2 (`L2Expanded`), Euclidean
(`L2SqrtExpanded`), cosine (`CosineExpanded`) and inner product
(`InnerProduct`). Returned distances/similarities are FP32. Final order is
ascending distance or descending inner product, then ascending corpus index,
including ties introduced by the Euclidean square root. Signed zeros compare
equal. As in Top1, a cosine comparison involving a zero-norm operand has
distance one. Invalid slots use index -1 and positive infinity; these sentinels
are not valid results and are excluded from descending-inner-product ordering.
Inputs must be
finite with representable metric intermediates; the search does not perform
a device-wide finite/overflow scan.

`brute_force::search_params::allow_approximate` defaults to false. True permits,
but does not force, this backend. Ordinary `all_neighbors::build` forwards the
option through `graph_build_params::brute_force_params::search_params`.
Mutual-reachability paths are unchanged. The named all-neighbors API remains
FP32-only; FP16 all-neighbors graphs can use brute-force self-search. Self
points are eligible in brute-force self-search. Deprecated overloads remain
default-off. C/Python exposure of the option is a separate integration task.
Adding a search-parameter field requires rebuilding consumers; this is not an
old-header/new-library binary compatibility promise.

## Public dispatch and scheduling

The public policy is capability-based, not a list of benchmark shapes.
On SM120, approximate opt-in calls can use PK2/PK4 for positive int32-sized
query/corpus extents and feature widths, including tails and unequal
query/corpus row counts. Supported metrics and both input types use the same
capability rule. There are no exact-D, exact-row-count or self-search-only
performance exceptions.

When query tiles alone underfill the GPU, a bounded corpus split adds
parallelism. Split count uses query/corpus tile counts and SM count; it is not
an exact-shape dispatch table. The following SM120 scheduling separates
selection from retained-value refinement where this reduces resource pressure:

| Call | Selection | Refinement |
| --- | --- | --- |
| Split FP32 PK2 | TM64 selection-only producer | TM32 retained-value kernel |
| Split FP32 PK4 | TM64 selection-only producer | TM32 retained-value kernel |
| Split FP16 PK2/PK4 | TM64 selection-only producer | TM32 retained-value kernel |
| Unsplit FP16 PK4 | TM64 producer with a selection witness | TM32 retained-value kernel |
| Other calls | Original fused producer/refinement | Within producer |

Split FP16 PK2 retains the fused producer when its actual padded width is at
most 64. Strict staged selection is used only when the corresponding image
and descriptors are compatible. Both stage handles are checked before any
stage is launched, and the same stream orders selection, refinement and
merge/cast. Unavailable or incompatible staged handles retain the original
fused implementation; this is runtime compatibility fallback, not permission
to omit link-time registrations. Unexpected CUDA/allocation errors propagate.

Nonaligned widths use fresh zero-padded storage rounded to a 16-byte row
pitch when representable and supported by the device copy-pitch limit.
Same-pointer query/corpus data shares one packed allocation; independent
search packs both inputs. Allocation, zero-fill, copy, producer,
refinement/merge and release belong to the public call. There is no hidden
cross-call cache. This is a general layout optimization; the relaxed launcher
remains available when preparation limits prevent padding. Zero extension
preserves the metric, while different kernel lowering may change approximate
membership. Padding can expand very narrow rows substantially.

Approximation permission, row-major query/corpus storage, one corpus
partition, an available backend, and valid launcher buffers remain required.
Filtered calls, unsupported metrics/layouts/devices, default-off calls and
launcher ABI limits retain existing cuVS dispatch. The Top1 path is unchanged.

## Compilation and launch contracts

FP32 strict/relaxed cubins use Top1's architecture metadata for SM80, SM86/89,
SM90, SM100 and SM120. Non-SM120 exports have compile-only coverage in this
qualification. Native458/runtime correctness and performance evidence reported
here is SM120-only, not execution or performance qualification on other GPUs.
Initial FP16 and staged exports are
SM120-only. Original FP16 images use TM64/TN128/TK32; staged images use the
stage-specific geometry above. Strict pitch requires D divisible by 4 for
float or 8 for half, with 16-byte base alignment. `strict` and `relaxed` label
alignment ABIs, not numerical accuracy modes.

Dtype-specific symbols and caches prevent cross-type reuse. Producer IDs are
int32, independent of output index width. Native launch/merge tests cover
int32 and int64 result indices; the public brute-force API returns int64
indices. Merge uses 64-bit row/offset arithmetic and a bounded overflow-safe
grid. Unsupported try-launch calls return false.

## Tests and reproducible screening

Tests cover all four metrics, retained values, ordering/ties, duplicates,
split boundaries, short corpora, tails, dtype/cache transitions, guards,
invalid calls, stage/fallback boundaries and both native index widths.
Independent-query tests use separate Q/X and nondefault streams. Public tests
check default-off, approximate opt-in/direct agreement where supported, and
fallback otherwise. Backend-only FP16 fixtures explicitly skip when SM120
cuTile is unavailable; failures on a supported target remain failures.
Public fallback tests are not skipped merely because cuTile is unavailable.

Disabled representative benchmarks are separate from ordinary unit tests.
`cpp/scripts/run_fused_topk_benchmark.py` runs three fresh processes, retaining
raw warmed CUDA-event samples, process variation, sampled direct-FP32 recall,
cuVS membership overlap and retained-value checks. PK2 has 12 Gaussian/clustered
cells (D128 all metrics; D32/512 squared L2); PK4 has 16 cells (D128 all metrics,
K3/K4). Optional PK2 near-tie input demonstrates approximate membership. Smoke
uses PK2 D32 L2 or PK4 D128 K3 L2.

```sh
python cpp/scripts/run_fused_topk_benchmark.py \
  --binary cpp/build/gtests/NEIGHBORS_TEST --output /path/to/new/evidence-directory
```

Timings are resident warmed public search, including internal allocations and
excluding initial compilation, index build and transfers. This portable
runner compares the same-input-dtype row-major baseline; it does not establish
fastest-layout performance or capture actual launch symbols. Route-aware
qualification additionally measures column-query cuVS against a row-built
corpus, captures actual symbols outside timing and uses sampled FP32/FP64
references. cuTile inputs remain row-major in that comparison.

Selected-value tolerance is 1e-6 + 2e-5*abs(reference). Membership is reported
separately with explicit sample denominators, not an exactness guarantee.
Repeated processes reuse inputs and do not increase independent quality
coverage. Self-excluded recall removes reference self without refilling the
logical prefix. Finite targeted workloads do not establish universal speedup,
other-GPU performance or end-to-end ML acceleration.
