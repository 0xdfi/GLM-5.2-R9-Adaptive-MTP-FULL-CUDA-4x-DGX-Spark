# Architecture

How the adaptive-MTP controller, the CUDA-graph layer and the four-node topology fit
together, and why each piece is shaped the way it is.

---

## 1. The deployment

Four DGX Spark nodes, **one GB10 per node** (`sm_121a`), joined by a Ray cluster over a
private RDMA/RoCE network.

| Dimension | Value | Notes |
|---|---|---|
| Tensor parallel | **4** | one rank per node, one GB10 per rank |
| Decode context parallel (DCP) | **2** | `--dcp-comm-backend a2a`, `--dcp-kv-cache-interleave-size 1` |
| Pipeline parallel | 1 | |
| Data parallel | 1 | |
| Executor | Ray | rank 0 is the Ray head and hosts the API server |
| Attention backend | `B12X_MLA_SPARSE` | plus `VLLM_USE_B12X_SPARSE_INDEXER=1` |
| MoE backend | `flashinfer_cutlass` | target and draft |
| Model runner | **V2** (`VLLM_USE_V2_MODEL_RUNNER=1`) | the FULL-graph exemption applies only to V2 |
| Quantization (weights) | `compressed-tensors` — Int4/Int8 mix | |
| KV cache dtype | `nvfp4_ds_mla` | NVFP4 KV, DeepSeek-MLA layout |
| `max_model_len` | 520,000 | |
| `kv_cache_memory_bytes` | 8,410,000,000 | → 525,887 physical KV tokens, measured |
| `max_num_seqs` | **exactly 3** | enforced by the launcher, not defaulted |
| `max_num_batched_tokens` | 1,024 | `long_prefill_token_threshold` also 1,024 |
| Scheduling | async scheduling on; decode-aware prefill on; prefix caching on | |
| Speculation | MTP, `num_speculative_tokens=5`, `adaptive_speculative_tokens_window=32` | |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes `[6,12,18]` | nine FULL decode descriptors |

The `--hf-overrides` carry the GB10-specific `index_topk_pattern` and `use_index_cache`
settings that `CosmicRaisins/glm-5.2-gb10` established as a requirement for GLM-5.2
sparse attention on this hardware.

---

## 2. MTP and what "depth" means

GLM-5.2's multi-token-prediction head drafts `K` tokens per scheduler step. The target
model verifies them in a single batch; rejection sampling accepts the longest correct
prefix and stops at the first miss.

Two consequences drive everything below:

1. **A verification batch has `K + 1` query rows per request.** So
   `decode_query_len = depth + num_new_sampled_tokens_per_step`, which for GLM MTP is
   `depth + 1`.
2. **Later positions are only reachable if all earlier ones were accepted.** A
   position's *conditional* acceptance rate and its *unconditional* contribution per
   batch are therefore very different quantities, and the controller uses the second.

---

## 3. The controller

`vllm/v1/spec_decode/dynamic/acceptance_length.py`. Taken byte-for-byte from
`CosmicRaisins/glm-5.2-gb10` @ `600848707ce93fe42fedbc9dd4429116696e425d`, apart from a
`from __future__ import annotations` and a provenance header.

### 3.1 The ladder

`VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5`. `K=2` is the floor and the starting rung; `K=5` is
the hard upper bound and must equal `num_speculative_tokens`; `K=3` is unreachable
because it is not a rung.

### 3.2 The gates

| current K | observation | window | decision | reason |
|---|---|---|---|---|
| 2 | head ratio ≥ 0.85 | 32 | → 4 | `probe_k4` |
| 2 | head ratio < 0.85 | 32 | stay 2 | `k2_baseline` |
| 4 | `(acc p2 + acc p3) / batches` ≥ 0.70 | 16 | → 5 | `probe_k5` |
| 4 | tail gain ∈ [0.35, 0.70) | 16 | stay 4 | hold |
| 4 | tail gain < 0.35 | 16 | → 2 | retreat |
| 5 | `acc p4 / batches` ≥ 0.15 | 16 | stay 5 | hold |
| 5 | p4 gain < 0.15, tail gain ≥ 0.35 | 16 | → 4 | `k5_p4_reject` |
| 5 | p4 gain < 0.15, tail gain < 0.35 | 16 | → 2 | retreat |

Baseline windows are 32 scheduler steps; **exploratory windows are half that**, so a
prose→code phase change stops paying for an inherited high `K` within 16 steps.

The 0.85 gate is the *only* head-acceptance gate. K4 and K5 decisions deliberately use
**unconditional marginal tokens earned per draft batch**, not a second whole-prefix
ratio, because a whole-prefix ratio rises mechanically as `K` falls and would create a
feedback loop that rewards retreating.

### 3.3 Stale-frame exclusion

Frames with `async_tokens_to_discard > 0` are excluded from both the controller's
counters and the telemetry samples. A window that counted pre-reset rejection counts
would misreport exactly the ratio the policy decides on.

### 3.4 Scope

The controller's depth is **batch-wide, not per request**. With three concurrent
streams of different character, one stream's acceptance drags the others' depth. The
policy's thresholds were tuned by the upstream community at `max_num_seqs=1`; this
deployment runs 3. That mismatch is the largest untested assumption in the design and
is stated as such rather than papered over.

---

## 4. CUDA graphs

`vllm/v1/worker/gpu/cudagraph_utils.py` and the narrow exemption in
`vllm/config/vllm.py`.

### 4.1 Why naive adaptive depth silently disables graphs

`BatchExecutionDescriptor._is_compatible()` requires an **exact** `uniform_token_count`
match. A `K=2` batch of 3 requests is `num_tokens=9, uniform=3`. A build that only ever
captured `uniform=6` descriptors matches nothing, `dispatch()` returns
`CUDAGraphMode.NONE`, and the step runs eager.

That is not a crash. It is a silent throughput regression on exactly the depths
adaptation exists to reach — which is why the predecessor revision took the safe route
and downgraded `cudagraph_mode` from FULL to PIECEWISE whenever adaptation was enabled.

### 4.2 Coverage by construction

`_init_adaptive_spec_candidates()` enumerates request counts **exactly**
(`1..min(max_num_seqs, 32)`) rather than rounding up to a capture size. The captured set
is therefore exactly

```
{ (d + 1) · n | d ∈ ladder, n ∈ 1..N }
```

which at ladder `2,4,5` and `N = 3` is nine shapes:

| depth | uniform query length | token counts |
|---|---|---|
| 2 | 3 | 3, 6, 9 |
| 4 | 5 | 5, 10, 15 |
| 5 | 6 | 6, 12, 18 |

All nine are ≤ `max_cudagraph_capture_size` (18) and ≤ `max_num_reqs × decode_query_len`
(18).

This is why `max_num_seqs` is **enforced at exactly 3** and not merely defaulted: `N` is
baked into the captured set and into what the load-time assertion checks.

### 4.3 Layering, not merging

The per-depth descriptors are **prepended** to the base candidate ranges; the base
ranges are built unchanged. Merging them (as the pinned community patch does) would
re-partition the token-count buckets and silently remove mixed-batch coverage: a
4-token prefill would resolve into a bucket holding only uniform-5 decode descriptors,
match nothing, and drop to eager instead of getting the PIECEWISE graph it used to get.

### 4.4 One ladder parser

`vllm/v1/spec_decode/dynamic/depth_ladder.py` is the single canonical parser of
`VLLM_ADAPTIVE_SPEC_DEPTHS`, imported by both the scheduler and the CUDA-graph layer.
The pinned community patch parses the variable twice with divergent results — the
scheduler unions `num_speculative_tokens` into its snap points, the graph side does not
— which lets the scheduler pick a depth the graph layer never captured. One parser makes
the schedulable set and the captured set the same object by construction.

### 4.5 Fail closed, at load, with no opt-out

Environment variables are not evidence: `cudagraph_mode` is resolved from the attention
backend's capabilities at load time and can land on PIECEWISE or NONE for reasons no
launch script can see. Three assertions raise `RuntimeError` before any execution:

| Assertion | Catches |
|---|---|
| `_assert_adaptive_spec_graph_coverage()` | resolved mode lacks FULL graphs, or any reachable shape is uncaptured |
| `_assert_adaptive_spec_graphs_possible()` | the two pre-existing early returns in `_init_candidates()` — a falsy/`NONE` `cudagraph_mode`, empty `cudagraph_capture_sizes`, or an empty descriptor set |
| `adaptive_spec_decode_query_lens()` | raises instead of returning `[]` once adaptation is configured and this manager owns the verification shape |

The second and third are not hypothetical holes. `CUDAGraphMode.NONE` is `0` and
therefore **falsy**, so `if not (self.cudagraph_mode and capture_sizes): return` fired
*before* the coverage check — for exactly the mode in which every ladder depth runs
eagerly. And an empty list from `adaptive_spec_decode_query_lens()` reads downstream as
"adaptive is not configured", so returning it for an unrecognised verifier shape failed
**open** while looking like it failed closed. Both now raise.

There is deliberately **no bypass environment variable**. A single container-env var
that can undo every launch guard after all of them have passed is a fail-open path, not
a debugging door.

### 4.6 The exemption itself

`_maybe_override_dynamic_sd_cudagraph_mode()` returns without downgrading **only** when
all three hold: acceptance-length adaptation is on, the batch-size schedule is off, and
the V2 runner is in use. It still downgrades for the V1 runner (no multi-depth
descriptor path, and it reads `num_spec_tokens_to_schedule` unresolved) and for the
batch-size schedule (its `K` comes from the batch size, not the ladder, so the captured
set does not bound it).

### 4.7 The draft speculators

`PrefillSpeculatorCudaGraphManager` is constructed with the same `decode_query_len` and
config as the target manager, so it derives an identical descriptor set and its
`attn_states[desc]` lookup — which reuses the target's captured states — still resolves.
`DecodeSpeculatorCudaGraphManager` is built with `decode_query_len=1` and is excluded by
the `decode_query_len > 1` guard: it drafts one token per step at every depth and has
nothing to follow.

### 4.8 Cost

Nine FULL decode descriptors instead of three. Measured graph-pool memory:

| Configuration | FULL decode shapes | graph capture |
|---|---|---|
| fixed `K=5` predecessor | 3 | 1.01 GiB, 3 s |
| R9 adaptive, 550K profile | 9 | 1.93 GiB, 8 s |
| R9 adaptive, 520K profile (current) | 9 | 2.14–2.25 GiB, 24 s |

On unified-memory nodes with no swap this is the binding constraint. See
[`BENCHMARKS.md`](BENCHMARKS.md) §5.

---

## 5. Telemetry

One `MTP_WINDOW_JSON <json>` line per `VLLM_MTP_INSTRUMENT_WINDOW` scheduler steps
(default 32), schema version 2. Every field is derived from CPU-side counts the
scheduler already holds after verification: **no model forward pass, no device read and
no GPU synchronization is added.**

Fields: `schema`, `window_index`, `configured_max_k`, `depth_ladder`,
`adaptive_enabled`, `active_k`, `active_k_at_window_start`, `controller_previous_k`,
`controller_next_k`, `decision_reason`, `controller_window`, `acceptance_ratchet`,
`verification_steps`, `draft_batches`, `drafts_attempted`, `drafts_accepted`,
`acceptance_ratio`, `acceptance_ratio_denominator`, `mean_attempted_per_batch`,
`mean_accepted_per_batch`, `position_eligible`, `position_accepted`,
`position_conditional_acceptance`, `position_unconditional_gain`, `tail_gain_23`,
`position_4_gain`, `controller_tail_gain_23`, `controller_position_4_gain`,
`output_tokens_emitted`, `window_elapsed_ms`, `context_tokens_min`,
`context_tokens_max`, `selected_k_window`, `selected_k_run`.

### 5.1 Four numbers that are not the same number

| Metric | Definition |
|---|---|
| `acceptance_ratio` | `drafts_accepted / drafts_attempted`. The denominator is draft tokens **actually offered for verification** in the window — not `draft_batches`, not `configured_max_k × draft_batches`. The record names its own denominator in `acceptance_ratio_denominator`. |
| `mean_accepted_per_batch` | `drafts_accepted / draft_batches`. Ranges 0..`active_k`. |
| `position_conditional_acceptance[i]` | `position_accepted[i] / position_eligible[i]`. *Eligible* means the batch drafted ≥ `i+1` tokens **and** accepted every earlier position. |
| `position_unconditional_gain[i]` | `position_accepted[i] / draft_batches`. |

The two policy inputs are `tail_gain_23 = (position_accepted[2] + position_accepted[3]) /
draft_batches` and `position_4_gain = position_accepted[4] / draft_batches`, both
recomputed inside the record from the window's own counts so a window can be audited
without the controller.

### 5.2 The conflation that must not happen

`acceptance_ratio` is **not comparable across arms on its own**. With adaptive depth the
denominator shrinks as `K` falls, so a configuration can post a *higher*
`acceptance_ratio` at a *lower* `mean_accepted_per_batch` and be slower overall. Any
report quoting one of these must quote the denominator and must report
`mean_accepted_per_batch` and `selected_k_run` alongside it. This repository does that
everywhere it quotes a number.

---

## 6. The five-file delta

| File | Pre-image | Change |
|---|---|---|
| `vllm/v1/spec_decode/dynamic/acceptance_length.py` | predecessor post-image | **replaced** with the 2/4/5 ladder controller |
| `vllm/v1/spec_decode/dynamic/depth_ladder.py` | *(new)* | the single ladder parser |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | raw upstream `e232d262` blob | multi-depth FULL descriptors + fail-closed assertions |
| `vllm/config/vllm.py` | predecessor post-image | the V2 FULL-graph exemption |
| `vllm/v1/core/sched/scheduler.py` | predecessor post-image | ladder wiring, per-position counts, telemetry |

`cudagraph_utils.py` is the only file this revision adds to the dependency set; earlier
revisions never touched it, so its pre-image is the exact upstream `e232d262` blob.

The delta is produced by an anchor-based applier
([`../patches/13_r9_adaptive_full_cuda.py`](../patches/13_r9_adaptive_full_cuda.py)),
not hand-carried, and the full 35-file post-image manifest is sealed in
[`../provenance/r9-postimage.sha256`](../provenance/r9-postimage.sha256).

---

## 7. Launcher fail-closed gates

`runtime/start-node.sh` (public template) mirrors the production launcher's refusal
behaviour. It renders **no launch line at all** unless every one of these holds:

| Exit | Condition |
|---|---|
| 2 | `MTP_K < 1` |
| 4 | adaptive disabled, or the adaptive window is not a bare positive integer |
| 5 | `VLLM_ADAPTIVE_SPEC_DEPTHS` does not normalise to exactly `2,4,5` |
| 6 | the ladder's top rung is not `MTP_K`, or `MTP_K != 5` |
| 7 | `ENFORCE_EAGER=1`, or the capture sizes cannot cover every (depth, request count) shape |
| 8 | MTP instrumentation disabled, or a malformed instrumentation window |
| 10 | `MAX_NUM_SEQS` is anything other than exactly `3` |
| 11 | any required placeholder is unset or still a `<...>` template value |

`08` is deliberately rejected as a window value: it is valid bash and invalid JSON.
