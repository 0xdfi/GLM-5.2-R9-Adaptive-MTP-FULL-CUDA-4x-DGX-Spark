# GLM-5.2 R9 — Adaptive MTP (K2→K4→K5) with FULL CUDA graphs, on 4× DGX Spark

**This recipe ships two launch profiles — `fast` and `balanced` — both at C4 concurrency (`max_num_seqs=4`) with FULL CUDA-graph coverage, adaptive MTP K2→K4→K5, and a downloadable ARM64 runtime image. The two profiles trade context capacity against single-request decode speed and aggregate throughput.** Numbers below were measured on the four-node DGX Spark cluster on 2026-07-31 with 4/4 nodes healthy and **0 preemptions** in every leg; see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for denominators and the receipts.

## The two profiles at a glance

`fast` runs decode-context-parallel size 1 (DCP1) — all KV state on one rank group, no cross-rank decode comm — for maximum prefill and aggregate throughput. `balanced` runs DCP2 — KV cache split across two rank groups — to reach a 520K context window at the cost of lower aggregate throughput and a comm layer. Both keep FULL CUDA graphs across the full C1–C4 concurrency range (twelve-shape set `[6,12,18,24]`).

| What it describes | `fast` | `balanced` |
|---|---|---|
| Decode-context parallel (DCP) | 1 (no decode comm) | 2 (KV split across ranks) |
| Max context window (`max_model_len`) | 319,000 tokens | 520,000 tokens |
| Max concurrent requests (`max_num_seqs`) | 4 | 4 |
| KV cache memory per rank | ~10.2 GB | ~8.4 GB |
| **Prefill, 200K cold prompt, 1 request** | **695.1 tok/s** | **602.0 tok/s** |
| **Decode, 1 request — natural prose** | **23.0 tok/s** | not measured |
| **Decode, 1 request — peak (synthetic)** | **33.0 tok/s** | **31.1 tok/s** |
| **Decode, 4 concurrent — aggregate** | **83.4 tok/s** | **71.8 tok/s** |
| Decode, 4 concurrent — per-request range | 23.8–27.7 tok/s | not reported per-lane |
| Preemptions (all legs) | 0 | 0 |
| FULL CUDA-graph coverage, C1–C4 | yes | yes |
| Nodes healthy | 4 / 4 | 4 / 4 |

**Read the table correctly.** `fast` wins on every speed leg — prefill, single-request prose decode, and 4-way aggregate decode — because it carries no DCP comm layer. `balanced`'s sole advantage is the 520K context window: it is the **only** profile that can serve a 500K-class cold prompt. The `balanced` 31.1 tok/s figure is a peak/synthetic single-request decode number, not a natural-prose number; a natural-prose decode leg was not measured on `balanced`. A natural-prose decode was measured on `fast` only (23.0 tok/s). Do not read the 31.1 as "balanced is faster at prose" — it is a different, faster payload type. The 200K prefill figure is a single-request, single-run number on this exact cluster — not a sustained rate and not generalizable. The C4 aggregate is the sum of per-request decode rates at concurrency 4 on a fixed payload, not a normalized benchmark score. The prefill row and the context-window row are different measurements: a 200K prefill leg on `fast` does **not** prove a 319K cold-prompt claim, and `balanced`'s 520K ceiling is the largest proven cold-prompt envelope in this table. Image: `sha256:6d7b06b1…` (release `r13-balanced-fast-c4`).

This is the performance/capacity branch of a line of work that previously shipped
[1M-context GLM-5.2 on 4× DGX Spark](https://github.com/0xdfi/GLM-5.2-1M-4x-DGX-Spark)
and a [655K MTP-k5 QuantTrio runtime](https://github.com/0xdfi/Keys-GLM-5.2-QuantTrio-655K-MTP-k5-4x-DGX-Spark).
Where those releases chased context length, this one chases **speculative-decoding
efficiency without giving up CUDA graphs** — and ships the exact ARM64 runtime image
that is serving it.

Everything below that is presented as a number was measured on real hardware and is
traceable to a receipt. Everything that was *not* measured is called out as not
measured. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the full denominators.

> **Status (2026-07-31).** The launch templates in [`runtime/`](runtime/) and the
> downloadable container image have both been updated to **R13**:
> - **Runtime templates** (`r9.env.example`, `start-node.sh`): two named profiles
>   (`fast`, `balanced`), both at C4 concurrency (`max_num_seqs=4`), with a launcher
>   guard that fixes a DCP1 boot crash and a twelve-shape FULL-graph coverage set
>   (`6,12,18,24`). Documented in [`runtime/README.md`](runtime/README.md).
> - **Container image** (release `r13-balanced-fast-c4`, image
>   `sha256:6d7b06b1…`, 20.3 GB, ARM64): published as a GitHub release. Download with
>   [`scripts/download-image.sh`](scripts/download-image.sh) and verify with
>   [`scripts/verify-image.sh`](scripts/verify-image.sh) before load.
>
> The R13 image's internal build tag is `r9.1-scheduler-liveness-4lane` (its
> `org.glm52.exp1.revision` label); the public release is named R13. The C4
> / twelve-shape behavior is a runtime configuration applied by `start-node.sh`, not an
> image rebuild — the image's baked-in provenance and capability labels therefore still
> describe the C3 nine-shape set, and `verify-image.sh` accounts for this. Measured
> Fast/Balanced performance numbers are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
> The legacy R9 image release remains available as `r9-adaptive-full-bae57bd`.

---

## 1. The headline: adaptive MTP that keeps FULL CUDA graphs

### What multi-token prediction (MTP) costs you

GLM-5.2's MTP draft head proposes `K` tokens per step; the target model verifies them
in one batch and accepts the longest correct prefix. Fixed `K=5` is the usual setting.
It is also a bet: every step pays for five drafted positions whether or not the text
is predictable enough for five to survive verification. On hard text most of that draft
work is thrown away.

### What "adaptive MTP" means here, concretely

The server watches its own acceptance telemetry over a rolling window of scheduler
steps and moves along a **discrete depth ladder — 2, 4, 5** — according to what the
extra draft positions actually earn:

* **It stops wasting draft work on difficult text.** Starting at the ladder floor
  `K=2`, it only probes deeper once the head acceptance ratio clears 0.85 over a
  32-step window. If tail positions stop paying — measured as *unconditional marginal
  tokens per draft batch*, not as a second whole-prefix ratio — it retreats to `K=4`
  or all the way back to the `K=2` floor.
* **It climbs when extra positions are useful.** From `K=4` it probes `K=5` when
  positions 2 and 3 together earn ≥ 0.70 accepted tokens per draft batch; it holds
  `K=5` while position 4 alone still earns ≥ 0.15 per batch.
* **Exploratory windows are half length (16 steps),** so a prose→code phase change
  stops paying for an inherited high `K` within 16 steps instead of 32.
* **`K=2` is a floor** — the controller never descends below it. **`K=3` is
  unreachable**, because it is not a rung on the ladder.

Full gate table: [§6](#6-adaptation-policy-k2--k4--k5).

### Why "while preserving FULL CUDA graphs" is the hard part

The obvious way to build this is what the previous revision (R8) did: enable adaptive
depth and **downgrade `cudagraph_mode` from FULL to PIECEWISE**, because full-graph
coverage for every reachable depth had not been proven.

The mechanism that forces that downgrade is exact-shape dispatch. A decode batch's
uniform query length is `depth + 1` for GLM MTP, so:

| depth | uniform query length | token counts at `max_num_seqs=3` |
|---|---|---|
| 2 | 3 | 3, 6, 9 |
| 4 | 5 | 5, 10, 15 |
| 5 | 6 | 6, 12, 18 |

vLLM's `BatchExecutionDescriptor._is_compatible()` requires an **exact**
`uniform_token_count` match. A `K=2` batch of 3 requests is `num_tokens=9, uniform=3`.
A fixed-`K=5` build has only captured `uniform=6` descriptors — so that batch matches
nothing, `dispatch()` returns `CUDAGraphMode.NONE`, and the step **silently runs
eager**. Adaptive depth on a fixed-`K` graph set does not fail loudly; it quietly
stops using CUDA graphs on exactly the depths adaptation exists to reach.

R9 removes the downgrade **only after removing the premise**:

1. **Coverage by construction.** The CUDA-graph manager enumerates request counts
   exactly (`1..min(max_num_seqs, 32)`) rather than rounding up to a capture size, so
   the captured set is exactly `{(d+1)·n | d ∈ ladder, n ∈ 1..N}` — the nine shapes
   above, a set you can check by hand.
2. **No collateral damage to mixed batches.** The per-depth descriptors are
   *prepended* to the base candidate ranges rather than merged into them, so every
   pre-existing mixed-batch PIECEWISE lookup stays reachable.
3. **One ladder parser, not two.** The scheduled depth set and the captured
   descriptor set are the same object by construction, so the scheduler cannot select
   a depth the graph layer never captured.
4. **Fail closed at load, with no opt-out.** Three load-time assertions raise
   `RuntimeError` *before any execution* if the resolved mode is not FULL, if any
   reachable shape is uncaptured, or if the descriptor set came out empty. There is
   no environment variable that can bypass them.

**Measured on the live four-node runtime**, every rank logs:

```
Acceptance-length adaptive speculative decoding is keeping
cudagraph_mode=FULL_AND_PIECEWISE: the V2 runner captures a FULL decode graph
for every depth on the adaptive ladder.

Adaptive speculative depth: FULL CUDA graph coverage verified for query lengths
[3, 5, 6] across request counts 1..3.
```

with **zero** PIECEWISE downgrades, **zero** uncaptured-shape warnings and **zero**
eager fallbacks. That is the whole claim, and it is a log line, not a design intent.

### What it costs

Nine FULL decode descriptors instead of three. Graph-pool memory measured at
**2.14–2.25 GiB per rank** on the current 520K profile, against **1.01 GiB** for the
fixed-`K=5` predecessor. On unified-memory GB10 nodes with no swap that is not free —
see [§8](#8-caveats-read-these) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## 2. Architecture and artifact matrix

| Layer | Value |
|---|---|
| Model | GLM-5.2, `QuantTrio/GLM-5.2-Int4-Int8Mix` (compressed-tensors, Int4/Int8 mix) |
| Serving engine | vLLM fork `local-inference-lab/vllm`, upstream base commit `e232d262369b8c918cf478a7a96a0fcf8127cf65` |
| Sparse attention | `B12X_MLA_SPARSE` backend (`lukealonso/b12x` 0.30.x) + `VLLM_USE_B12X_SPARSE_INDEXER=1` |
| MoE backend | `flashinfer_cutlass` |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=1` |
| Parallelism | TP4 · **DCP2** (decode context parallel, `a2a` comm backend) · PP1 · DP1 |
| Nodes | 4 × DGX Spark (GB10, `sm_121a`), **one GB10 per node**, Ray executor |
| Interconnect | RoCE / NCCL over the node's RDMA HCAs |
| KV cache dtype | **`nvfp4_ds_mla`** — NVFP4 KV cache (see [§5](#5-nvfp4-is-the-kv-cache-not-the-weights)) |
| API context | **520,000** tokens |
| Physical KV pool | **525,887** tokens (`kv_cache_memory_bytes=8410000000`) |
| Concurrency | `max_num_seqs=3` (enforced exactly — it defines the nine-shape coverage set) |
| Speculation | MTP, `num_speculative_tokens=5`, ladder `2,4,5`, window 32 |
| CUDA graphs | `FULL_AND_PIECEWISE`; capture sizes `[6,12,18]`; nine FULL decode descriptors |
| Image | Linux **ARM64**, 20,342,958,503 B, **no model weights embedded** |

Artifacts:

| Artifact | Where | Notes |
|---|---|---|
| Runtime container image | [GitHub release `r9-adaptive-full-bae57bd`](https://github.com/0xdfi/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark/releases/tag/r9-adaptive-full-bae57bd) | ARM64 only; weights not included; split archive with SHA-256 checksums |
| Model weights | [`QuantTrio/GLM-5.2-Int4-Int8Mix`](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix) | download separately, accept upstream terms |
| Launch templates | [`runtime/`](runtime/) | placeholders only; fail closed |
| Verification | [`scripts/verify-image.sh`](scripts/verify-image.sh), [`scripts/smoke-openai.py`](scripts/smoke-openai.py) | |
| Adaptive-MTP delta | [`patches/13_r9_adaptive_full_cuda.py`](patches/13_r9_adaptive_full_cuda.py) | anchor-based applier |
| Sealed file manifest | [`provenance/r9-postimage.sha256`](provenance/r9-postimage.sha256) | 35 files, verifiable against the image |

---

## 3. Download and run

### 3.1 Download and load

The exact Docker archive is published as five sub-2 GiB assets on the
[`r9-adaptive-full-bae57bd` release](https://github.com/0xdfi/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark/releases/tag/r9-adaptive-full-bae57bd).
The downloader resumes interrupted transfers, checks every part, reconstructs the
archive, and verifies its full SHA-256:

```bash
./scripts/download-image.sh
zstd -d -c image-download/glm52-r9-adaptive-full-bae57bd.oci.tar.zst | docker load
./scripts/verify-image.sh glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd
```

Run the download/load procedure on each DGX Spark node that needs the local image.
The image is **`linux/arm64` only** and will not run on x86_64. The archive contains
the exact image whose Docker config ID is
`sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39`.

`verify-image.sh` checks architecture, OS, the expected `org.glm52.exp1.*` provenance
labels, the adaptive-MTP capability string, and the absence of model weights. A future
GHCR mirror would have a registry manifest digest distinct from the local Docker image
config ID; this release does not pretend those hashes are interchangeable.

### 3.2 Prerequisite: download the model yourself

**The image contains no model weights.** It is ~20 GB of CUDA/vLLM/b12x runtime and
nothing else. You must obtain GLM-5.2 separately and accept the upstream licence and
any gating on the model page:

```bash
# Requires: huggingface_hub, and acceptance of the upstream model terms.
hf download QuantTrio/GLM-5.2-Int4-Int8Mix --local-dir "${MODEL_DIR:?set MODEL_DIR}"
```

* Quantized weights: [`QuantTrio/GLM-5.2-Int4-Int8Mix`](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
* Base model: [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2)

Read and accept the licence terms on those pages. This repository grants you no rights
to the weights. The runtime is launched with `--trust-remote-code`, which executes
model-repository Python — see [`SECURITY.md`](SECURITY.md) before pointing it at
anything you did not vet.

### 3.3 Four-node topology

One GB10 per node; rank 0 is the Ray head and hosts the OpenAI-compatible API server.

```
            ┌──────────────────────────────────────────────┐
            │  NODE 0  (Ray head + vLLM API server)        │
            │  GB10 ×1   ·   TP rank 0   ·   DCP group 0   │
            └───────┬──────────────────────────────────────┘
                    │  RoCE / NCCL   (RDMA HCAs, one NIC per node)
   ┌────────────────┼────────────────┬────────────────┐
┌──┴──────────┐ ┌───┴─────────┐ ┌────┴────────┐
│ NODE 1      │ │ NODE 2      │ │ NODE 3      │
│ GB10 ×1     │ │ GB10 ×1     │ │ GB10 ×1     │
│ TP rank 1   │ │ TP rank 2   │ │ TP rank 3   │
└─────────────┘ └─────────────┘ └─────────────┘

TP=4 across the four GB10s · DCP=2 (decode context parallel, a2a) · PP=1
```

### 3.4 Launch

Templates live in [`runtime/`](runtime/) and **fail closed**: they refuse to render a
launch line unless every placeholder has been replaced with a real value. Copy the env
example, fill it in, and run the launcher on each node.

```bash
cp runtime/r9.env.example runtime/r9.env
$EDITOR runtime/r9.env          # replace every <...> placeholder

# On the head node (rank 0):
NODE_ROLE=head ./runtime/start-node.sh runtime/r9.env

# On each worker node (ranks 1..3):
NODE_ROLE=worker ./runtime/start-node.sh runtime/r9.env
```

Render without launching anything:

```bash
RENDER_ONLY=1 NODE_ROLE=head ./runtime/start-node.sh runtime/r9.env
```

Then smoke-test:

```bash
python3 scripts/smoke-openai.py --base-url "http://<HEAD_NODE_IP>:<API_PORT>"
```

which checks `/health`, `/v1/models` (expecting `max_model_len = 520000`) and one
deterministic `temperature=0` generation.

> The templates bind the API to a **private address by default** and set no
> authentication. Do not expose this endpoint to an untrusted network. See
> [`SECURITY.md`](SECURITY.md).

---

## 4. The nine shapes, in one place

`decode_query_len = depth + num_new_sampled_tokens_per_step`, which is `depth + 1` for
GLM MTP. At `max_num_seqs=3` with ladder `2,4,5`:

| depth | uniform query length | captured FULL decode token counts |
|---|---|---|
| **2** | 3 | **3, 6, 9** |
| **4** | 5 | **5, 10, 15** |
| **5** | 6 | **6, 12, 18** |

All nine are ≤ `max_cudagraph_capture_size` (18) and ≤ `max_num_reqs × decode_query_len`
(18). The fixed-`K=5` predecessor captured only the bottom row.

`max_num_seqs` is **enforced at exactly 3** by the launcher, not merely defaulted: `N`
is baked into what the load-time coverage assertion checks and into the nine shapes
this build was qualified on. A different `N` is a different runtime and must be
re-qualified.

Coverage is asserted three ways: by construction, by tests that execute the shipped
descriptor code, and at runtime by fail-closed load-time assertions.
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has the details.

---

## 5. NVFP4 is the KV cache, not the weights

`--kv-cache-dtype nvfp4_ds_mla` applies **NVFP4 to the KV cache** in the DeepSeek-MLA
layout. It is what makes a half-million-token KV pool fit in four GB10s' unified
memory: the physical pool is 525,887 tokens inside `kv_cache_memory_bytes=8410000000`,
which works out to ~15.99 KiB per logical token per rank.

**Model weights are not NVFP4.** They are `QuantTrio/GLM-5.2-Int4-Int8Mix` —
an Int4/Int8 mixed quantization served through vLLM's `compressed-tensors` path
(`--quantization compressed-tensors`). The two quantizations are independent; conflating
them is the single most common misreading of this stack.

`kv_cache_memory_bytes` and `max_model_len` are also **a matched pair, not two knobs**.
The API context must sit safely below the capacity the byte budget actually buys, and
the byte budget must leave enough unified memory for R9's graph capture. The launcher
therefore accepts named profiles rather than arbitrary values:

| profile | `max_model_len` | `kv_cache_memory_bytes` | physical KV | status |
|---|---|---|---|---|
| `520k` | 520,000 | 8,410,000,000 | 525,887 tokens (measured) | **current live profile** |
| `550k` | 550,000 | 8,850,000,000 | 553,455 tokens (measured) | prior envelope; see caveats |

---

## 6. Adaptation policy (K2 → K4 → K5)

Baseline window **32** scheduler steps; exploratory windows **16**.

| current K | observation over the window | decision | reason code |
|---|---|---|---|
| 2 | head acceptance ratio **≥ 0.85** over 32 steps | probe **K4** | `probe_k4` |
| 2 | head acceptance ratio < 0.85 | stay **K2** | `k2_baseline` |
| 4 | `(accepted p2 + accepted p3) / draft_batches` **≥ 0.70** over 16 steps | probe **K5** | `probe_k5` |
| 4 | tail gain ∈ **[0.35, 0.70)** | stay **K4** | hold |
| 4 | tail gain **< 0.35** | fall to **K2** | retreat |
| 5 | `accepted p4 / draft_batches` **≥ 0.15** over 16 steps | stay **K5** | hold |
| 5 | p4 gain < 0.15 **and** tail gain ≥ 0.35 | fall to **K4** | `k5_p4_reject` |
| 5 | p4 gain < 0.15 **and** tail gain < 0.35 | fall to **K2** | retreat |

Notes that matter:

* `K=2` is a floor; the controller never goes below it.
* `K=3` is unreachable — it is not a rung.
* The 0.85 gate is the **only** head-acceptance gate. The K4/K5 decisions use the
  *unconditional marginal tokens* the extra positions earn per draft batch, not a
  second whole-prefix acceptance ratio. Those are different numbers and behave
  differently as `K` moves.
* `num_speculative_tokens = 5` is a hard upper bound; the ladder's top rung must equal
  it, or the launcher refuses to start.

The scheduler emits one `MTP_WINDOW_JSON` telemetry record per window with the full
per-position accounting, derived entirely from CPU-side counts it already holds — **no
extra forward pass, no device read, no GPU synchronization**.

---

## 7. Measured results

Full methodology, denominators and receipts: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### 7.1 Long-context capacity — 520K profile, live R9, one isolated request

| Metric | Value |
|---|---|
| Cold prompt (exact, from the live `/tokenize` endpoint) | **500,000 tokens** |
| Prefix-cache hit delta | 0 (unique first-block nonce) |
| Output | 128 tokens, `ignore_eos=true`, finish reason `length` |
| Prefill | 500,000 KV tokens in 972.400 s = **514.192 tok/s** |
| Decode | **33.430 tok/s** |
| TTFT | 973.196 s (server metrics) |
| End to end | 977.245 s (server metrics) |
| Max KV occupancy | **95.107 %** |
| Preemptions / waiting requests | **0 / 0** |
| Restart, rollback, GPU or driver fatal event | **none** |
| Health after completion | HTTP 200 |
| Profile | 520,000 API context · 525,887 physical KV tokens · DCP2 · C3 |

At 500K context the telemetry window selected **K5 for all 32 steps**, with
**82.86 %** draft-token acceptance and a mean of **4.143** accepted draft positions per
batch. No PIECEWISE, NONE, `FULL_DECODE_ONLY`, eager or fixed-K fallback markers
appeared.

> **This is one isolated long-context test at concurrency 1.** `33.430 tok/s` is
> *that measurement*, not a general decode-speed claim for this stack across
> workloads. Do not quote it as one.

### 7.2 Controller movement — earlier short live run

From the earlier R9 live mission (a shorter run on the 550K profile, ended by a host
OOM unrelated to controller logic — see [§8](#8-caveats-read-these)):

| Selected depth | scheduler steps |
|---|---|
| **K2** | **177** |
| **K4** | **64** |
| **K5** | **783** |

| Quantity | Value |
|---|---|
| Measured average selected K | **4.256** |
| Fixed-`K5` comparator's measured average K | **4.965 – 5.000** |
| **Draft positions avoided vs fixed K5** | **14.9 %** |
| Draft batches / drafted / accepted | 2,162 / 9,202 / 6,210 |
| Accepted per drafted token | 0.675 |
| Mean accepted per draft batch | 2.872 |
| FULL graph coverage | nine shapes — 3/6/9, 5/10/15, 6/12/18 |
| PIECEWISE downgrades / eager fallbacks | **0 / 0** |

Controller transitions carried the policy's own reason codes (`probe_k5`,
`k5_p4_reject`, …) and vLLM's native `SpecDecoding` metrics independently show the same
K2 → K4 → K5 → K4 → K5 shape across 21 intervals.

**14.9 % fewer draft positions is a measurement of speculative work avoided. It is not
a decode-speed measurement and must not be read as one.** No matched K2-vs-K5
decode-rate comparison exists yet; the harness for it is built but has never produced a
valid paired sample.

---

## 8. Caveats — read these

1. **This is a four-node, one-GB10-per-node, ARM64 deployment.** The image is
   `linux/arm64` and targets GB10 / `sm_121a`. Nothing here has been validated on any
   other topology, GPU or architecture.
2. **`33.43 tok/s` is one isolated long-context test**, at concurrency 1, at 500K
   context. It is not an across-workload speed claim.
3. **The physical KV pool is at the memory edge.** 520,000 is the API ceiling;
   500,000 is the largest cold prompt actually proven. During that test the head node's
   minimum available host memory was **484.6 MiB**, and the floor occurred 37 s into
   prefill at 4.11 % KV occupancy — an activation/workspace transient, not final KV
   fill. There is no evidence-backed room for a meaningful KV increase at
   `max_num_batched_tokens=1024`. These are unified-memory nodes with `SwapTotal: 0`,
   so a transient spike is an immediate kill, not a slowdown.
4. **The FULL-graph coverage set costs memory.** ~+0.92 GiB per rank versus the fixed-`K`
   predecessor on the 550K profile; 2.14–2.25 GiB of graph pool on the current profile
   versus 1.01 GiB. An earlier R9 canary on the 550K profile was OOM-killed by the Linux
   kernel on the head node for exactly this reason. The 520K profile exists to hand
   ~440 MB of unified memory per rank back to the host, and is what is live now.
5. **The container excludes model weights.** You must accept and download GLM-5.2
   separately.
6. **NVFP4 applies to the KV cache, not to the QuantTrio model weights.**
7. **The controller's depth is batch-wide, not per request.** With three concurrent
   streams of different character, one workload's acceptance drags the other's depth.
   The community policy this ladder comes from was tuned at `max_num_seqs=1`; this
   deployment runs `max_num_seqs=3`. This remains the largest untested assumption.
8. **Depth-transition cost is unmeasured.** Switching depth switches graph. A policy
   that oscillates could in principle pay more in transitions than it earns; the
   `selected_k_run` telemetry field is what would show it.
9. **Acceptance ratios are not comparable across arms on their own.** With adaptive
   depth the denominator shrinks as `K` falls, so a configuration can post a *higher*
   acceptance ratio at a *lower* mean-accepted-per-batch and be slower. Always quote
   the denominator, `mean_accepted_per_batch` and `selected_k_run` together.

---

## 9. Provenance

This image is a **five-file source delta** on top of a long chain of other people's
work. Full credit and licence labels: [`ATTRIBUTION.md`](ATTRIBUTION.md).

| | |
|---|---|
| Internal build tag | `glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd` |
| Image ID (local Docker config ID) | `sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39` |
| Architecture / OS | `arm64` / `linux` |
| Size | 20,342,958,503 bytes |
| Source commit label | `bae57bd87b03b7c802ca391064996ec27a02d2bb` |
| vLLM upstream base | `e232d262369b8c918cf478a7a96a0fcf8127cf65` |
| Pinned adaptive-MTP source | `CosmicRaisins/glm-5.2-gb10` @ `600848707ce93fe42fedbc9dd4429116696e425d` |
| Sealed file manifest | 35 files, [`provenance/r9-postimage.sha256`](provenance/r9-postimage.sha256) |
| Model weights in image | **none** |

The five files R9 moves against its parent image:

| File | Change |
|---|---|
| `vllm/v1/spec_decode/dynamic/acceptance_length.py` | replaced with the 2/4/5 ladder controller |
| `vllm/v1/spec_decode/dynamic/depth_ladder.py` | **new** — the single canonical ladder parser |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | multi-depth FULL descriptors + fail-closed coverage assertions |
| `vllm/config/vllm.py` | the narrow V2-runner FULL-graph exemption |
| `vllm/v1/core/sched/scheduler.py` | ladder wiring, per-position counts, `MTP_WINDOW_JSON` telemetry |

See [`docs/BUILD.md`](docs/BUILD.md) for what can and cannot currently be rebuilt from
public sources — the honest answer is **not all of it**, and the reasons are written
down.

### Prior releases in this line

* [github.com/0xdfi/GLM-5.2-1M-4x-DGX-Spark](https://github.com/0xdfi/GLM-5.2-1M-4x-DGX-Spark)
* [huggingface.co/0xdfi/GLM-5.2-1M-context-NVFP4-4x-DGX-Spark](https://huggingface.co/0xdfi/GLM-5.2-1M-context-NVFP4-4x-DGX-Spark)
* [github.com/0xdfi/Keys-GLM-5.2-QuantTrio-655K-MTP-k5-4x-DGX-Spark](https://github.com/0xdfi/Keys-GLM-5.2-QuantTrio-655K-MTP-k5-4x-DGX-Spark)
* [huggingface.co/0xdfi/GLM-5.2-QuantTrio-Abliterated](https://huggingface.co/0xdfi/GLM-5.2-QuantTrio-Abliterated)

---

## 10. Documentation map

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | controller internals, graph dispatch, fail-closed design, telemetry schema |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | every measured number, its denominator, and what is *not* measured |
| [`docs/BUILD.md`](docs/BUILD.md) | exact image provenance and public-rebuild reality |
| [`docs/IMAGE.md`](docs/IMAGE.md) | release archive, checksums, image identity, verification commands |
| [`ATTRIBUTION.md`](ATTRIBUTION.md) | upstream credit and licences |
| [`NOTICE.md`](NOTICE.md) | required notices, including NVIDIA's |
| [`SECURITY.md`](SECURITY.md) | threat model, `--trust-remote-code`, network defaults |
| [`IMAGE_AUDIT.md`](IMAGE_AUDIT.md) | pre-publication audit of the image for secrets and private data |
| [`VALIDATION.md`](VALIDATION.md) | what was run to validate this repository, with real output |
| [`PUBLISHING.md`](PUBLISHING.md) | exact publication runbook |

## 11. Licence

Apache-2.0 for the material authored in this repository — see [`LICENSE`](LICENSE).
Derived files retain their upstream licences; the container image carries NVIDIA and
third-party terms. Read [`NOTICE.md`](NOTICE.md) before redistributing anything.
