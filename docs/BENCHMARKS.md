# Benchmarks

Only measured, evidenced results appear here. Where something was not measured, this
document says so instead of estimating. Every number carries its denominator.

**All measurements are from the four-node DGX Spark deployment described in
[`ARCHITECTURE.md`](ARCHITECTURE.md): 4 nodes × 1 GB10 each, ARM64, TP4/DCP2.** Nothing
here generalizes to other hardware, other topologies, other concurrency levels or other
workloads, and none of it is offered as such.

---

## 1. Long-context capacity — 500,000-token cold prompt

**Run:** one isolated `C1` (concurrency-1) chat request against the live R9 runtime on
the 520K profile.
**Date:** 2026-07-26 UTC.
**Image:** `sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39`.
**Profile:** 520,000 API `max_model_len` · 525,887 physical KV tokens ·
`kv_cache_memory_bytes = 8,410,000,000` · DCP2 · `max_num_seqs = 3` ·
`max_num_batched_tokens = 1,024`.

### Method

* Prompt token count taken from the live `/tokenize` endpoint on the templated prompt:
  **exactly 500,000 tokens**.
* Unique first-block nonce; measured **prefix-cache hit delta: 0** — this was a genuine
  cold prefill, not a cache replay.
* Completion capped at **128 tokens with `ignore_eos=true`**, so the arm could not stop
  early and look faster. Finish reason `length`.
* Total sequence usage 500,128 tokens.
* No unload, restart, runtime mutation or rollback during the test.

### Result

| Metric | Value |
|---|---|
| Request outcome | succeeded, finish reason `length` |
| Prefill | 500,000 computed KV tokens in 972.400 s = **514.192 tokens/s** |
| TTFT | 973.196 s (server metrics); 973.473 s client-observed |
| End to end | 977.245 s (server metrics); 977.272 s client-observed |
| **Decode** | **33.430 tokens/s** |
| Max KV-cache occupancy | **95.107 %** |
| Waiting requests | 0 |
| Preemptions | 0 |
| Health after completion | HTTP 200 |
| Restarts / rollbacks / GPU or driver fatal events | none |

Four-node postflight: all four containers still running the expected R9 image with
`OOMKilled=false` and restart count 0; **zero** OOM, killed-process, NVIDIA Xid or
`NV_ERR` events in the kernel logs on any node for the test interval; endpoint HTTP 200
at `max_model_len` 520,000; idle KV usage back to 0.

### Denominators and caveats — read before quoting

* **`33.430 tok/s` is one isolated test.** One request, concurrency 1, at 500K context,
  on this exact hardware and profile. It is not a general decode-speed figure for this
  stack, and quoting it as one would be wrong.
* **`514.192 tok/s` prefill is `computed KV tokens ÷ prefill seconds` for this
  request.** It is not a sustained-throughput number and not a multi-request figure.
* **The 500K figure is a cold-prompt capacity proof**, not a latency target: TTFT was
  ~16 minutes.

### Adaptive / FULL-CUDA behaviour at 490K+ context

The telemetry window covering the 500K decode selected **K5 for all 32 steps**:

| Field | Value |
|---|---|
| `selected_k_window` | `{5: 32}` |
| context range | 500,012 – 500,078 tokens |
| `acceptance_ratio` | **82.86 %** (denominator: draft tokens offered for verification) |
| `mean_accepted_per_batch` | **4.143** accepted draft positions |
| `output_tokens_emitted` | 72 |
| PIECEWISE / NONE / `FULL_DECODE_ONLY` / eager / fixed-K fallback markers | **none** |

That the controller sat at K5 throughout is the expected outcome, not a null result:
at 82.86 % acceptance the tail positions were earning their keep, and the policy holds
K5 exactly when they do.

**Receipt:** `r9-500k-c1-stress-receipt-20260726`.

---

## 2. Controller movement — earlier short live run

**Run:** the earlier R9 live cutover, on the **550K profile**, which served for
3 min 20 s before the head node was OOM-killed by the Linux kernel (§5). The controller
evidence below is from that window and is unaffected by the OOM, which was a host
memory-capacity event, not a controller fault.

### Selected depth, in scheduler steps (32 `MTP_WINDOW_JSON` windows)

| depth | scheduler steps |
|---|---|
| **K2** | **177** |
| **K4** | **64** |
| **K5** | **783** |

Controller transitions carried the policy's own reason codes:

| window | transition | reason |
|---|---|---|
| w5 | K4 → K5 | `probe_k5` |
| w12 | K5 → K4 | `k5_p4_reject` |
| w13 | K4 → K5 | `probe_k5` |
| w17 | K4 → K5 | `probe_k5` |

vLLM's native `SpecDecoding` metrics independently show the same K2 → K4 → K5 → K4 → K5
shape across 21 intervals. The controller starts at the ladder floor (K2) by
construction and climbs on acceptance.

### Accepted / drafted accounting (cumulative over the run, from `/metrics`)

| Quantity | Value |
|---|---|
| draft batches | 2,162 |
| drafted tokens | 9,202 |
| accepted tokens | 6,210 |
| **measured average selected K** | **4.256** |
| accepted per drafted token | 0.675 |
| mean accepted per draft batch | 2.872 |
| **draft positions avoided vs fixed K5** | **14.9 %** |

Per-position unconditional gain (accepted at position ÷ draft batches):

| position | p0 | p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|
| gain | 0.895 | 0.747 | 0.483 | 0.411 | 0.335 |

9,202 drafted over 2,162 batches is a mean scheduled depth of 4.256, which reconciles
with the K2:177 / K4:64 / K5:783 step distribution.

### The fixed-K5 comparator

A matched fixed-`K=5` baseline was captured on the same endpoint before the cutover
(11 cases, 3 measured repetitions each after a discarded warmup, deterministic
tokenized fixtures, `temperature 0 / top_p 1 / top_k −1`, seed pinned, streaming with
`include_usage`, decode rate excluding TTFT). Its **measured** `avg_selected_k` was
**4.965 – 5.000** on every case — checked, not assumed, which is what makes it a valid
fixed-K5 comparator.

**14.9 % is the reduction in scheduled draft positions relative to that comparator.**

### What this does *not* establish

* **It is not a decode-speed measurement.** Avoiding 14.9 % of draft positions is a
  measurement of speculative work avoided. Whether that converts into tokens per
  second, and by how much, is not established by these numbers.
* **No matched K2-vs-K5 decode-rate sample exists.** The A/B/A harness for it is built
  and staged; the R9 arm errored on its first case because the engine had already died,
  and the run was stopped rather than hammering a dead endpoint. A K2 verdict with no
  K2 samples is not a result, and the mixed aggregate is not a substitute for one.
* The fixed-K5 baseline itself was captured on a **shared endpoint carrying external
  traffic** (median 1–2 concurrent requests beyond the harness), which inflates TTFT
  and depresses decode rates on some cases. Per-case observed load was recorded rather
  than hidden. That contamination is why those absolute decode rates are not reproduced
  as headline numbers here.

**Receipt:** `R9_MISSION_RECEIPT_20260726` §7–§8.

---

## 3. FULL CUDA graph coverage — live evidence

From the live 520K-profile server log, on all four ranks:

```
Acceptance-length adaptive speculative decoding is keeping
cudagraph_mode=FULL_AND_PIECEWISE: the V2 runner captures a FULL decode graph
for every depth on the adaptive ladder.

Adaptive speculative depth: FULL CUDA graph coverage verified for query lengths
[3, 5, 6] across request counts 1..3.
```

`[3, 5, 6] × 1..3` is the complete nine-shape set: **3/6/9, 5/10/15, 6/12/18**.

| Check | Result |
|---|---|
| Resolved `cudagraph_mode` | `FULL_AND_PIECEWISE` |
| Capture sizes | `[6, 12, 18]`; `max_cudagraph_capture_size` 18 |
| `enforce_eager` | `False` |
| Coverage line present on ranks | 4 / 4 |
| PIECEWISE downgrades | **0** |
| Uncaptured-shape warnings | **0** |
| Eager / `NONE` / `FULL_DECODE_ONLY` / fixed-K fallback markers | **0** |
| Engine starts | 1 |

Graph capture cost, measured:

| Configuration | FULL decode shapes | capture time | graph pool |
|---|---|---|---|
| fixed `K=5` predecessor | 3 | 3 s | 1.01 GiB |
| R9 adaptive, 550K profile | 9 | 8 s | 1.93 GiB |
| R9 adaptive, 520K profile (current live) | 9 | 24 s | 2.14 – 2.25 GiB |

---

## 4. KV capacity, measured

| Quantity | 550K profile | 520K profile (current) |
|---|---|---|
| `kv_cache_memory_bytes` | 8,850,000,000 | 8,410,000,000 |
| **Measured GPU KV cache size** | **553,455 tokens** | **525,887 tokens** |
| `max_model_len` (API ceiling) | 550,000 | **520,000** |
| Largest cold prompt proven | — | **500,000 tokens** |
| Max concurrency at full context | — | 1.01× |

Measured KV slope: **≈ 15.98 KiB per logical token per rank** (`8.85 GB ÷ 553,455`
tokens = 15,990.46 B/token). The 520K profile hands ~440 MB of unified memory per rank
back to the host relative to the 550K profile.

**Explicitly unavailable — no counter exists and none was invented:**

* **Draft-KV write/allocation counters.** vLLM exposes no per-draft KV allocation metric
  on this build. Adaptive depth saves speculative draft *work* and the KV activity
  associated with it; it does **not** shrink the preallocated main KV pool, which is
  measurably unchanged. Any "total KV saving" claim would be fabricated.
* **Pure prefill throughput as a general rate.** Through this endpoint TTFT also
  contains queueing, tokenization, scheduling and the first decode step, so a general
  prefill TPS is not directly measurable. The §1 figure is `computed KV tokens ÷
  prefill seconds` for one specific request and is labelled as such.
* **`nvidia-smi` memory figures.** These return `N/A` on unified-memory GB10 nodes.

---

## 5. Memory floor — why the envelope is where it is

During the 500,000-token request, minimum observed `MemAvailable` per node:

| Node role | minimum `MemAvailable` |
|---|---|
| **head (binding host)** | **496,236 kB = 484.6 MiB** |
| worker A | 2,413,428 kB = 2,356.9 MiB |
| worker B | 2,852,360 kB = 2,785.5 MiB |
| worker C | 2,745,288 kB = 2,680.9 MiB |

The head node's minimum occurred **37.1 s into prefill at only 4.11 % KV occupancy**,
which shows the floor is set by the initial activation/kernel-workspace transient, not
by final KV fill. At 95.107 % KV occupancy the head node still had 523.4 MiB available;
postflight it recovered to 775.6 MiB.

`SwapTotal` is **0** on all four nodes. `pswpin`/`pswpout` were byte-identical across an
entire cutover-and-restore cycle — there is no swap, so a transient spike is an
immediate kill rather than a slowdown.

### Capacity conclusion

The 525,887-token KV allocation cleanly supports a 500,000-token cold prompt plus 128
decode tokens. It does **not** demonstrate safe room for a meaningful KV increase at
`max_num_batched_tokens = 1,024`. Projecting the observed head-node floor with the
measured ~15.98 KiB/token/rank slope:

| KV increase | cost per rank | projected head-node floor |
|---|---|---|
| +5K logical tokens | ~78.0 MiB | ~406.6 MiB |
| +10K | ~156.1 MiB | ~328.6 MiB |
| +20K | ~312.1 MiB | ~172.5 MiB |

A 512 MiB OS/runtime reserve leaves **zero** evidence-backed extra KV. Even a thin
384 MiB reserve permits only ~6.4K tokens, which is not a prudent production increase.
The 25,887 tokens of physical pool beyond the tested 500K prompt are needed for API and
output margin and should not be read as free memory.

### The OOM that set this boundary

An earlier R9 canary on the **550K** profile was OOM-killed by the Linux kernel on the
head node 3 min 20 s into serving. Root cause was measured, not inferred: R9's
per-depth FULL-graph coverage costs **+0.92 GiB per rank** over the fixed-`K` build, and
the head node — which additionally carries the API server, the EngineCore process, and
the Ray head/GCS/dashboard — had **647 MB** of host memory available before the cutover.
The three workers had 3.1–3.5 GB and would have absorbed it. The runtime fit at load and
through graph capture, then had nothing left for peak activation once concurrent
requests arrived.

The 520K profile exists precisely to give that memory back. It is what is live and what
the §1 measurement was taken on.

---

## 6. Summary of what is and is not claimed

| Claim | Status |
|---|---|
| Exact 500,000-token cold prompt + 128 output tokens succeeds at 520K API context | **measured** |
| 514.192 tok/s prefill, 33.430 tok/s decode, for that one request | **measured, single isolated test** |
| 95.107 % max KV occupancy, 0 preemptions, no restart or fatal event | **measured** |
| FULL CUDA graphs retained across all nine reachable shapes, 0 downgrades, 4/4 ranks | **measured** |
| Controller genuinely moves K2 → K4 → K5 under real traffic | **measured** |
| 14.9 % of draft positions avoided vs a verified fixed-K5 comparator | **measured** |
| Adaptive depth is *faster* than fixed K5 in tokens/second | **NOT established** — no valid matched sample exists |
| A general decode-speed figure for this stack | **NOT established** |
| Acceptance-rate targets (e.g. ≥ 92 %) | **NOT established**; the build makes them measurable, it does not measure them |
| Depth-transition cost | **NOT measured** |
| Behaviour of the 2/4/5 thresholds at `max_num_seqs=3` vs the upstream `max_num_seqs=1` tuning | **open question**, the largest untested assumption |
| Any result on hardware other than 4× DGX Spark (GB10, ARM64) | **not attempted** |
