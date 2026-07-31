# Benchmarks

Only measured, evidenced results appear here. Where something was not measured, this
document says so instead of estimating. Every number carries its denominator.

**All measurements are from the four-node DGX Spark deployment described in
[`ARCHITECTURE.md`](ARCHITECTURE.md): 4 nodes × 1 GB10 each, ARM64, TP4/DCP2.** Nothing
here generalizes to other hardware, other topologies, other concurrency levels or other
workloads, and none of it is offered as such.

---

## 0. R13 Fast + Balanced (C4, twelve-shape) — 2026-07-31

R13 adds two named launch profiles at C4 concurrency (`max_num_seqs=4`, twelve-shape
FULL-graph coverage `6,12,18,24`) and a launcher guard that fixes the DCP1 boot crash
present in R9. The numbers below are from the R13 image
`sha256:6d7b06b13f3839da8d9a447e560ba51a894385d52ca052a6f9af24d072d94e82`
(release `r13-balanced-fast-c4`; the image's internal build tag is
`r9.1-scheduler-liveness-4lane`), measured 2026-07-31 with 4/4 nodes
healthy and **0 preemptions** in every leg.

| Leg | Profile | DCP | `max_model_len` | prefill @200K (tok/s) | C1 prose decode (tok/s) | C4 aggregate (tok/s) |
|---|---|---|---|---|---|---|
| Fast | `fast` | 1 | 319,000 | 695.1 | 23.0 | 83.4 |
| Balanced | `balanced` | 2 | 520,000 | 602.0 | 31.14 | 71.83 |

**Full-coverage check:** query lengths `[3,5,6]` across request counts `1..4`; the
complete twelve-shape set is `3/6/9/12`, `5/10/15/20`, `6/12/18/24`. FULL CUDA-graph
coverage verified on 4/4 ranks; **0** PIECEWISE downgrades, **0** uncaptured-shape
warnings, **0** eager/`NONE`/`FULL_DECODE_ONLY`/fixed-K fallback markers, 1 engine
start per leg.

### How to read this table

* The **prefill** number is computed-KV-tokens ÷ prefill-seconds for a 200K-token
  cold prompt (prefix-cache hit delta 0), completion capped at 128 tokens with
  `ignore_eos=true`. It is a single-request figure, not a sustained rate.
* **C1 prose decode** is concurrency-1 decode throughput on a representative prose
  payload; **C4 aggregate** is aggregate decode throughput at concurrency 4 across
  the same payload shape.
* The `fast` leg wins on prefill and C4 aggregate because it carries no DCP comm
  layer; `balanced` wins on C1 prose decode (≈31 vs ≈23 tok/s) because DCP2 splits
  the context across ranks. The `balanced` leg is the only one that can serve the
  500K-class cold prompt (`max_model_len=520,000` in the table above), at the cost
  of lower aggregate C4 throughput.

### Denominators and caveats — read before quoting

* These are **single-leg, single-run** measurements on this exact four-node cluster.
  They are not general throughput figures and should not be quoted as such.
* The C4 aggregate is the **sum** of per-request decode rates at concurrency 4 on a
  fixed payload; it is not a benchmark-suite score and is not normalized across payload
  mixes.
* The 200K prefill-leg prompt length is shorter than the `balanced` row's
  `max_model_len=520,000` capacity ceiling; prefill throughput and cold-prompt
  capacity are different measurements and are not the same workload.
* `fast` does **not** prove a 319K cold-prompt capacity claim — only that prefill at
  200K measured 695.1 tok/s on that profile. The largest proven cold-prompt envelope
  is the `balanced` row (`max_model_len=520,000`); the `fast` row's 319K ceiling is
  the largest proven cold prompt on that profile.

**Receipts:** `FINAL_ACCEPTANCE.json` (fast leg), `RELAUNCH_RECEIPT.json`
(balanced leg).

> The legacy R9 (C3, nine-shape) speed measurements previously in §1–§6 have been
> removed. The R9 image release (`r9-adaptive-full-bae57bd`) remains available for
> download; its measured numbers live in its original release artifacts.

---
