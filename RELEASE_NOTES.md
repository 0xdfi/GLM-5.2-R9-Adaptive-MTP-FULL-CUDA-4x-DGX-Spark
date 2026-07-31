# R13 Fast + Balanced runtime image (C4, twelve-shape)

This release contains the **Linux/ARM64** container image for the GLM-5.2 R13
runtime: two named launch profiles (`fast`, `balanced`), both at C4 concurrency,
plus a launcher guard that fixes a DCP1 boot crash present in R9.

> **R13 supersedes R9.** The legacy R9 notes are retained below. The R9 release
> assets remain available for reproducibility.

## What changed in R13

1. **Two named profiles, both C4.** R9 ran a single `520k`/`550k` envelope at C3
   concurrency (`max_num_seqs=3`, nine-shape coverage). R13 ships two profiles
   selectable by name, both at C4 concurrency (`max_num_seqs=4`, twelve-shape
   coverage `6,12,18,24`):
   - **`fast`** — DCP1, `max_model_len=319000`, `kv_cache_memory_bytes=10233000000`.
     Highest prefill and C4-aggregate throughput. No DCP comm layer.
   - **`balanced`** — DCP2, `max_model_len=520000`, `kv_cache_memory_bytes=8410000000`,
     `--dcp-comm-backend a2a`. Faster prose decode at C1; lower C4 aggregate than fast.
2. **DCP guard fix.** At DCP1 the vLLM engine rejects an explicit
   `--dcp-comm-backend` flag and crashes on boot; R9 always emitted the flag.
   `start-node.sh` now emits the DCP comm flags **only** when the profile selects
   DCP>1 (`${DCP_ARGS[@]+"${DCP_ARGS[@]}"}`, compatible with `set -u` and bash 3.2).
3. **Twelve-shape FULL-graph coverage.** The CUDA-graph capture set widens from
   `6,12,18` (C3, nine shapes) to `6,12,18,24` (C4, twelve shapes) so the largest
   reachable per-step batch (`max_num_seqs=4` × deepest draft) is captured exactly.

## Measured performance (four-node DGX Spark cluster, 2026-07-31)

Receipts: `FINAL_ACCEPTANCE.json` (fast) and `RELAUNCH_RECEIPT.json` (balanced).
All numbers are from the R13 image `sha256:6d7b06b1…`, 4/4 nodes healthy,
**0 preemptions** in every leg.

| Leg | Profile | prefill (tok/s, 200K) | C1 prose decode (tok/s) | C4 aggregate (tok/s) |
|---|---|---|---|---|
| Fast | `fast` (DCP1, 319K) | 695.1 | 23.0 | 83.4 |
| Balanced | `balanced` (DCP2, 520K) | 602.0 | 31.14 | 71.83 |

FULL CUDA-graph coverage verified for query lengths `[3,5,6]` across request counts
`1..4`; **zero** PIECEWISE downgrades, **zero** uncaptured-shape warnings. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for full denominators and per-shape detail.

## Image contents

- Runtime image only; **model weights are not included**.
- Image platform: `linux/arm64`.
- Docker image config ID: `sha256:6d7b06b13f3839da8d9a447e560ba51a894385d52ca052a6f9af24d072d94e82`.
- Uncompressed Docker-reported size: `20,327,717,376` bytes (~20.3 GB).
- Compressed archive: `glm52-r13-balanced-fast-c4.oci.tar.zst`, ~7.4 GB,
  split into four sub-2 GiB parts.
- Source tag: `glm52-exp1-sm121a-368-canary:r9.1-scheduler-liveness-4lane` (internal build tag; the public release is named R13).
- Built from the **same source commit** as R9 (`bae57bd87b03b7c802ca391064996ec27a02d2bb`);
  the C4/twelve-shape behavior is a runtime configuration, not an image rebuild.

`SHA256SUMS` covers every part and the reassembled archive. Do not trust an image
whose parts fail the checksum.

## Download and load

From a clone of the repository:

```bash
./scripts/download-image.sh
zstd -d -c image-download/glm52-r13-balanced-fast-c4.oci.tar.zst | docker load
./scripts/verify-image.sh glm52-exp1-sm121a-368-canary:r9.1-scheduler-liveness-4lane
```

The downloader retrieves every numbered part from this release, verifies each part,
reconstructs the archive, and verifies the full archive. `verify-image.sh` checks
architecture, OS, provenance labels, and the adaptive-MTP capability strings.

## Model weights

Download separately from [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix), subject to its terms. See the repository README for the four-node launch procedure, security cautions, evidence, and full attribution.

---

# R9 adaptive-MTP FULL-CUDA runtime image (legacy)

This release contains the exact **Linux/ARM64** container image used for the verified GLM-5.2 R9 adaptive-MTP runtime on four DGX Spark nodes.

## What changed

R9 makes speculative depth adaptive across **K=2, K=4, and K=5**. The runtime spends less draft work on difficult text, climbs when deeper drafts are paying off, and preserves **FULL CUDA-graph coverage for every reachable adaptive depth**. Load-time assertions fail closed if the scheduler can select a shape that was not captured.

## Image contents

- Runtime image only; **model weights are not included**.
- Image platform: `linux/arm64`.
- Docker image config ID: `sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39`.
- Uncompressed Docker-reported size: `20,342,958,503` bytes.
- Source tag: `glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd`.

The compressed Docker archive is split into sub-2 GiB release assets because GitHub limits individual release-asset size. `SHA256SUMS` covers the complete archive and every part.

## Download and load

From a clone of the repository:

```bash
./scripts/download-image.sh
zstd -d -c image-download/glm52-r9-adaptive-full-bae57bd.oci.tar.zst | docker load
./scripts/verify-image.sh glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd
```

The downloader retrieves every numbered part from this release, verifies each part, reconstructs the archive, and verifies the full archive.

## Model weights

Download separately from [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix), subject to its terms. See the repository README for the four-node launch procedure, security cautions, evidence, and full attribution.
