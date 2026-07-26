# R9 adaptive-MTP FULL-CUDA runtime image

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
