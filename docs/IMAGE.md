# Image: tags, digests, verification

## 1. Public location

The canonical image artifact is the
[`r9-adaptive-full-bae57bd` GitHub release](https://github.com/0xdfi/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark/releases/tag/r9-adaptive-full-bae57bd).
It contains one zstd-compressed Docker archive split into five sub-2 GiB assets plus
`SHA256SUMS`. The release tag is immutable and will not be re-pointed.

```bash
./scripts/download-image.sh
zstd -d -c image-download/glm52-r9-adaptive-full-bae57bd.oci.tar.zst | docker load
```

The complete archive SHA-256 is
`f1c8a209f503da0a76b655eeb38e4fa573f1408e25b3547b605fbec5e7a67dc4`.
A future GHCR mirror may be added, but the downloadable release archive is the verified
public distribution path for this release.

## 2. Facts that are already known and immutable

| Field | Value |
|---|---|
| Architecture | `arm64` |
| OS | `linux` |
| Size | 20,342,958,503 bytes |
| RootFS layers | 70 |
| Local Docker image ID (config hash) | `sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39` |
| Internal build tag | `glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd` |
| Source commit label | `bae57bd87b03b7c802ca391064996ec27a02d2bb` |
| Model weights included | **no** |
| Entrypoint | `/opt/nvidia/nvidia_entrypoint.sh` |
| Working directory | `/workspace/vllm` |
| Python | 3.12, packages at `/usr/local/lib/python3.12/dist-packages` |
| CUDA | 13.0.2 |
| Base | NVIDIA CUDA container, `ubuntu` 24.04, `sbsa` |

## 3. Verify what you loaded

```bash
./scripts/verify-image.sh glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd
```

Checks performed:

1. the ref resolves and can be inspected;
2. `Architecture == arm64` and `Os == linux`;
3. the expected `org.glm52.exp1.*` provenance labels are present with the expected
   values (`source_commit`, `dockerfile_sha256`, `build_script_sha256`,
   `contract_sha256`, `parent_image_id`, `revision`);
4. the `capabilities` label contains `adaptive-mtp-depth-245-default-on`,
   `adaptive-mtp-full-cudagraphs` and `mtp-window-telemetry`;
5. the `adaptive_mtp` label states the ladder `2,4,5` and the nine capture shapes;
6. no model-weight artifacts are present (optional deep probe, `--deep`);
7. the local Docker config ID is reported separately from the release archive SHA-256.

Deeper source verification (35-file manifest, AST assertions) is in
[`BUILD.md`](BUILD.md) §5.

## 4. Archive SHA-256 vs Docker image ID

| Name | What it hashes | Value |
|---|---|---|
| **Archive SHA-256** | the complete zstd-compressed `docker save` archive published in five parts | `f1c8a209f503da0a76b655eeb38e4fa573f1408e25b3547b605fbec5e7a67dc4` |
| **Docker image ID** | the image config JSON loaded into Docker | `sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39` |

They are different hashes over different objects. `scripts/download-image.sh` checks
the first; `scripts/verify-image.sh` checks and reports the second. If a GHCR mirror is
published later, its registry manifest digest will be a third distinct hash.

## 5. Running it

The image's entrypoint is NVIDIA's container entrypoint; the vLLM server is launched as
a command inside a long-lived container. See [`../runtime/`](../runtime/) for the public
four-node templates, and [`../README.md`](../README.md) §3 for the short path.

Minimum host requirements:

* `linux/arm64` with NVIDIA GB10 (`sm_121a`) and a driver satisfying the image's
  `NVIDIA_REQUIRE_CUDA` constraint (CUDA ≥ 13.0);
* the NVIDIA container runtime;
* a locally downloaded GLM-5.2 weights directory to bind-mount read-only at `/models`;
* a writable directory for JIT caches, bind-mounted read-write;
* four such nodes on a private RDMA/RoCE network for the documented topology.

## 6. Non-goals

* **No x86_64 image.** The stack is GB10/ARM64-specific.
* **No registry mirror yet.** The canonical release artifact is the verified split Docker
  archive; a GHCR mirror can be added later without changing these bytes.
* **No signature or SBOM attestation at this time.** If either is added later it will be
  recorded in [`../release-manifest.json`](../release-manifest.json) and here.
* **No bundled weights.** Deliberate — see [`../SECURITY.md`](../SECURITY.md) and
  [`../NOTICE.md`](../NOTICE.md).
