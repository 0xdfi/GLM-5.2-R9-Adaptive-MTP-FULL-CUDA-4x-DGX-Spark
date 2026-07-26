# Image: tags, digests, verification

## 1. Public location

```
ghcr.io/0xdfi/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark
```

| Tag | Meaning | Stability |
|---|---|---|
| `r9-adaptive-full-bae57bd` | **the exact production image**, named for its source commit | immutable — this tag will never be re-pointed |
| `r9` | convenience alias for the above | may move if a later R9 build is published |

Recommended: **pin the digest**, not the tag.

```bash
docker pull ghcr.io/0xdfi/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark@sha256:<GHCR_DIGEST>
```

`<GHCR_DIGEST>` is `null` in [`../release-manifest.json`](../release-manifest.json)
until the image is actually pushed; the push step in
[`../PUBLISHING.md`](../PUBLISHING.md) fills it in from the registry rather than from a
local computation.

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

## 3. Verify what you pulled

```bash
./scripts/verify-image.sh ghcr.io/0xdfi/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark@sha256:<GHCR_DIGEST>
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
7. the registry digest is reported **separately** from the local config ID, with an
   explicit note that they are different hashes over different objects.

Deeper source verification (35-file manifest, AST assertions) is in
[`BUILD.md`](BUILD.md) §5.

## 4. Image ID vs registry digest — do not confuse these

| Name | What it hashes | Where it comes from |
|---|---|---|
| **Image ID** `sha256:50261a39…` | the image **config JSON** | computed locally by the Docker daemon |
| **Registry digest** `…@sha256:…` | the **manifest** (or manifest list) as stored in the registry | assigned by the registry at push time; returned by `docker buildx imagetools inspect` or `docker inspect --format '{{index .RepoDigests 0}}'` after a push/pull |

They are **not equal**, and neither can be derived from the other. Anything that tells
you `docker inspect`'s `Id` is your pull digest is wrong.
[`../scripts/verify-image.sh`](../scripts/verify-image.sh) prints both and labels them
distinctly.

Also note: before this image is pushed anywhere, `RepoDigests` is `[]` — the image has
no digest at all. That is expected for a locally built image and is not a defect.

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
* **No multi-arch manifest list.** A single-platform `linux/arm64` manifest is
  published.
* **No signature or SBOM attestation at this time.** If either is added later it will be
  recorded in [`../release-manifest.json`](../release-manifest.json) and here.
* **No bundled weights.** Deliberate — see [`../SECURITY.md`](../SECURITY.md) and
  [`../NOTICE.md`](../NOTICE.md).
