# Build and provenance

What the published image actually is, how it was produced, and — stated plainly — what
can and cannot be rebuilt from public sources today.

---

## 1. The image

| Field | Value |
|---|---|
| Internal build tag | `glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd` |
| Local Docker image ID (config hash) | `sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39` |
| Architecture / OS | `arm64` / `linux` |
| Size | 20,342,958,503 bytes |
| RootFS layers | 70 |
| Model weights embedded | **none** |
| Source commit label | `bae57bd87b03b7c802ca391064996ec27a02d2bb` |
| Recipe file | `Dockerfile.r9-r8base`, sha256 `0758f672f797fe0c2f4812170511821e20384c0d5e0f2e18a8f68df858199a1d` |
| Build script | `scripts/r9-r8base-build-on-nord.sh`, sha256 `620cee21545dd570c8664c169214def0dadb453b76272ca5b9b72ccefa2a4633` |
| Contract | `R9_CONTRACT.json`, sha256 `1b6f00259724f86eb58c047373dabeb9b4df21a1b7bd87995c09e6cfa16bfd77` |
| Immediate build parent | private predecessor image, `sha256:7cc2e13a5f6504bdc31dd173f637239ae587e928b06410dcc8b0d29232a9cb2c` |
| Build host arch | ARM64 (native; not cross-built, not emulated) |

The image ID above is the **local Docker config hash**. It is *not* the registry
manifest digest, which is assigned at push time and recorded in
[`../release-manifest.json`](../release-manifest.json). See
[`IMAGE.md`](IMAGE.md) §4.

---

## 2. What the build actually did

The build is a **five-file source delta** applied on top of the immediate parent image:

| File | Change |
|---|---|
| `vllm/config/vllm.py` | changed |
| `vllm/v1/core/sched/scheduler.py` | changed |
| `vllm/v1/spec_decode/dynamic/acceptance_length.py` | changed |
| `vllm/v1/spec_decode/dynamic/depth_ladder.py` | **added** |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | **added** to the tracked set (pre-image was the raw upstream `e232d262` blob) |

0 files removed. 30 files carried forward from the parent are **inherited from parent
layers and never re-COPYed**, so any drift in them is drift inside the parent and fails
the guards rather than being silently re-stamped.

The build added **13 layers** on the parent's 57. That count is recorded as an
*inspected output of the build*, never asserted in advance — an earlier revision of this
work claimed "one layer" and was wrong, and the correction is kept visible rather than
quietly edited out.

### Build-time guards, all inside the image

| Guard | What it proves |
|---|---|
| Guard 1 | The parent's post-image filesystem contract: 34 files across two manifests, verified **before** any R9 `COPY`. A cryptographic statement about the parent's contents rather than its tag name. |
| Guard 2 | The 35-file post-image manifest, plus a stale-bytecode drop for the 5 moved files, `py_compile` of 24 files, string-level capability/exclusion greps, and an on-disk delta guard proving the delta against the parent is exactly 5 files (3 changed, 2 added, 0 removed) and that the 30 carried-forward files still hash to their sealed values. |
| Guard 3 | Structural AST assertions ([`../patches/r9-image-guard.py`](../patches/r9-image-guard.py)) — including the one that deliberately **contradicts** the predecessor's guard, because R9 inverts that predecessor's contract on purpose. |
| Guard 4 | Commit binding: the source commit must be full 40-hex, the three recipe hashes must be 64-hex, and the in-image contract must hash to the declared value. Writes `/opt/r9/BUILD_PROVENANCE`. |

### The commit binding is not self-certifying, and says so

`org.glm52.exp1.source_commit` and the three recipe hashes are **injected as build args**
because the build stage is a directory rather than a git checkout. The image label
states this in its own text. What makes it checkable rather than circular:

1. the build script re-hashes the **staged** files — including its own bytes — and
   refuses to build on any mismatch;
2. Guard 4 re-hashes the contract that actually landed in the image;
3. the script reads all four labels back off the finished image afterwards.

### Why `FROM` names a tag, not an ID

BuildKit cannot resolve a bare image ID: `FROM sha256:7cc2e13a…` is parsed as the Docker
Hub repository `docker.io/library/sha256:7cc2e13a…` and fails with `pull access
denied`. The build host runs Docker 29.2.1, which is BuildKit-only, so the classic
builder's local-ID resolution is unavailable. Because a tag is mutable, the parent's
identity is instead established three independent ways: preflight and post-build
verification that the tag resolves to the expected image ID, Guard 1's filesystem
contract, and a post-build proof that the parent's 57 ordered content-addressed RootFS
layer digests are an **exact prefix** of this image's layer list.

---

## 3. Recovery-build history — stated, not hidden

This image is a **replacement**. The history matters because it changes what provenance
claims are true.

1. A first R9 image (`r9-adaptive-full-9ebeb1c`,
   `sha256:7fc4fd7f676bfa7dde53016aa2a83a1c66ac62486a039512dec8468069f6828d`) was built
   from a sealed R6 base by a different recipe and passed 70 independent static checks.
2. **64 seconds after that build completed**, an external, unattributed fleet-wide
   `docker system prune`-class operation deleted 13 images and 2 containers across all
   four nodes — including that image and both R6 build-base tags. It was not issued by
   the build session; the forensic event log is preserved internally.
3. The R6 base has no registry digest (never pushed), no saved tar anywhere, and is not
   reproducible from any available Dockerfile. The original recipe therefore **cannot be
   executed truthfully any more.**
4. The published image was consequently rebuilt from a **different immediate parent**
   (the R8 predecessor image) by a **different recipe** (`Dockerfile.r9-r8base`).

Consequences, stated exactly:

* The published image is **not** a bit-identical reproduction of the destroyed one and
  does not claim to be. Different parent, different recipe, different layer structure,
  necessarily different image ID.
* The **five-file R9 runtime payload is byte-for-byte identical** — the 35-file
  post-image manifest is unchanged and still pins every file, so the reviewed
  controller, the FULL-graph coverage, the fail-closed assertions and the telemetry are
  the same bytes.
* The R8-based derivation is in one respect *more* checkable: because the 30
  carried-forward files are never `COPY`ed, any drift in them fails Guard 1 or the delta
  guard rather than being re-stamped by the build.

The image labels record all of this (`org.glm52.exp1.supersedes_build`,
`…supersedes_build_note`, `…build_kind=replacement-recovery-build`,
`…base_relationship`).

---

## 4. What can and cannot be rebuilt from public sources

### Can be verified publicly, today

| Item | How |
|---|---|
| The **five-file adaptive-MTP delta** | [`../patches/13_r9_adaptive_full_cuda.py`](../patches/13_r9_adaptive_full_cuda.py) is the anchor-based applier that produces it. |
| The **35-file sealed post-image manifest** | [`../provenance/r9-postimage.sha256`](../provenance/r9-postimage.sha256), checkable against the published image's own filesystem — see §5. |
| The **structural contract** of the shipped source | [`../patches/r9-image-guard.py`](../patches/r9-image-guard.py) runs against the published image with `python3` and `ast` only. |
| The **upstream base** of the runtime | vLLM `e232d262369b8c918cf478a7a96a0fcf8127cf65`; b12x `97b3d642…`; adaptive-MTP source `CosmicRaisins/glm-5.2-gb10` @ `600848707c…`. |
| The image's **architecture, labels, and absence of weights** | [`../scripts/verify-image.sh`](../scripts/verify-image.sh). |

### Cannot be rebuilt from public sources, today

**A full public rebuild of this image is not possible.** The reasons, in order of
severity:

1. **The immediate build parent is a private image.** `Dockerfile.r9-r8base`'s `FROM`
   resolves to `sha256:7cc2e13a…`, which has never been pushed to any registry. Without
   it, the recipe has nothing to build on.
2. **The full patch chain that produced that parent is not published here.** R9 is patch
   13 of a chain `03 → 06 → 12 → 13`, and patches 01–12 belong to the earlier revisions
   of this project. Only patch 13 and the image guard are included in this repository,
   because they are the minimum needed to describe and verify the adaptive-MTP delta.
3. **The original R6 base of that chain no longer exists anywhere** (§3), so even
   internally the chain cannot be replayed from its own beginning.
4. **The wheel-level base build is not reproducible from this repo.** The vLLM wheel
   (`0.1.dev17863+ge232d2623.exp1sm121a368r4dtypefix`, sha256 `b8ed2f89…`) and the
   FlashInfer `sm_121` wheels were produced by a separate base-image build whose recipe
   is not part of this release.

What this means for a reader: **treat the published image as an artifact you verify,
not an artifact you reproduce.** Verify the architecture, the labels, the 35-file source
manifest, the structural assertions, and the absence of weights. Those checks are real,
they run against the image you actually pulled, and they are the honest limit of what
this release can offer.

### If you want to reproduce the *idea* rather than the image

The adaptive-MTP work itself is reproducible from public sources on a vLLM fork that
already has the GB10/B12X stack working:

1. Start from [`CosmicRaisins/glm-5.2-gb10`](https://github.com/CosmicRaisins/glm-5.2-gb10)
   at `600848707ce93fe42fedbc9dd4429116696e425d` and get GLM-5.2 serving on your GB10
   nodes with `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_USE_B12X_SPARSE_INDEXER=1` and
   `--attention-backend B12X_MLA_SPARSE`.
2. Apply that project's `adaptive-mtp/overlay/.../acceptance_length.py` for the 2/4/5
   policy.
3. Apply the FULL-graph reconciliations described in
   [`ARCHITECTURE.md`](ARCHITECTURE.md) §4.3–4.5 — **do not** apply
   `patches/adaptive-mtp-vllm-hooks.patch` verbatim; its duplicate ladder parser and
   merged candidate ranges are the two defects that make adaptive depth fall back to
   eager. [`../patches/13_r9_adaptive_full_cuda.py`](../patches/13_r9_adaptive_full_cuda.py)
   shows exactly what to do instead.

---

## 5. Verifying the sealed manifest against a pulled image

The manifest paths are relative to the image's `dist-packages` root
(`/usr/local/lib/python3.12/dist-packages`). No GPU is needed.

```bash
IMAGE="ghcr.io/0xdfi/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark@sha256:<GHCR_DIGEST>"

docker run --rm --network none \
  -v "$PWD/provenance/r9-postimage.sha256:/tmp/manifest.sha256:ro" \
  --entrypoint /bin/sh "$IMAGE" -c \
  'cd /usr/local/lib/python3.12/dist-packages && sha256sum -c --strict /tmp/manifest.sha256'
```

Expect `35` `OK` lines and a zero exit status. Any `FAILED` line means the image you
pulled does not carry the sealed source this release describes.

Structural assertions on the same image:

```bash
docker run --rm --network none --entrypoint /bin/sh "$IMAGE" -c \
  'cd /usr/local/lib/python3.12/dist-packages && python3 /opt/r9/patches/r9-image-guard.py'
```

Both probes run CPU-only with `--network none` and no GPU request, import neither torch
nor vLLM, and allocate no device memory.

---

## 6. Provenance labels reference

The image carries an `org.glm52.exp1.*` label set. The ones worth knowing:

| Label | Meaning |
|---|---|
| `source_commit` | the commit the recipe was taken from (injected; see §2) |
| `dockerfile_sha256`, `build_script_sha256`, `contract_sha256` | the three recipe files, pinned by hash |
| `parent_image`, `parent_image_id`, `parent_relationship` | immediate build parent and the fact that carried-forward files are inherited |
| `base`, `base_image_id`, `base_relationship` | the R6 ancestor — **explicitly labelled ANCESTOR ONLY, NOT the immediate parent** |
| `r9_source_delta` | the five-file delta, spelled out |
| `adaptive_mtp` | ladder, window, capture shapes, and the fail-closed statement |
| `adaptive_mtp_source` | the pinned community source and the two reconciliations |
| `capabilities` | the full capability string, including `adaptive-mtp-full-cudagraphs` |
| `upstream_commits`, `upstream_commits_excluded` | what was and was not ported |
| `measured` | says **NOTHING** — the build establishes source and image structure only. All performance evidence in this repository was gathered *after* the build, on live runs, and is in [`BENCHMARKS.md`](BENCHMARKS.md). |
| `serving_status` | **stale by construction** — says "NOT SERVED, NOT A CANARY, NOT DISTRIBUTED", which was true at build time and is no longer true. See [`../IMAGE_AUDIT.md`](../IMAGE_AUDIT.md). |
