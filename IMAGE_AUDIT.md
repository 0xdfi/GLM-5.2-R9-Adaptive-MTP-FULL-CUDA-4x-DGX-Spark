# Pre-publication image audit

**Subject:** the exact production runtime image proposed for public release.

| | |
|---|---|
| Internal tag | `glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd` |
| Image ID | `sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39` |
| Architecture / OS | `arm64` / `linux` |
| Size | 20,342,958,503 bytes |
| RootFS layers | 70 |
| RepoDigests at audit time | `[]` — never pushed to any registry |
| Audit date | 2026-07-26 |
| Audit method | read-only `docker image inspect`, `docker history --no-trunc`, and short-lived `docker run --rm --network none --entrypoint /bin/sh` probes |

---

## Verdict

> ## **GO — with four disclosed residuals**
>
> The image contains **no credentials, no private network addresses, and no model
> weights**. It is safe to publish on those axes.
>
> It does contain **four items of non-secret internal metadata** (§3). None is a
> credential and none is routable. Publishing the *exact* production image means
> publishing them, because labels and embedded files are baked into the layers and
> cannot be edited without producing a different image.
>
> **The owner must explicitly accept those four residuals before the push**, or choose
> the sanitized-rebuild route in §6 — which produces a *different* image that is no
> longer the exact artifact that served the measurements in `docs/BENCHMARKS.md`.
>
> The recommendation is to **publish the exact image and disclose the residuals** (as
> `SECURITY.md` and `docs/BUILD.md` already do). The alternative trades away the
> single most valuable property of this release — that the published bytes are the
> bytes that produced the published numbers — to hide a LAN hostname and a home
> directory name that matches the owner's already-public handle.

---

## 1. Scope and safety of the audit itself

Every command was read-only or ran in a throwaway container. Specifically:

* `docker image inspect` and `docker history --no-trunc` — read-only.
* Filesystem probes: `docker run --rm --network none --entrypoint /bin/sh <id> -c '<probe>'`
  — no network, no GPU request, auto-removed, never importing torch or vLLM, never
  allocating device memory.
* **The live R9 service was not stopped, restarted, reconfigured, retagged or
  otherwise touched.** No image was pushed, tagged, exported, saved or deleted. No
  registry login occurred.

Live-state reads used to corroborate the current runtime (`/health`, `/v1/models`, the
container's `/proc/<pid>/cmdline` and `/proc/<pid>/environ`, and the server log) were
also read-only.

## 2. Clean findings

### 2.1 No secrets — PASS

Searched: image `Config.Env` (all 47 variables), all labels, the full 157-line build
history, and the filesystem.

| Probe | Result |
|---|---|
| Regex sweep for `hf_*`, `ghp_*`, `sk-*`, `AKIA*`, `BEGIN … PRIVATE KEY`, `Authorization: Bearer` across `/opt`, `/workspace`, `/etc` | **0 hits** |
| `/root/.ssh` | exists, **empty** |
| `/root/.docker` | **does not exist** |
| `/etc/pip.conf`, `/root/.config/pip` | **do not exist** |
| `.netrc`, `.git-credentials`, `.npmrc`, `credentials` files | **none** |
| `*.pem` / `*.key` hits | CA-bundle and trust-store files only (`certifi`, `/etc/ssl/certs`, `grpc` roots) |
| `python3.12/dist-packages/anthropic/lib/credentials` | **false positive** — the Anthropic SDK's cloud-provider auth-helper *module directory* (`_auth.py`, `_providers.py`, …). Python source, no credential values. |
| Build-history env/args | only `R9_SOURCE_COMMIT` and three sha256 recipe hashes |

### 2.2 No private network addresses — PASS

| Probe | Result |
|---|---|
| IPv4 regex over the full build history | 3 hits, all **package version strings** (`12.6.3.3`, `13.0.1.2`, `13.1.0.3`) |
| RFC1918 / CGNAT regex over `/opt` (all embedded contracts, patches and manifests) | **0 hits** |
| Image `Config.Env` | no address of any kind |

The cluster addresses that appear in the *running container's* environment
(`VLLM_HOST_IP`, `RAY_ADDRESS`, `--host`) are **runtime** values supplied at
`docker run` time. They are not in the image.

### 2.3 No model weights — PASS

| Probe | Result |
|---|---|
| `/models`, `/opt/models` | **do not exist** |
| `find / -xdev` for `*.safetensors`, `*.gguf`, `*.pt`, model `*.bin` | 2 hits, both benign: `compressed_tensors/transform/utils/hadamards.safetensors` (a small library lookup table) and `/usr/share/vim/vim91/tutor/tutor.pt` (vim's Portuguese tutorial) |
| `/root/.cache` | contains only `pip` |
| Hugging Face cache | absent |

The ~20 GB is CUDA, PyTorch, vLLM, b12x, FlashInfer, Nsight Compute and the wider
Python dependency tree.

### 2.4 No private operational receipts — PASS

`/opt/r9` and `/opt/r8` contain only: the two contract JSONs, the sealed sha256
manifests, the patch appliers, the image guards, and `BUILD_PROVENANCE` (four
`key=value` lines: source commit and three recipe hashes). `/opt/exp1-evidence`
contains one file, `mtp_mapping_smoke.py` — a self-contained unit smoke test using
`unittest.mock` against `deepseek_mtp`, with no data, no paths and no addresses.

No cutover logs, benchmark JSON, `/metrics` snapshots, kernel logs or transaction
directories are inside the image.

### 2.5 The shipped R9 source is clean — PASS

The five R9 files under `dist-packages` were grepped for home paths, `/Users/`,
internal hostnames, addresses and usernames: **0 hits**. They carry correct SPDX
headers (`Apache-2.0`, `Copyright contributors to the vLLM project`) and provenance
comments naming the upstream source and commit.

---

## 3. Residual findings — non-secret, disclosed, and NOT automatically fixable

These are the four items the owner must accept. None is a credential. None is
routable. All are baked into image layers.

### R-1. Internal build hostname `<BUILD_HOST>`

Appears in:
* label `org.glm52.exp1.serving_status` ("Built on `<BUILD_HOST>` only.")
* `/opt/r9/R9_CONTRACT.json` — 5 occurrences (activation, preflight, authorization
  scope, build host, persistence form)

**Assessment:** a LAN alias for a machine on a private network. It is not resolvable
from the internet and reveals nothing beyond "the build machine is called this". Low
sensitivity. It is also, arguably, useful provenance — it is the only place the
build host is named at all.

### R-2. Private absolute paths containing the build user's directory name

Three distinct paths, all inside embedded contract JSON:

* `/home/<user>/dgx-spark/autostart/active-runtime` — in `R9_CONTRACT.json`
* `/home/<user>/exp1-build/r9-adaptive-full-r8base` — in `R9_CONTRACT.json`
* `/home/<user>/exp1-evidence/r8-container-transaction.json` — in `R8_CONTRACT.json`

**Assessment:** the directory name is a short Linux username that matches the owner's
already-public GitHub/Hugging Face handle. It discloses a filesystem layout on a
private machine. Low sensitivity, but it *is* a username and a private path, which is
exactly the class this project's own rules say not to publish — hence flagging it for
an explicit decision rather than waving it through.

### R-3. The owner's first name, once

`/opt/r9/R9_CONTRACT.json` line 265: `"authorization_scope": "<OWNER_FIRST_NAME>
authorized source/contract/test changes …"` — the name appears verbatim in the image;
it is redacted here.

**Assessment:** a first name in an authorization sentence. Already associated with the
public account. Low sensitivity.

### R-4. Internal lineage labels, including one that is now false

The image carries ~60 `org.glm52.exp1.*` labels and 2 `org.hermes.*` labels describing
the internal experiment lineage (R4 → R6 → R7 → R8 → R9, destroyed builds, rejected
builds, excluded upstream commits, and so on).

Most of this is **good** provenance and `docs/BUILD.md` publishes it deliberately. One
label, however, is **stale by construction and now false**:

```
org.glm52.exp1.serving_status =
  "NOT SERVED, NOT A CANARY, NOT DISTRIBUTED. Built on <BUILD_HOST> only.
   Serving requires a SEPARATE explicit authorization; this image has never
   been launched."
```

That was true when the label was written. The image has since served production
traffic on four nodes and is being published. The label cannot be edited without
rebuilding, so it must be **contradicted in documentation**, which
[`SECURITY.md`](SECURITY.md), [`docs/BUILD.md`](docs/BUILD.md) and
[`scripts/verify-image.sh`](scripts/verify-image.sh) all now do explicitly.

The label `org.glm52.exp1.measured = "NOTHING…"` is similarly build-time-scoped and is
still *literally* correct — the build measured nothing; the measurements came later and
live in `docs/BENCHMARKS.md`.

---

## 4. Licensing findings

**This is the one finding with real legal weight, and it is not a defect — it is an
obligation.**

The image's base is an NVIDIA CUDA container. `/NGC-DL-CONTAINER-LICENSE` is present in
the image root. Relevant terms, read in full during this audit:

* **§1c permits** developing and extending the container into a *Compatible derived
  container that includes the entire container plus other software with primary
  functionality*, and **distributing that derived container**. This image qualifies:
  it adds a full vLLM/b12x serving stack.
* **§2 imposes distribution requirements**: material additional functionality beyond
  the included portions (satisfied); the notice **"This software contains source code
  provided by NVIDIA Corporation."** must be included in distributed modifications and
  derivative works of source code (now in [`NOTICE.md`](NOTICE.md)); onward
  distribution must be under terms at least as protective (stated in `NOTICE.md`);
  and NVIDIA must be notified of known non-compliant distribution.
* **§2's definition of "distribution" includes deploying the container in a service
  for third parties over the internet.** Anyone who serves this image publicly inherits
  the obligations.
* **§4b/§4c prohibit** removing proprietary notices and distributing the container as a
  stand-alone product.

**Action taken:** [`NOTICE.md`](NOTICE.md) carries the required NVIDIA notice verbatim,
summarizes the distribution requirements, and states that they pass to downstream
redistributors. [`LICENSE`](LICENSE) explicitly scopes the Apache-2.0 grant to
repository material only and disclaims any grant over the image.

**Not legal advice.** The owner should satisfy themselves that publishing a derived
NVIDIA container on a public registry is acceptable in their jurisdiction and for their
purposes before pushing. This is the one item in this audit that is a judgement call
rather than a fact.

Other bundled licences (vLLM Apache-2.0, b12x Apache-2.0, FlashInfer Apache-2.0,
DeepGEMM MIT, NCCL BSD-3-Clause) are present in-image under each package's
`*.dist-info/licenses` and are catalogued in `NOTICE.md`.

---

## 5. Why publishing a private-parent image still works

The image's immediate build parent (`sha256:7cc2e13a…`) has never been pushed anywhere.
That does **not** block publication: a registry push uploads the **reachable layer
closure** of the image being pushed, which includes every inherited parent layer. The
consumer pulls a complete, self-contained image.

Two honest consequences:

1. **The parent's layers become public too.** They were audited here as part of the
   filesystem probes (the probes see the merged filesystem, not just R9's 13 layers),
   and the findings above cover them. The `/opt/r8` contract and patch set, and finding
   R-2's third path, are parent-layer content.
2. **Publishing the child does not make the parent independently pullable** — no tag
   or digest is published for it — but its content is present in the child. Anyone who
   pulls this image has the parent's filesystem.

## 6. Sanitized-rebuild route, if the residuals are rejected

If the owner declines R-1/R-2/R-3, the minimal change that removes them is a
one-layer derived build:

```dockerfile
# Dockerfile.sanitized  — NOT executed by this audit; provided for the owner's decision.
FROM glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd

# Replace the two embedded contracts with redacted copies. Redact ONLY the
# hostname, the /home/<user>/... paths and the first name; keep every hash,
# commit, manifest and structural statement byte-identical so the contract
# remains verifiable.
COPY sanitized/R9_CONTRACT.json /opt/r9/R9_CONTRACT.json
COPY sanitized/R8_CONTRACT.json /opt/r8/R8_CONTRACT.json

# Re-state serving_status truthfully and overwrite the hostname-bearing label.
LABEL org.glm52.exp1.serving_status="Served in production on a four-node DGX Spark cluster; published for public use."
```

**Costs of doing this, stated plainly:**

* The result is a **different image** with a different ID. It is no longer the exact
  artifact that produced the measurements in `docs/BENCHMARKS.md`, and every
  `verify-image.sh` expectation, the `contract_sha256` label, and the in-image
  `BUILD_PROVENANCE`/`R9_CONTRACT.json` hash binding would all have to be recomputed
  and re-documented.
* The old contract hash (`1b6f0025…`) would no longer match the file at
  `/opt/r9/R9_CONTRACT.json`, so **Guard 4's binding would be broken** unless the whole
  chain is re-derived. That is a meaningful loss of provenance integrity in exchange
  for hiding a LAN hostname.
* Layers from the parent still carry the original bytes in the *history*, though the
  merged filesystem the consumer sees would show the redacted files.

This is why the recommendation is disclosure, not sanitization.

---

## 7. Checklist summary

| Check | Result |
|---|---|
| Tokens / keys / passwords / auth headers | **none** |
| SSH material / registry credentials | **none** (`/root/.ssh` empty, no `.docker`) |
| Private / RFC1918 / CGNAT addresses | **none** |
| Model weights or HF cache | **none** |
| Private operational receipts, logs, benchmarks | **none** |
| Private home paths | **3**, in embedded contract JSON (R-2) |
| Internal hostname | **yes**, in labels + contracts (R-1) |
| Owner first name | **1 occurrence** (R-3) |
| Internal lineage labels | **yes**, deliberately (R-4) |
| Stale/false label | **1** — `serving_status` (R-4), contradicted in documentation |
| Upstream licences present in-image | **yes**, incl. `/NGC-DL-CONTAINER-LICENSE` |
| NVIDIA notice obligation satisfied in this package | **yes**, `NOTICE.md` |
| Architecture | `arm64` / `linux`, single-platform |
| **Overall** | **GO, conditional on owner acceptance of R-1 … R-4** |

## 8. What must happen before the push

1. Owner reads §3 and accepts R-1 through R-4, **or** chooses §6.
2. Owner accepts the §4 licensing position on redistributing a derived NVIDIA
   container.
3. Push per [`PUBLISHING.md`](PUBLISHING.md), then record the registry digest in
   `release-manifest.json` and `docs/IMAGE.md`.
4. Re-run `scripts/verify-image.sh` against the **pulled** digest-pinned reference from
   a machine that has never held the local image, to confirm anonymous pull works and
   the labels survived the round trip.
