# Security

This document covers what is and is not inside the published image, the real risks of
running it, and the defaults this repository ships. It is written for an operator
deciding whether to pull a 20 GB container onto a machine they care about.

## Reporting

Open a GitHub issue for anything non-sensitive. For a suspected secret leak in the
published image or repository, or any other sensitive finding, open a **private**
GitHub security advisory on this repository rather than a public issue.

---

## 1. What is in the image

**No secrets.** The image was audited before publication — see
[`IMAGE_AUDIT.md`](IMAGE_AUDIT.md) for the exact commands and findings. The audit
covered the image config, environment, labels, build history, and filesystem, and
found:

* **No** API tokens, keys, passwords, `Authorization` headers, connection strings,
  registry credentials, or SSH material. `/root/.ssh` exists and is **empty**; there is
  no `/root/.docker`, no `.netrc`, no `.git-credentials`, no `pip.conf`.
* **No** private cluster IP addresses in the image config, labels, environment or
  build history.
* **No model weights.** `/models` does not exist in the image; there are no
  `.safetensors`, `.gguf`, `.bin` or `.pt` model artifacts, and no populated Hugging
  Face cache. (Two false positives exist and are benign: a small
  `compressed_tensors/.../hadamards.safetensors` lookup table shipped by the
  `compressed-tensors` library, and `vim`'s tutorial file.)
* The only `credentials` path found is
  `python3.12/dist-packages/anthropic/lib/credentials`, which is the Anthropic SDK's
  cloud-provider **auth-helper module directory** — Python source, not credentials.

**Residual non-secret metadata that IS in the image**, disclosed rather than hidden:
internal build hostname `<BUILD_HOST>` in labels and embedded contract JSON, three
private absolute build/evidence paths under a `/home/<user>/…` tree inside the embedded
`R9_CONTRACT.json` / `R8_CONTRACT.json`, the maintainer's first name in one
authorization sentence, and internal lineage labels (`org.hermes.*`,
`org.glm52.exp1.*`). None of these is a credential or a routable address. Full detail
and a sanitized-rebuild route: [`IMAGE_AUDIT.md`](IMAGE_AUDIT.md).

One label is **stale by construction**: `org.glm52.exp1.serving_status` says
`NOT SERVED, NOT A CANARY, NOT DISTRIBUTED`. That was true at build time. It is no
longer true — the image has since served production traffic and is being published.
Labels are baked at build time and cannot be edited without changing the image, and
this repository publishes the *exact* production image rather than a relabelled
lookalike. Treat that label as a build-time statement, not a current one.

## 2. `--trust-remote-code` — the biggest single risk

The runtime is launched with `--trust-remote-code`. **This executes arbitrary Python
from the model repository inside the container, as root.** GLM-5.2 requires it for the
custom architecture; there is no supported way to run this stack without it.

Consequences you must accept before running:

* Only point `MODEL_DIR` at weights you obtained from a source you trust and verified.
  Verify checksums against the upstream model card.
* A malicious or tampered model directory is equivalent to remote code execution on
  every node in the cluster.
* Re-verify after any re-download. `--download-dir` points at the same directory.

The container runs as `root` and is started with GPU device access. Do not run it with
`--privileged` and do not add capabilities beyond what your GPU runtime requires.

## 3. Network posture — private by default, no authentication

The vLLM OpenAI-compatible server in this configuration has **no authentication, no
authorization and no TLS**. Anyone who can reach the port can submit prompts, consume
the whole KV pool, and read whatever the model produces.

The templates in [`runtime/`](runtime/) therefore:

* require you to set `API_BIND_ADDR` explicitly and **refuse to start** if it is left
  as a placeholder,
* default the documented deployment to a **private cluster network** interface,
* never bind `0.0.0.0` for you.

If you need external access, put a reverse proxy with authentication in front of it and
keep the vLLM port off any routable interface. Ray's cluster ports
(`RAY_ADDRESS`, the GCS and object-manager ports) and NCCL/RoCE traffic must likewise
stay on a trusted, isolated network segment — Ray's cluster interface is not an
authenticated boundary.

## 4. Denial of service is trivially easy at this configuration

`max_num_seqs=3` with a 520,000-token context means **three concurrent requests can
occupy the entire KV pool**. A single 500K-token prompt takes ~16 minutes of prefill.
There is no per-user quota in this configuration. Rate limiting and request-size limits
are your responsibility, upstream of the server.

## 5. Memory posture

These are unified-memory GB10 nodes with `SwapTotal: 0`. The measured minimum available
host memory on the head node during a 500K-token request was **484.6 MiB**. There is no
elastic buffer: an unexpected transient allocation is an immediate kernel OOM kill of a
worker, which takes the engine down. Do not co-locate other memory-hungry workloads on
these nodes, and do not raise `kv_cache_memory_bytes`, `max_num_batched_tokens` or
`max_num_seqs` without re-measuring. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) §5.

## 6. Supply-chain verification

Verify what you pulled before you run it:

```bash
./scripts/verify-image.sh ghcr.io/0xdfi/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark@sha256:<GHCR_DIGEST>
```

The published digest is recorded in [`release-manifest.json`](release-manifest.json)
and [`docs/IMAGE.md`](docs/IMAGE.md).

**A caution about digests.** The local Docker *image ID*
(`sha256:50261a39…`) is the hash of the image **config**, computed locally. The
registry **manifest digest** (`ghcr.io/...@sha256:…`) is a different hash over a
different object, assigned when the image is pushed. They are not equal and neither can
be derived from the other. `verify-image.sh` prints both and never claims otherwise.
Pin to the **registry digest** for reproducible pulls.

The image is **not** currently signed with cosign/sigstore and carries no SBOM
attestation. If either is added later it will be recorded in
[`release-manifest.json`](release-manifest.json).

## 7. Reproducibility limits

This image cannot presently be rebuilt bit-for-bit from public sources. Its immediate
build parent is a private predecessor image, and the full patch chain that produced
that parent is not published here. What *is* publicly verifiable is the sealed
35-file source manifest ([`provenance/r9-postimage.sha256`](provenance/r9-postimage.sha256)),
which you can check against the running image's filesystem yourself. See
[`docs/BUILD.md`](docs/BUILD.md), which states the limits plainly rather than implying a
reproducible build that does not exist.

## 8. What this repository will never contain

No credentials, tokens, keys, cookies, connection strings or SSH material; no private
cluster IP addresses, internal host aliases, local usernames or absolute host paths; no
container IDs or internal transaction directories; no model weights. Every network
address, hostname, interface name and filesystem path in [`runtime/`](runtime/) is a
placeholder that fails closed if left unset.
