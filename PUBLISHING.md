# Publishing runbook

For the operator performing the reviewed publication. Nothing in this document has been
executed: no repository was created, no registry was logged into, no image was tagged or
pushed, and no production state was changed.

**Read [`IMAGE_AUDIT.md`](IMAGE_AUDIT.md) §3 and §4 first and make the two decisions it
asks for.** Steps 3 onward are irreversible in practice — a 20 GB layer set pushed to a
public registry may be mirrored, cached or indexed within minutes, and deleting the
package later does not un-publish it.

Placeholders used below:

| Placeholder | Meaning |
|---|---|
| `<BUILD_HOST>` | the SSH alias of the node holding the local image |
| `<GH_USER>` | the GitHub account/org that will own the repo and package (`0xdfi`) |
| `<GH_TOKEN_ENV>` | name of an environment variable holding a GHCR PAT with `write:packages` — **never** paste the token into a shell command |
| `<GHCR_DIGEST>` | the registry manifest digest, known only after step 4 |

---

## 0. Pre-flight decisions

- [ ] Residuals R-1 … R-4 in [`IMAGE_AUDIT.md`](IMAGE_AUDIT.md) §3 accepted, **or** the
      §6 sanitized route chosen instead (in which case stop and re-derive; this runbook
      publishes the exact image).
- [ ] The NVIDIA Deep Learning Container License position in
      [`IMAGE_AUDIT.md`](IMAGE_AUDIT.md) §4 accepted.
- [ ] Understood that pushing publishes the private parent image's **layer content**
      (not a pullable tag for it) — [`IMAGE_AUDIT.md`](IMAGE_AUDIT.md) §5.
- [ ] Sufficient upload bandwidth and time budgeted: ~20 GB from `<BUILD_HOST>`.

---

## 1. Create the GitHub repository and push this package

```bash
cd /path/to/r9-publication-opus5

# Confirm the package is clean and committed before anything leaves the machine.
git status --short
git diff --check
python3 -m unittest discover -s tests -v

gh repo create <GH_USER>/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark \
  --public \
  --description "GLM-5.2 on 4x DGX Spark with adaptive MTP K2/K4/K5, FULL CUDA graphs, DCP2, 520K context, and a downloadable ARM64 runtime image." \
  --source . --remote origin --push
```

Add topics so it is discoverable alongside the prior releases:

```bash
gh repo edit <GH_USER>/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark \
  --add-topic glm-5-2 --add-topic dgx-spark --add-topic gb10 \
  --add-topic vllm --add-topic speculative-decoding --add-topic mtp \
  --add-topic cuda-graphs --add-topic nvfp4 --add-topic arm64 \
  --add-topic long-context
```

Confirm CI went green before continuing:

```bash
gh run list --repo <GH_USER>/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark --limit 3
gh run watch --repo <GH_USER>/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark
```

---

## 2. Log in to GHCR from the build host — without exposing the token

Run **on `<BUILD_HOST>`**. Do not echo, log or paste the token.

```bash
ssh <BUILD_HOST>

# The token must already be in the environment; this command never prints it.
printf '%s' "${<GH_TOKEN_ENV>:?token not set}" \
  | docker login ghcr.io -u <GH_USER> --password-stdin
```

If the login output contains anything token-shaped, treat it as `[REDACTED]` and do not
copy it into any receipt.

---

## 3. Tag and push the exact image

**Verify you are tagging the right bytes before you tag.** The image ID is the check.

```bash
# On <BUILD_HOST>:
EXPECT_ID="sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39"
SRC="glm52-exp1-sm121a-368-canary:r9-adaptive-full-bae57bd"
ACTUAL_ID="$(docker image inspect --format '{{.Id}}' "$SRC")"
[ "$ACTUAL_ID" = "$EXPECT_ID" ] || { echo "REFUSING: image ID mismatch: $ACTUAL_ID"; exit 1; }

docker image inspect --format 'arch={{.Architecture}} os={{.Os}} size={{.Size}}' "$SRC"
# expect: arch=arm64 os=linux size=20342958503

DEST="ghcr.io/<GH_USER>/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark"

docker tag "$SRC" "$DEST:r9-adaptive-full-bae57bd"
docker tag "$SRC" "$DEST:r9"

# ~20 GB. Push the immutable tag first; push the alias only after it succeeds.
docker push "$DEST:r9-adaptive-full-bae57bd"
docker push "$DEST:r9"
```

> `docker tag` creates a new reference to the same local image. It does **not** modify,
> move or delete the internal tag, and it does not touch the running containers. The
> live R9 service is unaffected by every command in this step.

---

## 4. Obtain the registry digest — read it, do not compute it

```bash
# On <BUILD_HOST>, preferred:
docker buildx imagetools inspect "$DEST:r9-adaptive-full-bae57bd" | head -5

# Or, after the push, from RepoDigests:
docker image inspect --format '{{join .RepoDigests "\n"}}' "$DEST:r9-adaptive-full-bae57bd"
```

Take the `sha256:…` that follows the repository name. That is `<GHCR_DIGEST>`.

**Do not** substitute the local image ID (`sha256:50261a39…`). It is the hash of the
image config, not of the registry manifest, and the two are never equal.

---

## 5. Make the package public

GHCR packages default to private even in a public repository.

```bash
# Via the web UI: Packages -> glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark
#   -> Package settings -> Change visibility -> Public
# and, in the same screen, connect the package to the repository.
#
# Or via the API:
gh api -X PATCH \
  /user/packages/container/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark \
  -f visibility=public
```

---

## 6. Patch the documentation and manifest with the real digest

From the repository clone, replacing every `<GHCR_DIGEST>` placeholder:

```bash
DIGEST="sha256:....."   # from step 4

python3 - "$DIGEST" <<'PY'
import json, pathlib, sys
digest = sys.argv[1]
assert digest.startswith("sha256:") and len(digest) == 71, "not a sha256 digest"

manifest = pathlib.Path("release-manifest.json")
data = json.loads(manifest.read_text())
data["image"]["ghcr_digest"] = digest
manifest.write_text(json.dumps(data, indent=2) + "\n")
print("release-manifest.json updated")

for path in (pathlib.Path("README.md"), pathlib.Path("docs/IMAGE.md"),
             pathlib.Path("docs/BUILD.md"), pathlib.Path("SECURITY.md"),
             pathlib.Path("NOTICE.md"), pathlib.Path("runtime/r9.env.example")):
    text = path.read_text()
    if "sha256:<GHCR_DIGEST>" in text:
        path.write_text(text.replace("sha256:<GHCR_DIGEST>", digest))
        print("patched", path)
PY

python3 -m unittest discover -s tests -v
git diff --check
git add -A
git commit -m "docs: pin the published GHCR digest"
git push
```

Also set the `verify-image.sh` reference if you want the digest baked in — it is not
required, since the script accepts any ref.

---

## 7. Tag a GitHub release

```bash
TAG="v1.0.0-r9-adaptive-full-bae57bd"
git tag -a "$TAG" -m "GLM-5.2 R9: adaptive MTP K2/K4/K5 with FULL CUDA graphs, 520K context, 4x DGX Spark"
git push origin "$TAG"

gh release create "$TAG" \
  --repo <GH_USER>/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark \
  --title "R9 — adaptive MTP K2->K4->K5 with FULL CUDA graphs" \
  --notes-file - <<'NOTES'
GLM-5.2 on 4x DGX Spark (GB10, ARM64) with adaptive multi-token-prediction depth
2 -> 4 -> 5 while retaining FULL CUDA graphs for every reachable depth and
concurrency shape, DCP2, NVFP4 KV cache, and a 520,000-token API context.

Measured, with denominators in docs/BENCHMARKS.md:
  * exact 500,000-token cold prompt + 128 output tokens succeeded
  * 514.192 tok/s prefill, 33.430 tok/s decode (ONE isolated concurrency-1 test)
  * 95.107% max KV occupancy, 0 preemptions, no restart or fatal event
  * 525,887 physical KV tokens, 520,000 API context, DCP2
  * FULL graph coverage verified on all nine shapes (3/6/9, 5/10/15, 6/12/18),
    zero PIECEWISE downgrades, zero eager fallbacks, on all four ranks
  * controller demonstrably moved K2/K4/K5: 177/64/783 scheduler steps,
    average selected K 4.256 vs a verified fixed-K5 comparator at 4.965-5.000,
    14.9% of draft positions avoided

Not established: that adaptive depth is faster than fixed K5 in tokens/second.
No valid matched sample exists. See docs/BENCHMARKS.md section 6.

Runtime image (linux/arm64, no model weights):
  ghcr.io/<GH_USER>/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark@<GHCR_DIGEST>

Credit: CosmicRaisins/glm-5.2-gb10, local-inference-lab/vllm, lukealonso/b12x,
vllm-project/vllm, flashinfer-ai/flashinfer, deepseek-ai/DeepGEMM, and the
contributors named in ATTRIBUTION.md.
NOTES
```

Then record the release URL and commit:

```bash
python3 - <<'PY'
import json, pathlib, subprocess
data = json.loads(pathlib.Path("release-manifest.json").read_text())
data["release"]["github_release_tag"] = subprocess.run(
    ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True).stdout.strip()
data["release"]["published_commit"] = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
data["release"]["github_release_url"] = None  # paste from `gh release view --json url`
pathlib.Path("release-manifest.json").write_text(json.dumps(data, indent=2) + "\n")
PY
gh release view "$TAG" --json url --jq .url
```

Fill `github_release_url` and `published_at_utc` from that output, commit, and push.

---

## 8. Verify anonymous pull — from a machine that never held the image

This is the step that actually proves publication worked. Run it somewhere that has
**not** logged in to GHCR and has never built or loaded the image.

```bash
docker logout ghcr.io || true

REF="ghcr.io/<GH_USER>/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark@<GHCR_DIGEST>"

# Manifest-only check first: cheap, and it fails fast if the package is still private.
docker buildx imagetools inspect "$REF"

# Full pull (~20 GB, ARM64 host required to run it; any host can pull it).
docker pull "$REF"

./scripts/verify-image.sh --deep "$REF"
```

Expect `verify-image.sh` to report `0 failed`. Then record it:

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("release-manifest.json")
d = json.loads(p.read_text())
d["release"]["anonymous_pull_verified"] = True
p.write_text(json.dumps(d, indent=2) + "\n")
PY
git commit -am "chore: record verified anonymous pull" && git push
```

---

## 9. Cross-link the prior releases

Add a line pointing here to the READMEs of:

* `https://github.com/<GH_USER>/GLM-5.2-1M-4x-DGX-Spark`
* `https://github.com/<GH_USER>/Keys-GLM-5.2-QuantTrio-655K-MTP-k5-4x-DGX-Spark`
* `https://huggingface.co/<GH_USER>/GLM-5.2-1M-context-NVFP4-4x-DGX-Spark`
* `https://huggingface.co/<GH_USER>/GLM-5.2-QuantTrio-Abliterated`

This repository already links **to** all four.

---

## 10. Rollback

If something is wrong after the push:

```bash
# Make the package private again immediately (fastest mitigation):
gh api -X PATCH /user/packages/container/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark \
  -f visibility=private

# Delete a specific version if necessary:
gh api /user/packages/container/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark/versions
gh api -X DELETE /user/packages/container/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark/versions/<VERSION_ID>
```

**Assume anything published was mirrored.** Treat deletion as damage limitation, not
erasure. This is why the audit gates come before the push and not after.

---

## 11. What this runbook deliberately does not do

* It does not stop, restart, reconfigure or retag anything in the running production
  runtime. Every command in step 3 is additive and local to the Docker image store.
* It does not delete, prune or garbage-collect any image.
* It does not modify model weights, the active-runtime pointer, autostart descriptors,
  health-check timers or any runtime registration.
* It does not publish model weights, and it does not publish the private parent image
  under any tag of its own.
