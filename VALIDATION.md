# Validation record

Everything below was executed. The results are transcribed, not summarized
optimistically. Where a check was skipped or is not possible, it says so.

**Date:** 2026-07-26
**Platform for local checks:** macOS (Darwin, arm64), `bash` 3.2, `python3` 3.9.6,
`shellcheck` (Homebrew), `git`, `docker` CLI.
**Platform for image checks:** the ARM64 build host, via read-only SSH.

---

## 1. Static checks on this repository

| Check | Command | Result |
|---|---|---|
| Shell syntax | `bash -n` on every `*.sh` | **PASS** — `runtime/start-node.sh`, `scripts/verify-image.sh` |
| ShellCheck | `shellcheck --severity=warning` on both scripts | **PASS** — 0 findings |
| ShellCheck (stricter) | `shellcheck --severity=info` on both scripts | **PASS** — 0 findings |
| Python syntax | `python3 -m py_compile` on every `*.py` | **PASS** — all files compile |
| JSON syntax | `json.load` on `release-manifest.json` | **PASS** |
| YAML syntax | `yaml.safe_load` on `.github/workflows/validate.yml` | **PASS** |
| Whitespace / conflict markers | `git diff --check` and `git diff --cached --check` | **PASS** — clean |
| Internal links | offline resolver over every `*.md` | **PASS** — 0 broken; 15 external URLs found and deliberately **not** fetched |

## 2. Test suite

```
$ python3 -m unittest discover -s tests -t .
......................................................
----------------------------------------------------------------------
Ran 54 tests in 0.236s

OK
```

| Module | Tests | Result |
|---|---|---|
| `tests.test_no_private_data` | 10 | **OK** |
| `tests.test_provenance` | 17 | **OK** |
| `tests.test_runtime_templates` | 27 | **OK** |
| **Total** | **54** | **0 failures, 0 errors, 0 skips** |

What those tests actually assert:

* **Provenance** — the copied `patches/` and `provenance/` files hash to pinned
  sha256 values; the sealed manifest has exactly 35 well-formed, unique, relative
  entries; the five R9 delta files' manifest digests equal the values in the image's
  own `org.glm52.exp1.*_sha256` labels; the included applier parses, names all five
  files, and preserves upstream attribution; the image guard parses and still asserts
  the V2-runner exemption; `release-manifest.json` agrees with the manifest and keeps
  the not-yet-known fields `null` rather than guessing them.
* **Hygiene** — a credential-shape sweep over every file; a private-address sweep with
  **no exemptions at all**; an internal-hostname and username sweep with **no
  exemptions at all** (the audit documents use redacted placeholders instead); a
  private-path sweep exempting only the audit documents; a container-ID sweep; a check that every network,
  interface and path value in `runtime/r9.env.example` is a bare `<PLACEHOLDER>`; and
  a check that every git-tracked file was actually covered by the sweep.
* **Runtime templates** — 16 fail-closed cases, each asserting the documented exit
  code fires for its documented condition (including that `08` is rejected as an
  adaptive window because it is valid bash and invalid JSON, and that `03` is rejected
  as `MAX_NUM_SEQS`), plus render-time assertions that a valid configuration emits the
  qualified profile, the adaptive speculative config, the FULL-graph capture sizes, the
  NVFP4 KV dtype, a read-only model mount, no `--enforce-eager`, no API server on
  worker nodes, and a `--speculative-config` that parses as JSON.

## 3. Verification scripts run against the real artifacts

### 3.1 `scripts/verify-image.sh --deep` against the production image

Run on the build host against the local image ID, read-only. Deep probes ran
`docker run --rm --network none` with no GPU request.

```
== platform ==
  PASS  Architecture = arm64
  PASS  Os = linux
  ....  Size: 20342958503 bytes (reference build: 20342958503)

== identity ==
  ....  local image ID (config hash): sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39
  PASS  local image ID matches the reference build
  ....  registry digest: none recorded (image was never pushed or pulled by digest)

== provenance labels ==
  PASS  org.glm52.exp1.source_commit = bae57bd87b03b7c802ca391064996ec27a02d2bb
  PASS  org.glm52.exp1.dockerfile_sha256 = 0758f672f797fe0c2f4812170511821e20384c0d5e0f2e18a8f68df858199a1d
  PASS  org.glm52.exp1.build_script_sha256 = 620cee21545dd570c8664c169214def0dadb453b76272ca5b9b72ccefa2a4633
  PASS  org.glm52.exp1.contract_sha256 = 1b6f00259724f86eb58c047373dabeb9b4df21a1b7bd87995c09e6cfa16bfd77
  PASS  org.glm52.exp1.parent_image_id = sha256:7cc2e13a5f6504bdc31dd173f637239ae587e928b06410dcc8b0d29232a9cb2c
  PASS  org.glm52.exp1.revision = r9-adaptive-full-cuda

== capabilities ==
  PASS  capabilities contains 'adaptive-mtp-depth-245-default-on'
  PASS  capabilities contains 'adaptive-mtp-full-cudagraphs'
  PASS  capabilities contains 'mtp-window-telemetry'
  PASS  capabilities contains 'mtp-k5-default'
  PASS  adaptive_mtp contains 'VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5'
  PASS  adaptive_mtp contains '3,6,9 / 5,10,15 / 6,12,18'
  PASS  adaptive_mtp contains 'fails closed'

== deep probe (CPU-only, --network none, no GPU) ==
  PASS  no model-weight artifacts found in the image
  PASS  /models inside the image = absent
  PASS  /opt/r9/BUILD_PROVENANCE contains 'source_commit=bae57bd87b03b7c802ca391064996ec27a02d2bb'

== result ==
  19 passed, 0 failed
  OK
```

Exit status `0`.

### 3.2 The `docs/BUILD.md` §5 manifest procedure, executed

The documented command was run verbatim against the image:

```
$ docker run --rm --network none -v <manifest>:/tmp/manifest.sha256:ro \
    --entrypoint /bin/sh <image> -c \
    'cd /usr/local/lib/python3.12/dist-packages && sha256sum -c --strict /tmp/manifest.sha256'
```

**35 of 35 files reported `OK`. Exit status `0`.** The sealed manifest published in
this repository is the manifest the running image satisfies.

### 3.3 `scripts/smoke-openai.py` against the live endpoint

Run from inside the private cluster network against the currently-serving R9 runtime.

```
== 1. /health ==
  PASS  /health returned 200

== 2. /v1/models ==
  PASS  served model 'glm-5.2' present
  PASS  max_model_len = 520000

== 3. deterministic generation ==
  PASS  generation returned the sentinel ('R9-SMOKE-OK')
  ....  usage: prompt=22 completion=8

== result ==
  4 passed, 0 failed
  OK
```

Exit status `0`. This confirms the documented 520,000-token API context and that the
deterministic-generation check works end to end against a real server, not a mock.

## 4. Live-state facts confirmed read-only during this work

| Fact | Observed |
|---|---|
| Currently serving image | `sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39` — the image this package publishes |
| Container restarts / OOM | `restarts=0`, `OOMKilled=false` |
| `/v1/models` | `glm-5.2` at `max_model_len` **520000** |
| Live `--kv-cache-memory-bytes` | `8410000000` |
| Live `--kv-cache-dtype` | `nvfp4_ds_mla` |
| Live `--speculative-config` | `num_speculative_tokens: 5`, `adaptive_speculative_tokens_window: 32` |
| Live `--compilation-config` | `cudagraph_capture_sizes: [6,12,18]` |
| Live `VLLM_ADAPTIVE_SPEC_DEPTHS` | `2,4,5` |
| Measured GPU KV cache size in the live log | **525,887 tokens** |
| FULL-graph coverage line in the live log | `FULL CUDA graph coverage verified for query lengths [3, 5, 6] across request counts 1..3`, present on **4/4 ranks** |
| Graph capture in the live log | `Graph capturing finished in 24 secs, took 2.25 GiB` (and 2.14 GiB on the worker ranks) |
| Downgrade markers in the live log | **0** for `enforce_eager=True`, eager-mode disable, `Overriding … PIECEWISE/NONE/FULL_DECODE_ONLY`, `requires PIECEWISE`, `falling back`, `fixed-K fallback` |

**The earlier statement that a fixed-K5 predecessor is the live runtime is stale and is
not repeated anywhere in this package.** R9 is live, on the 520K profile.

## 5. Production safety

No production state was changed at any point while producing this package.

| | Before | After |
|---|---|---|
| Serving container | `glm-exp1-head`, started `2026-07-26T02:54:31Z`, restarts 0 | identical |
| Serving image | R9 `sha256:50261a39…` | identical |
| Local image list | 4 tags (R7, R8, R9-rejected, R9-live) | identical |
| Endpoint health | HTTP 200 | HTTP 200 |

Actions **not** taken: no image was pushed, tagged, saved, exported, loaded, pruned or
deleted; no registry login occurred; no GitHub repository, release or remote was
created; no container was stopped, restarted, recreated or reconfigured; no
`active-runtime` pointer, autostart descriptor, timer or model byte was touched. Three
temporary files copied to the build host's `/tmp` for the verification runs above were
removed afterwards.

## 6. Checks that were NOT possible, and why

| Check | Why not |
|---|---|
| Anonymous `docker pull` of the published image | The image has not been pushed. `RepoDigests` is `[]`. This is step 8 of [`PUBLISHING.md`](PUBLISHING.md) and belongs to the operator performing the publication. |
| Registry digest verification | Same reason — the digest does not exist yet, which is why `release-manifest.json` carries `null` rather than a placeholder that looks like a value. |
| External link liveness | Deliberately not fetched, locally or in CI. A validation job that fails because a third party's server was slow is a job people learn to ignore. The 15 external URLs are enumerated by the CI step for human review. |
| Running the CI workflow | GitHub Actions cannot run before the repository exists. Every step in `validate.yml` was, however, run by hand here, and each corresponds to a check in §1–§2 above. |
| Full public rebuild of the image | Not possible — the immediate build parent is private and the earlier chain no longer exists. This is stated plainly in [`docs/BUILD.md`](docs/BUILD.md) §4 rather than implied away. |
| Multi-node launch of `runtime/start-node.sh` | Requires four idle GB10 nodes. The templates were validated by rendering (`RENDER_ONLY=1`) and by 16 fail-closed behavioural tests; they were not used to bring up a cluster. |

## 7. Known limitations of this validation

* The launcher tests validate **rendering and refusal**, not a real four-node bring-up.
  A template that renders correctly can still be wrong about something only a live
  cluster reveals.
* `verify-image.sh` was run against the **local** image, not a registry-pulled one.
  The label and architecture assertions are the same either way, but the round trip
  through GHCR is unproven until step 8 of `PUBLISHING.md`.
* The secret scan is pattern-based. It catches the common credential shapes and every
  private identifier known to this deployment; it cannot prove the absence of a secret
  in a shape nobody anticipated.
* `tests/test_no_private_data.py` is exempt from its own identifier and container-ID
  sweeps, because it necessarily contains the literals it bans. It is **not** exempt
  from the credential or private-address sweeps.
