# Attribution

This runtime is a **five-file source delta** on top of a stack built almost entirely by
other people. Full credit to them. Nothing in this repository should be read as a claim
over the projects below.

If you are listed here and want a correction, a different attribution, or removal,
open an issue and it will be fixed.

---

## 1. Foundational work

### CosmicRaisins / glm-5.2-gb10

<https://github.com/CosmicRaisins/glm-5.2-gb10>

Pioneered GLM-5.2 serving on DGX Spark (GB10, `sm_121`). Everything in this repository
sits on top of that work, including:

* the `index_topk_pattern` `hf-overrides` requirement,
* the DCP stack patches (PR #72 — draft config propagation, `topk_scores_buffer` for
  B12X, `build_for_drafting`),
* the serving configuration `VLLM_USE_V2_MODEL_RUNNER=1` +
  `VLLM_USE_B12X_SPARSE_INDEXER=1` + `--attention-backend B12X_MLA_SPARSE`,
* the `draft-quant-packed-mapping` fix,
* **and the tuned 2/4/5 adaptive-depth policy and multi-depth CUDA-graph hooks that
  are this release's entire headline feature.**

Pinned at commit `600848707ce93fe42fedbc9dd4429116696e425d`. Files used:

| File | How it is used here |
|---|---|
| `adaptive-mtp/overlay/vllm/v1/spec_decode/dynamic/acceptance_length.py` | applied **byte-for-byte** from `from dataclasses import dataclass` onward; the only edits are a `from __future__ import annotations` and a provenance header |
| `patches/adaptive-mtp-vllm-hooks.patch` | **reconciled, not applied verbatim** — see §4 |
| `adaptive-mtp/README.md` | policy documentation and the `max_num_seqs=1` tuning caveat |

* PR #72: <https://github.com/CosmicRaisins/glm-5.2-gb10/pull/72>

### local-inference-lab / vllm — Luke Alonso

<https://github.com/local-inference-lab/vllm>

The vLLM fork this image installs, and the origin of the adaptive-speculative-depth
feature the community policy is built on: feature commit
`d179dc83755ca7365a6c1b1294c74d7908106bc7`. Installed build reports upstream commit
`e232d262369b8c918cf478a7a96a0fcf8127cf65`.

### Aiden Le (aidendle94)

Credited by the pinned sources as the author of the **acceptance-length controller
foundation** that the 2/4/5 ladder replaces the policy of.

### lukealonso / b12x

<https://github.com/lukealonso/b12x>

The `b12x` sparse-MLA kernels and integration (`b12x.integration.mla`) that the
`B12X_MLA_SPARSE` backend is. Installed 0.30.x; commit
`97b3d642b8ce08ce23184a36882710ce3b60ba13` recorded in the image labels.

---

## 2. Upstream sources carried in the image

| Project | Repository | Pin as recorded in the image | Licence |
|---|---|---|---|
| vLLM (fork) | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) | installed build reports upstream `e232d262369b8c918cf478a7a96a0fcf8127cf65`; version string `0.1.dev17863+ge232d2623.exp1sm121a368r4dtypefix` | Apache-2.0 |
| vLLM (upstream) | [vllm-project/vllm](https://github.com/vllm-project/vllm) | the project the fork tracks; source of several fixes below | Apache-2.0 |
| b12x | [lukealonso/b12x](https://github.com/lukealonso/b12x) | `97b3d642b8ce08ce23184a36882710ce3b60ba13`, installed 0.30.x | Apache-2.0 |
| FlashInfer | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | prebuilt `sm_121` wheels; build metadata records `f2f9646ec388d9f178b2fbda6ae0ec4246d8e7dc` | Apache-2.0 |
| DeepGEMM | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | `nv_dev` | MIT |
| NCCL (fork) | [zyang-dev/nccl](https://github.com/zyang-dev/nccl) | `dgxspark-3node-ring`; a live-loaded `libnccl.so.2.30.7` was used in earlier revisions of this stack | BSD-3-Clause |
| NCCL (upstream, in base image) | [NVIDIA/nccl](https://github.com/NVIDIA/nccl) | `2.28.3-1+cuda13.0` packaged in the CUDA base | BSD-3-Clause |
| Docker / deploy scaffolding | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | latest at the time of the original base build | Apache-2.0 |
| CUDA base image | NVIDIA CUDA 13.0.2 container (`ubuntu 24.04`, `sbsa`) | see `/NGC-DL-CONTAINER-LICENSE` in the image | **NVIDIA Deep Learning Container License** — see [`NOTICE.md`](NOTICE.md) |
| GLM-5.2 weights (quantized) | [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix) | **not shipped in the image** | MIT as stated on the model card — verify on the page before use |
| GLM-5.2 (base model) | [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | **not shipped in the image** | see the model card |

Licence labels above are recorded as stated by each upstream at the time of writing.
They are a convenience, not legal advice — check each project's own `LICENSE` before
you rely on it.

---

## 3. Upstream fixes and patches carried forward from earlier revisions

These are not R9's work. They arrived in the image via the R6 → R7 → R8 chain and are
listed so the full derivation is visible.

### Sparse-MLA indexer block-table alignment

Aligns the sparse-MLA indexer's block-table width to the runner's
`128 / block_size` multiple. Relevant upstream discussion:

* <https://github.com/vllm-project/vllm/issues/46074>
* <https://github.com/vllm-project/vllm/issues/44827>
* <https://github.com/vllm-project/vllm/pull/43970>
* <https://github.com/vllm-project/vllm/pull/48404>
* the asymmetry introduced by <https://github.com/vllm-project/vllm/pull/39324>
  (commit `38e16678`)

### B12X stale top-k indices buffer

Backports **Fix #4 of <https://github.com/vllm-project/vllm/pull/46994>** by
**eastwood-c**, as curated by **Ciprian / Light Foundry** in the `gb10-glm-5.2`
project.

### Decode-aware prefill scheduler

Original patch, on-hardware validation, test report and tuning data by
**penguinchang**, published on the NVIDIA Developer Forums (2026-07-15):

* <https://forums.developer.nvidia.com/t/glm-5-2-int4-int8-on-8x-gb10-1-200-t-s-prefill-33-54-t-s-avg-decode-generic-coding-structured/376831>

The fork adaptation (targeted string replacements robust to the fork's spec-decode
padding block) is **Ciprian / Light Foundry**'s
`mods/decode-aware-scheduler/apply_scheduler_patch.py`.

### A2A CUDA-graph buffer lifetime

Commit `639aeff84c9b7ad822be6fd4e477c5a5bf4235b8`, *"fix dcp a2a buffers across full
cuda graphs"*, by **Martin Vit (voipmonitor)** — PR #117 against the vLLM fork
[local-inference-lab/vllm](https://github.com/local-inference-lab/vllm). The commit
SHA is given rather than a constructed link because this repository has not verified a
public URL for that fork PR.

### Deterministic CuTe DSL cache key

b12x commit `c7dc73322cc50609f843fa2bbcc53283a90003b3`, *"Fix CuTe DSL option cache key
stability"*, also by **Martin Vit (voipmonitor)**, in
[lukealonso/b12x](https://github.com/lukealonso/b12x).

### FSM tool-call correctness under MTP

Upstream **<https://github.com/vllm-project/vllm/pull/44993>**. The minimal fork
adaptation is **Ciprian / Light Foundry**'s v18 mod `fix-fsm-toolcall-v18`.

### DCP FP32 LSE contract / capture-warmup LSE

Fork commits `f05603fa287ab020acc5faafd49c5decf0762533` and
`b64b0086a12d727ebb9ac72a9c86acfbdf1f0911`, recorded in the image's
`org.glm52.exp1.upstream_commits` label.

---

## 4. What R9 itself contributes

Five files, and two deliberate reconciliations of the pinned community patch. Both
reconciliations exist because applying the patch verbatim would have defeated the
feature:

**(a) One ladder parser instead of two.** The pinned patch parses
`VLLM_ADAPTIVE_SPEC_DEPTHS` twice — once in `scheduler.py`, once in
`cudagraph_utils.py` — and the two copies disagree: the scheduler unions
`num_speculative_tokens` into its snap points and the CUDA-graph side does not. Any
ladder whose top rung is below the configured maximum therefore lets the scheduler
select a depth the graph layer never captured a descriptor for — the exact
silent-eager failure this release exists to remove. R9 ships one parser,
`vllm/v1/spec_decode/dynamic/depth_ladder.py`, imported by both call sites.

**(b) Layered candidate ranges instead of merged ones.** The pinned patch merges the
per-depth descriptors into the base `descs_by_token_lora`, which re-partitions the
token-count ranges. Under the `FULL_AND_PIECEWISE` default this silently *removes*
mixed-batch coverage: introducing a bucket at 5 tokens makes a 4-token prefill resolve
to a bucket holding only uniform-5 decode descriptors, which no mixed batch is
compatible with, so it drops to eager instead of the PIECEWISE graph it used to get.
R9 builds the base ranges unchanged and **prepends** the per-depth descriptors, so
every base lookup stays reachable.

Beyond those: the fail-closed load-time assertions, the `MTP_WINDOW_JSON` telemetry
schema and its explicit denominator semantics, the launcher's fail-closed gates, and
the sealed file manifests.

---

## 5. `research/gb10-glm-5.2` — a local tree with no public URL

Several patches in the R7/R8 chain are recorded as being ported from a local working
directory called `research/gb10-glm-5.2`. **That path is a local checkout on the build
machine and has no public repository URL.** No URL is asserted for it here, and none
should be constructed.

Accurately stated: it is a working copy of the **`gb10-glm-5.2` project curated by
Ciprian / Light Foundry**, whose public counterpart for the adaptive-MTP work used in
this release is [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10)
at the pinned commit `600848707ce93fe42fedbc9dd4429116696e425d`. This repository does
**not** claim that the local tree is byte-identical to any public commit other than
that pinned adaptive-MTP source, which was verified file-by-file.

---

## 6. Explicitly not claimed

* **A2A DCP itself.** The `dcp_comm_backend="a2a"` path and `dcp_a2a_lse_reduce` are
  already present in the pinned vLLM fork. This work selects `a2a` at runtime and
  carries one upstream fix on top; it wrote no A2A code.
* **Sparse-MLA decode/extend kernels.** `b12x.integration.mla`, by the b12x authors.
* **`VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE`.** Already an env toggle in the fork. This
  work only wires it explicitly and documents why it must stay `0` under MTP: with
  MTP a verification batch has `k + 1` query rows per request, so routing those into
  the sparse-MLA *decode* kernel — which handles independent one-token query rows only
  — would stop later draft rows attending to earlier draft rows in the same verifier
  batch. It is a correctness regression, not a speed lever.
* **The acceptance-length controller machinery.** Foundation credited to
  Aiden Le / aidendle94 and Luke Alonso; the tuned 2/4/5 policy to
  CosmicRaisins/glm-5.2-gb10.

## 7. Deliberately not ported

Recorded in the image's `org.glm52.exp1.upstream_commits_excluded` label: the b12x PCIe
A2A fastpath (`7e5efa61a`), variable-width speculation, async draft output, DSpark,
DFlash block capacity, CausalCascade, `draft_gumbel_pos`, EPLB injection, local-argmax
reduction, and the two community-patch behaviours reconciled in §4.

---

## 8. Licence of this repository's own material

Apache-2.0 — see [`LICENSE`](LICENSE). Files derived from upstream projects retain
their upstream licences and SPDX headers; see [`NOTICE.md`](NOTICE.md).
