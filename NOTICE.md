# NOTICE

`GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark`
Copyright 2026 0xdfi.

Material originally authored in this repository is licensed under the Apache
License, Version 2.0 — see [`LICENSE`](LICENSE). This NOTICE records the third-party
terms that travel with the published container image and with the derived source
included here. It is a factual record of obligations, not legal advice.

---

## 1. NVIDIA — required notice

**This software contains source code provided by NVIDIA Corporation.**

The published container image is a derived work of an NVIDIA CUDA container
(CUDA 13.0.2, `ubuntu` 24.04, `sbsa`/ARM64). The **NVIDIA Deep Learning Container
License** ships inside the image at `/NGC-DL-CONTAINER-LICENSE` and is the
authoritative text. Read it before redistributing the image.

Points that materially affect anyone who redistributes this image:

* The licence contemplates distributing a **derived container** that includes the
  entire original container plus other software with primary functionality
  (§1c). This image qualifies: it adds a vLLM/b12x serving stack.
* Distribution requirements (§2) include that the service or application have
  **material additional functionality beyond the included portions of the
  container**, that the notice above be included in distributed modifications and
  derivative works of source code, and that onward distribution be **subject to
  terms at least as protective** as NVIDIA's.
* The licence prohibits distributing or sublicensing the container **as a
  stand-alone product** (§4c) and prohibits removing copyright or proprietary
  notices (§4b).
* Under the licence's own definition, "distribution" also means **deploying the
  container in a service or application for third parties to access over the
  internet**. If you put this image behind a public API, that is distribution and
  the requirements above apply to you.

If you redistribute, re-serve, or re-publish this image, those obligations pass to
you. This repository does not and cannot waive them.

The image also contains NVIDIA proprietary components (CUDA runtime libraries,
cuBLAS, cuSPARSE, NPP, NCCL packages, Nsight Compute). Those are licensed only to run
on systems with NVIDIA GPUs.

## 2. vLLM

Portions of the runtime, and the source emitted by
[`patches/13_r9_adaptive_full_cuda.py`](patches/13_r9_adaptive_full_cuda.py), are
derived from the vLLM project and carry:

```
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

Licensed under the Apache License, Version 2.0.
Upstream: <https://github.com/vllm-project/vllm> · fork:
<https://github.com/local-inference-lab/vllm> · base commit
`e232d262369b8c918cf478a7a96a0fcf8127cf65`.

## 3. b12x

<https://github.com/lukealonso/b12x> — Apache License, Version 2.0.

## 4. CosmicRaisins / glm-5.2-gb10

<https://github.com/CosmicRaisins/glm-5.2-gb10> — the adaptive-MTP controller applied
in this release is taken from that project at commit
`600848707ce93fe42fedbc9dd4429116696e425d` and is applied byte-for-byte apart from a
`from __future__ import annotations` and a provenance header. Consult that repository
for its own licence terms.

## 5. Other bundled components

| Component | Upstream | Licence as stated upstream |
|---|---|---|
| FlashInfer | <https://github.com/flashinfer-ai/flashinfer> | Apache-2.0 |
| DeepGEMM | <https://github.com/deepseek-ai/DeepGEMM> | MIT |
| NCCL | <https://github.com/NVIDIA/nccl>, fork <https://github.com/zyang-dev/nccl> | BSD-3-Clause |
| PyTorch, Triton, and the wider Python dependency tree | various | see each package's own `*.dist-info/licenses` inside the image |
| Docker/deploy scaffolding | <https://github.com/eugr/spark-vllm-docker> | Apache-2.0 |
| tiktoken encodings fetched at base-build time | `openaipublic.blob.core.windows.net` | see the tiktoken project's terms |

A complete per-package licence set is present **inside the image** under each
distribution's `*.dist-info/` directory. To enumerate it without a GPU:

```bash
docker run --rm --network none --entrypoint /bin/sh \
  ghcr.io/0xdfi/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark@sha256:<GHCR_DIGEST> \
  -c 'ls -d /usr/local/lib/python3.12/dist-packages/*.dist-info'
```

## 6. Model weights — not distributed here

GLM-5.2 weights are **not** included in the image and are **not** covered by anything
in this repository. Obtain them from their sources and accept their terms:

* <https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix>
* <https://huggingface.co/zai-org/GLM-5.2>

## 7. Measurement claims

Every performance number published in this repository was measured on the four-node
DGX Spark deployment described in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), under the
denominators and caveats stated there. No claim is made that those numbers generalize
to other hardware, other topologies, other workloads, or other concurrency levels.
