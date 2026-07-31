#!/usr/bin/env bash
# GLM-5.2 R13 adaptive-MTP / FULL-CUDA — public four-node launch template.
#
# SPDX-License-Identifier: Apache-2.0
#
#   NODE_ROLE=head   ./runtime/start-node.sh runtime/r9.env
#   NODE_ROLE=worker ./runtime/start-node.sh runtime/r9.env
#   RENDER_ONLY=1 NODE_ROLE=head ./runtime/start-node.sh runtime/r9.env
#
# This script FAILS CLOSED. It renders nothing and launches nothing unless every
# placeholder is replaced and every invariant of the qualified runtime holds.
# The invariants are not stylistic: the twelve-shape FULL CUDA graph coverage set
# is a function of the ladder and MAX_NUM_SEQS, and the load-time assertion
# inside the server will refuse to start if they disagree.
#
# R13 vs R9: concurrency 3 -> 4 (twelve-shape capture set [6,12,18,24]) and a
# DCP-comm-arg guard so a DCP1 (fast) profile boots clean. Both profiles share one
# image; the profile selects the context/KV/DCP envelope. See RELEASE_NOTES.md.
#
# It is a TEMPLATE. It is deliberately smaller than the production launcher,
# which additionally owns container preservation, a rollback transaction and a
# deadline watchdog. Read it before you run it.

set -euo pipefail

die() { printf 'refusing to run: %s\n' "$1" >&2; exit "${2:-1}"; }

ENV_FILE="${1:-}"
[[ -n "$ENV_FILE" ]] || die "usage: NODE_ROLE=head|worker $0 <env-file>" 11
[[ -r "$ENV_FILE" ]] || die "env file not readable: $ENV_FILE" 11

set -a
# shellcheck source=/dev/null
. "$ENV_FILE"
set +a

NODE_ROLE="${NODE_ROLE:-}"
RENDER_ONLY="${RENDER_ONLY:-0}"

case "$NODE_ROLE" in
  head|worker) ;;
  *) die "NODE_ROLE must be exactly 'head' or 'worker' (got '${NODE_ROLE}')" 11 ;;
esac

# --------------------------------------------------------------------------
# Gate 11 — every placeholder must be replaced.
# --------------------------------------------------------------------------
REQUIRED_VARS=(
  IMAGE_REF CONTAINER_NAME
  HEAD_NODE_IP NODE_IP RAY_PORT API_BIND_ADDR API_PORT
  NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA
  MODEL_DIR JIT_CACHE_DIR
  R13_PROFILE MAX_NUM_SEQS MTP_K VLLM_ADAPTIVE_SPEC_DEPTHS ADAPTIVE_SPEC_WINDOW
  CUDAGRAPH_SIZES VLLM_MTP_INSTRUMENT VLLM_MTP_INSTRUMENT_WINDOW
  MAX_NUM_BATCHED_TOKENS LONG_PREFILL_TOKEN_THRESHOLD
  DECODE_PREFILL_TOKEN_BUDGET IDLE_PREFILL_TOKEN_BUDGET
  MAX_LONG_PREFILLS_PER_STEP
  TENSOR_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE
  DCP_COMM_BACKEND SERVED_MODEL_NAME
)
for v in "${REQUIRED_VARS[@]}"; do
  val="${!v-}"
  [[ -n "$val" ]] || die "$v is unset or empty in $ENV_FILE" 11
  case "$val" in
    *"<"*">"*) die "$v still contains a placeholder: $val" 11 ;;
  esac
done

# --------------------------------------------------------------------------
# Gate 2 / 4 / 5 / 6 / 7 / 8 / 10 — the qualified runtime invariants.
# Exit codes mirror the production launcher so receipts are comparable.
# --------------------------------------------------------------------------
is_bare_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

is_bare_positive_int "$MTP_K" || die "MTP_K must be a bare positive integer" 2

# '08' is valid bash and invalid JSON; a leading zero is rejected on purpose.
is_bare_positive_int "$ADAPTIVE_SPEC_WINDOW" \
  || die "ADAPTIVE_SPEC_WINDOW must be a bare positive integer (no leading zeros)" 4

LADDER_NORM="$(printf '%s' "$VLLM_ADAPTIVE_SPEC_DEPTHS" | tr -d '[:space:]')"
[[ "$LADDER_NORM" == "2,4,5" ]] \
  || die "VLLM_ADAPTIVE_SPEC_DEPTHS must normalise to exactly '2,4,5' (got '$LADDER_NORM')" 5

LADDER_TOP="${LADDER_NORM##*,}"
[[ "$LADDER_TOP" == "$MTP_K" ]] \
  || die "the ladder's top rung ($LADDER_TOP) must equal MTP_K ($MTP_K)" 6
[[ "$MTP_K" == "5" ]] \
  || die "this build is qualified at MTP_K=5 only" 6

[[ "${ENFORCE_EAGER:-0}" == "0" ]] \
  || die "ENFORCE_EAGER=1 defeats the entire point of this build" 7

[[ "$MAX_NUM_SEQS" == "4" ]] \
  || die "MAX_NUM_SEQS must be exactly 4; it defines the twelve-shape FULL graph coverage set" 10

is_bare_positive_int "$VLLM_MTP_INSTRUMENT_WINDOW" \
  || die "VLLM_MTP_INSTRUMENT_WINDOW must be a bare positive integer" 8
[[ "$VLLM_MTP_INSTRUMENT" == "1" ]] \
  || die "MTP telemetry is mandatory in this build" 8

# Gate 7 — the capture sizes must cover every (depth+1) x n shape.
# ladder 2,4,5 and MAX_NUM_SEQS=4 give query lengths 3,5,6 and the twelve shapes
# 3/6/9/12, 5/10/15/20, 6/12/18/24. vLLM captures per uniform query length, so
# what the launcher must guarantee is that the largest shape fits the largest
# capture size and that the capture list is exactly the qualified set.
CAPTURE_NORM="$(printf '%s' "$CUDAGRAPH_SIZES" | tr -d '[:space:]')"
[[ "$CAPTURE_NORM" == "6,12,18,24" ]] \
  || die "CUDAGRAPH_SIZES must be exactly '6,12,18,24' for the qualified twelve-shape set (got '$CAPTURE_NORM')" 7

MAX_CAPTURE="${CAPTURE_NORM##*,}"
MAX_SHAPE=$(( (MTP_K + 1) * MAX_NUM_SEQS ))
(( MAX_SHAPE <= MAX_CAPTURE )) \
  || die "largest reachable shape ($MAX_SHAPE) exceeds max capture size ($MAX_CAPTURE)" 7

# -------------------------------------------------------------------------- #
# Profile resolution — context, KV budget and DCP size are a matched triple,
# picked by name. The context must sit below the KV capacity the byte budget
# buys, and the byte budget must leave enough unified memory for the twelve-shape
# FULL graph capture. The profile also fixes DCP_SIZE; see the DCP guard below.
# -------------------------------------------------------------------------- #
case "$R13_PROFILE" in
  fast)
    MAX_MODEL_LEN=319000
    KV_CACHE_MEMORY_BYTES=10233000000
    DCP_SIZE=1
    ;;
  balanced)
    MAX_MODEL_LEN=520000
    KV_CACHE_MEMORY_BYTES=8410000000
    DCP_SIZE=2
    ;;
  *)
    cat >&2 <<'GATE'
refusing to run: R13_PROFILE must be one of the authorized named profiles.

R13 ships two matched envelopes. context, KV budget and DCP size are a
matched triple, not three knobs: the context must sit below the KV capacity
the byte budget buys, and the byte budget must leave enough unified memory for
the twelve-shape FULL graph capture.

  fast       max_model_len=319000  kv_cache_memory_bytes=10233000000  DCP1
                    short-context throughput. Smaller API context, larger per-rank
                    KV budget, no DCP comm layer. Highest prefill / C4 aggregate.
                    Measured: ~695 tok/s prefill, ~83 tok/s C4 aggregate.
  balanced   max_model_len=520000  kv_cache_memory_bytes=8410000000    DCP2
                    long-context capacity. Larger API context, smaller per-rank KV
                    budget, DCP2 comm layer. Faster prose decode C1 (~31 tok/s vs
                    ~23) at the cost of lower C4 aggregate (~72 tok/s).
                    Measured: ~602 tok/s prefill.

Both profiles are C4 (MAX_NUM_SEQS=4, twelve-shape capture set 6,12,18,24).
The tradeoffs are in docs/BENCHMARKS.md. Pick the profile that matches the
workload; do not mix the three numbers by hand.

Nothing was touched.
GATE
    exit 12
    ;;
esac

# -------------------------------------------------------------------------- #
# DCP comm backend is only valid for DCP>1.
# Newer vLLM rejects an explicit --dcp-comm-backend at
# decode-context-parallel-size 1 (it would crash the engine on boot). DCP1 omits
# both DCP comm flags entirely; DCP2 passes them. This is the R13 fix.
# -------------------------------------------------------------------------- #
if [[ "${DCP_SIZE}" -gt 1 ]]; then
  [[ -n "$DCP_COMM_BACKEND" ]] \
    || die "DCP_COMM_BACKEND is required when the profile uses DCP>1" 6
  DCP_ARGS=(--dcp-comm-backend "$DCP_COMM_BACKEND" --dcp-kv-cache-interleave-size 1)
else
  DCP_COMM_BACKEND=""
  DCP_ARGS=()
fi

# --------------------------------------------------------------------------
# Host preconditions.
# --------------------------------------------------------------------------
[[ -d "$MODEL_DIR" ]]     || die "MODEL_DIR does not exist: $MODEL_DIR (download GLM-5.2 first)" 13
[[ -d "$JIT_CACHE_DIR" ]] || die "JIT_CACHE_DIR does not exist: $JIT_CACHE_DIR" 13
command -v docker >/dev/null 2>&1 || die "docker not found on PATH" 13

# --------------------------------------------------------------------------
# Render.
# --------------------------------------------------------------------------
CACHE_ROOT="/cache"

DOCKER_ENV=(
  -e "VLLM_HOST_IP=${NODE_IP}"
  -e "RAY_ADDRESS=${HEAD_NODE_IP}:${RAY_PORT}"
  -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  -e "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
  -e "NCCL_IB_HCA=${NCCL_IB_HCA}"
  -e "NCCL_IB_DISABLE=0"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_MIN_NCHANNELS=4"
  -e "NCCL_MAX_NCHANNELS=4"
  -e "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  -e "CUDA_DEVICE_MAX_CONNECTIONS=32"
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  -e "SAFETENSORS_FAST_GPU=1"
  -e "VLLM_USE_V2_MODEL_RUNNER=1"
  -e "VLLM_USE_B12X_SPARSE_INDEXER=1"
  -e "VLLM_USE_FLASHINFER_SAMPLER=1"
  -e "VLLM_MARLIN_USE_ATOMIC_ADD=1"
  -e "VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0"
  -e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256"
  -e "VLLM_ENABLE_PCIE_ALLREDUCE=0"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "VLLM_WORKER_MULTIPROC_METHOD=spawn"
  -e "VLLM_ADAPTIVE_SPEC_DEPTHS=${LADDER_NORM}"
  -e "VLLM_MTP_INSTRUMENT=${VLLM_MTP_INSTRUMENT}"
  -e "VLLM_MTP_INSTRUMENT_WINDOW=${VLLM_MTP_INSTRUMENT_WINDOW}"
  -e "VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm"
  -e "B12X_CUTE_COMPILE_CACHE_DIR=${CACHE_ROOT}/b12x-cute"
  -e "TRITON_CACHE_DIR=${CACHE_ROOT}/triton"
  -e "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor"
  -e "TORCH_EXTENSIONS_DIR=${CACHE_ROOT}/torch-extensions"
  -e "RAY_memory_monitor_refresh_ms=0"
  -e "RAY_memory_usage_threshold=0.99"
)

DOCKER_RUN=(
  docker run -d --name "$CONTAINER_NAME"
  --network host --ipc host --gpus all
  --restart no
  -v "${MODEL_DIR}:/models:ro"
  -v "${JIT_CACHE_DIR}:${CACHE_ROOT}:rw"
  "${DOCKER_ENV[@]}"
  "$IMAGE_REF"
  sleep infinity
)

# DCP args are emitted ONLY when DCP>1. At DCP1 the engine rejects an explicit
# comm backend (it crashes on boot); --dcp-kv-cache-interleave-size alongside a
# DCP1 size is also inconsistent. Both flags are omitted for the fast profile.
# DCP_ARGS is set by the guard above the profile block and is empty at DCP1.

# hf-overrides: the GB10 sparse-attention requirement established by
# CosmicRaisins/glm-5.2-gb10. F = full attention layer, S = sparse.
HF_OVERRIDES='{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
SPEC_CONFIG="{\"model\":\"/models\",\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K},\"moe_backend\":\"flashinfer_cutlass\",\"draft_attention_backend\":\"B12X_MLA_SPARSE\",\"draft_sample_method\":\"probabilistic\",\"adaptive_speculative_tokens_window\":${ADAPTIVE_SPEC_WINDOW}}"
COMPILATION_CONFIG="{\"cudagraph_capture_sizes\":[${CAPTURE_NORM}]}"

SERVER_ARGV=(
  python3 -m vllm.entrypoints.openai.api_server
  --model /models
  --tokenizer /models
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --download-dir /models
  --load-format auto
  --quantization compressed-tensors
  --distributed-executor-backend ray
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --decode-context-parallel-size "$DCP_SIZE"
  ${DCP_ARGS[@]+"${DCP_ARGS[@]}"}
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
  --gpu-memory-utilization 0.88
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --generation-config vllm
  --hf-overrides "$HF_OVERRIDES"
  --host "$API_BIND_ADDR"
  --port "$API_PORT"
  --no-enable-log-requests
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  --kv-cache-dtype nvfp4_ds_mla
  --attention-backend B12X_MLA_SPARSE
  --moe-backend flashinfer_cutlass
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --speculative-config "$SPEC_CONFIG"
  --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD"
  --async-scheduling
  --enable-decode-aware-prefill
  --decode-prefill-token-budget "$DECODE_PREFILL_TOKEN_BUDGET"
  --idle-prefill-token-budget "$IDLE_PREFILL_TOKEN_BUDGET"
  --max-long-prefills-per-step "$MAX_LONG_PREFILLS_PER_STEP"
  --compilation-config "$COMPILATION_CONFIG"
  --enable-prefix-caching
)

if [[ "$NODE_ROLE" == "head" ]]; then
  RAY_ARGV=(ray start --head --node-ip-address "$NODE_IP" --port "$RAY_PORT" --num-gpus 1 --block)
else
  RAY_ARGV=(ray start --address "${HEAD_NODE_IP}:${RAY_PORT}" --node-ip-address "$NODE_IP" --num-gpus 1 --block)
fi

render() {
  printf '# ---- container (%s) ----\n' "$NODE_ROLE"
  printf '%q ' "${DOCKER_RUN[@]}"; printf '\n\n'
  printf '# ---- ray (%s), inside the container ----\n' "$NODE_ROLE"
  printf 'docker exec -d %q ' "$CONTAINER_NAME"; printf '%q ' "${RAY_ARGV[@]}"; printf '\n\n'
  if [[ "$NODE_ROLE" == "head" ]]; then
    printf '# ---- vLLM API server, head node only, AFTER all 4 ranks have joined ----\n'
    printf 'docker exec -d %q ' "$CONTAINER_NAME"; printf '%q ' "${SERVER_ARGV[@]}"; printf '\n'
  else
    printf '# ---- worker node: no API server is started here ----\n'
  fi
}

if [[ "$RENDER_ONLY" == "1" ]]; then
  render
  exit 0
fi

printf 'starting %s node with profile %s (max_model_len=%s, kv_cache_memory_bytes=%s)\n' \
  "$NODE_ROLE" "$R9_PROFILE" "$MAX_MODEL_LEN" "$KV_CACHE_MEMORY_BYTES" >&2

"${DOCKER_RUN[@]}"
docker exec -d "$CONTAINER_NAME" "${RAY_ARGV[@]}"

if [[ "$NODE_ROLE" == "worker" ]]; then
  printf 'worker joined. Start the API server on the head node once all 4 ranks are up.\n' >&2
  exit 0
fi

cat >&2 <<'WAIT'

Head node container and Ray head are running.

Confirm all four ranks have joined BEFORE starting the server -- launching with a
partial cluster wastes a full model load:

    docker exec <CONTAINER_NAME> ray status

Then start the API server:

    docker exec -d <CONTAINER_NAME> <the vLLM argv printed by RENDER_ONLY=1>

Then smoke-test:

    python3 scripts/smoke-openai.py --base-url http://<HEAD_NODE_IP>:<API_PORT>

WAIT
