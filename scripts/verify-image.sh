#!/usr/bin/env bash
# Verify a pulled GLM-5.2 R13 adaptive-MTP / FULL-CUDA image.
#
# SPDX-License-Identifier: Apache-2.0
#
#   ./scripts/verify-image.sh <image-ref>
#   ./scripts/verify-image.sh ghcr.io/0xdfi/...@sha256:<digest>
#   ./scripts/verify-image.sh --deep <image-ref>     # also probes the filesystem
#
# Checks architecture, OS, the expected OCI/provenance labels, the adaptive-MTP
# capability strings, and (with --deep) the absence of model weights.
#
# NAMING / LINEAGE: this is the R13 public release. The image it verifies was
# built and tagged internally as `r9.1-scheduler-liveness-4lane` (its
# `org.glm52.exp1.revision` label, which is what docker inspect returns and what
# EXP_REVISION below pins). The C4 / twelve-shape behavior is a runtime
# configuration applied by start-node.sh, not an image rebuild, so the image's
# baked-in labels still describe the C3 nine-shape set and this script accounts
# for that. The image ID (sha256:6d7b06b1…) is the integrity anchor.
#
# HONESTY NOTE, deliberately not softened: the local Docker image ID is the hash
# of the image CONFIG and is computed locally. The registry digest
# (`repo@sha256:...`) is a hash of the MANIFEST and is assigned by the registry.
# They are different hashes over different objects and neither can be derived
# from the other. This script prints both and never claims they are equal.

set -uo pipefail

DEEP=0
if [[ "${1:-}" == "--deep" ]]; then DEEP=1; shift; fi

IMAGE_REF="${1:-}"
if [[ -z "$IMAGE_REF" ]]; then
  echo "usage: $0 [--deep] <image-ref>" >&2
  exit 64
fi

command -v docker >/dev/null 2>&1 || { echo "docker not found on PATH" >&2; exit 64; }

PASS=0; FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
note() { printf '  ....  %s\n' "$1"; }
# --- expected values, from the sealed build ---------------------------------
# R13 public release. The image's INTERNAL build tag is r9.1-scheduler-liveness
# 4lane (its org.glm52.exp1.revision label); EXP_REVISION pins that exact string
# because it is what `docker inspect` returns and what this script checks against.
# The image is built on the SAME source commit as R9 (bae57bd); the source tree,
# Dockerfile, build script and contract are byte-identical to the R9 build. What
# changed is the revision label, the parent (this image layers on top of the R9
# image, which is itself layered on the R8 base), and the resulting image ID. The
# C4/twelve-shape behavior is a RUNTIME configuration applied by start-node.sh,
# not a rebuild — the baked-in adaptive_mtp / capabilities labels still describe
# the C3 nine-shape set. The image ID (sha256:6d7b06b1…) is the true anchor.
EXP_ARCH="arm64"
EXP_OS="linux"
EXP_SOURCE_COMMIT="bae57bd87b03b7c802ca391064996ec27a02d2bb"
EXP_DOCKERFILE_SHA="0758f672f797fe0c2f4812170511821e20384c0d5e0f2e18a8f68df858199a1d"
EXP_BUILD_SCRIPT_SHA="620cee21545dd570c8664c169214def0dadb453b76272ca5b9b72ccefa2a4633"
EXP_CONTRACT_SHA="1b6f00259724f86eb58c047373dabeb9b4df21a1b7bd87995c09e6cfa16bfd77"
EXP_PARENT_IMAGE_ID="sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39"
EXP_REVISION="r9.1-scheduler-liveness-4lane"
EXP_LOCAL_IMAGE_ID="sha256:6d7b06b13f3839da8d9a447e560ba51a894385d52ca052a6f9af24d072d94e82"

echo "== image =="
note "ref: $IMAGE_REF"

if ! docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
  echo "  image not present locally; attempting pull..." >&2
  if ! docker pull "$IMAGE_REF" >/dev/null; then
    bad "cannot inspect or pull $IMAGE_REF"
    exit 1
  fi
fi

inspect() { docker image inspect --format "$1" "$IMAGE_REF" 2>/dev/null; }
label()   { docker image inspect --format "{{index .Config.Labels \"$1\"}}" "$IMAGE_REF" 2>/dev/null; }

check_eq() { # name expected actual
  if [[ "$3" == "$2" ]]; then ok "$1 = $2"; else bad "$1: expected '$2', got '$3'"; fi
}
check_contains() { # name needle haystack
  case "$3" in
    *"$2"*) ok "$1 contains '$2'" ;;
    *)      bad "$1 does not contain '$2'" ;;
  esac
}

echo
echo "== platform =="
check_eq "Architecture" "$EXP_ARCH" "$(inspect '{{.Architecture}}')"
check_eq "Os"           "$EXP_OS"   "$(inspect '{{.Os}}')"

ACTUAL_SIZE="$(inspect '{{.Size}}')"
note "Size: ${ACTUAL_SIZE} bytes (reference build: 20342958503)"

echo
echo "== identity =="
LOCAL_ID="$(inspect '{{.Id}}')"
note "local image ID (config hash): $LOCAL_ID"
if [[ "$LOCAL_ID" == "$EXP_LOCAL_IMAGE_ID" ]]; then
  ok "local image ID matches the reference build"
else
  note "local image ID differs from the reference build ($EXP_LOCAL_IMAGE_ID)."
  note "This is EXPECTED for a registry-pulled copy on some Docker/containerd"
  note "configurations, and is NOT by itself evidence of tampering. The label"
  note "and manifest checks below are the meaningful ones."
fi

REPO_DIGESTS="$(inspect '{{join .RepoDigests \",\"}}')"
if [[ -n "$REPO_DIGESTS" ]]; then
  note "registry digest(s): $REPO_DIGESTS"
else
  note "registry digest: none recorded (image was never pushed or pulled by digest)"
fi
note "the two hashes above are over DIFFERENT objects and are never equal"

echo
echo "== provenance labels =="
check_eq "org.glm52.exp1.source_commit"       "$EXP_SOURCE_COMMIT"     "$(label org.glm52.exp1.source_commit)"
check_eq "org.glm52.exp1.dockerfile_sha256"   "$EXP_DOCKERFILE_SHA"    "$(label org.glm52.exp1.dockerfile_sha256)"
check_eq "org.glm52.exp1.build_script_sha256" "$EXP_BUILD_SCRIPT_SHA"  "$(label org.glm52.exp1.build_script_sha256)"
check_eq "org.glm52.exp1.contract_sha256"     "$EXP_CONTRACT_SHA"      "$(label org.glm52.exp1.contract_sha256)"
check_eq "org.glm52.exp1.parent_image_id"     "$EXP_PARENT_IMAGE_ID"   "$(label org.glm52.exp1.parent_image_id)"
check_eq "org.glm52.exp1.revision"            "$EXP_REVISION"          "$(label org.glm52.exp1.revision)"

echo
echo "== capabilities =="
CAPS="$(label org.glm52.exp1.capabilities)"
check_contains "capabilities" "adaptive-mtp-depth-245-default-on" "$CAPS"
check_contains "capabilities" "adaptive-mtp-full-cudagraphs"      "$CAPS"
check_contains "capabilities" "mtp-window-telemetry"              "$CAPS"
check_contains "capabilities" "mtp-k5-default"                    "$CAPS"

ADAPTIVE="$(label org.glm52.exp1.adaptive_mtp)"
check_contains "adaptive_mtp" "VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5"   "$ADAPTIVE"
check_contains "adaptive_mtp" "3,6,9 / 5,10,15 / 6,12,18"         "$ADAPTIVE"
check_contains "adaptive_mtp" "fails closed"                      "$ADAPTIVE"

echo
echo "== stale-by-construction labels (informational) =="
SERVING="$(label org.glm52.exp1.serving_status)"
if [[ -n "$SERVING" ]]; then
  note "serving_status: ${SERVING%%.*}."
  note "This label was baked at build time and is NO LONGER TRUE: the image has"
  note "since served production traffic and is published. See IMAGE_AUDIT.md."
fi
MEASURED="$(label org.glm52.exp1.measured)"
[[ -n "$MEASURED" ]] && note "measured: ${MEASURED%%.*}. -- performance evidence lives in docs/BENCHMARKS.md, not in the image."

if [[ "$DEEP" == "1" ]]; then
  echo
  echo "== deep probe (CPU-only, --network none, no GPU) =="
  WEIGHTS="$(docker run --rm --network none --entrypoint /bin/sh "$IMAGE_REF" -c \
    'find / -xdev \( -name "*.safetensors" -o -name "*.gguf" -o -name "*.pt" \) 2>/dev/null | grep -vE "compressed_tensors/transform/utils/hadamards.safetensors|/usr/share/vim/" | head -20' 2>/dev/null)"
  if [[ -z "$WEIGHTS" ]]; then
    ok "no model-weight artifacts found in the image"
  else
    bad "unexpected weight-shaped files found:"
    printf '        %s\n' "$WEIGHTS"
  fi

  MODELS_DIR="$(docker run --rm --network none --entrypoint /bin/sh "$IMAGE_REF" -c \
    '[ -e /models ] && echo present || echo absent' 2>/dev/null)"
  check_eq "/models inside the image" "absent" "$MODELS_DIR"

  PROV="$(docker run --rm --network none --entrypoint /bin/sh "$IMAGE_REF" -c \
    'cat /opt/r9/BUILD_PROVENANCE 2>/dev/null | tr "\n" ";"' 2>/dev/null)"
  check_contains "/opt/r9/BUILD_PROVENANCE" "source_commit=$EXP_SOURCE_COMMIT" "$PROV"

  echo
  echo "  To verify the sealed 35-file source manifest, see docs/BUILD.md section 5."
fi

echo
echo "== result =="
printf '  %d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
echo "  OK"
