#!/usr/bin/env bash
set -euo pipefail

REPO="0xdfi/GLM-5.2-R9-Adaptive-MTP-FULL-CUDA-4x-DGX-Spark"
TAG="r9-adaptive-full-bae57bd"
BASE="https://github.com/${REPO}/releases/download/${TAG}"
OUT_DIR="${1:-image-download}"
ARCHIVE="glm52-r9-adaptive-full-bae57bd.oci.tar.zst"
PARTS=(
  "${ARCHIVE}.part-00.bin"
  "${ARCHIVE}.part-01.bin"
  "${ARCHIVE}.part-02.bin"
  "${ARCHIVE}.part-03.bin"
  "${ARCHIVE}.part-04.bin"
)

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 1; }
mkdir -p "$OUT_DIR"

fetch() {
  local name="$1"
  if [[ -f "$OUT_DIR/$name" ]]; then
    echo "Already present: $name"
    return
  fi
  echo "Downloading $name"
  curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - --output "$OUT_DIR/$name" "$BASE/$name"
}

fetch SHA256SUMS
for part in "${PARTS[@]}"; do fetch "$part"; done

(
  cd "$OUT_DIR"
  for part in "${PARTS[@]}"; do
    grep "  ${part}$" SHA256SUMS
  done | sha256sum --check --strict -

  rm -f "${ARCHIVE}.tmp"
  for part in "${PARTS[@]}"; do
    cat "$part" >> "${ARCHIVE}.tmp"
  done
  mv "${ARCHIVE}.tmp" "$ARCHIVE"
  grep "  ${ARCHIVE}$" SHA256SUMS | sha256sum --check --strict -
)

printf '\nVerified archive: %s/%s\n' "$OUT_DIR" "$ARCHIVE"
printf 'Load it with:\n  zstd -d -c %q | docker load\n' "$OUT_DIR/$ARCHIVE"
