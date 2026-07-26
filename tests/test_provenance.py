# SPDX-License-Identifier: Apache-2.0
"""Provenance tests for the published package.

These prove that the provenance artifacts shipped here are internally consistent
and match the values recorded in the image labels, so a reader can check the
package against the image without trusting the prose.

Run: python3 -m pytest tests -q        (or: python3 tests/test_provenance.py)
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "provenance" / "r9-postimage.sha256"
PATCH_13 = ROOT / "patches" / "13_r9_adaptive_full_cuda.py"
IMAGE_GUARD = ROOT / "patches" / "r9-image-guard.py"
RELEASE = ROOT / "release-manifest.json"

# Pinned sha256 of the files copied into this package. If a copy is ever
# silently edited, these fail.
PINNED_FILE_HASHES = {
    "patches/13_r9_adaptive_full_cuda.py":
        "fb8142e78508aeeed5112dbc96447c822df95018728c6cd77bac15531875a134",
    "patches/r9-image-guard.py":
        "1e9a9417925412f9828c63830fe0c064b2806cc183c0c0d6750e5bde1f6ff69c",
    "provenance/r9-postimage.sha256":
        "6c1f64b563b99aebec920f5b0126b56aa3a5c97938c03c4b30c489eefc520536",
}

# The five files R9 moves, and the sha256 each carries in the image's own
# org.glm52.exp1.<name>_sha256 labels.
R9_DELTA_FILES = {
    "vllm/v1/spec_decode/dynamic/acceptance_length.py":
        "4b4e17521cb9bb22e2b8062e57f6bee43c170abc54d3b126bef409706cf3fbd9",
    "vllm/v1/spec_decode/dynamic/depth_ladder.py":
        "674487015cd59f5528c98517276f59a5c78a79b510dc93e2ab06989b6a338e9f",
    "vllm/v1/worker/gpu/cudagraph_utils.py":
        "7ce3dfb47a050b3cee7285f3ce796973b4fe285968bc78f35c86d2794044e454",
    "vllm/config/vllm.py":
        "9d90933ee15f382e87d60e2fa46919535f6d66c095c39a3cebef0ef1e53b9fd8",
    "vllm/v1/core/sched/scheduler.py":
        "c05eb6359230c7883b5a1a859db48c19a2d23a0c3a7bba9259ca6ae7d92d2723",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if not SHA256_RE.match(digest) or not name:
            raise AssertionError(f"{path.name}:{lineno}: malformed entry: {raw!r}")
        if name in entries:
            raise AssertionError(f"{path.name}:{lineno}: duplicate path {name}")
        entries[name] = digest
    return entries


class TestCopiedFileHashes(unittest.TestCase):
    """The provenance files copied into this package are byte-exact."""

    def test_pinned_hashes_match(self) -> None:
        for rel, expected in PINNED_FILE_HASHES.items():
            with self.subTest(file=rel):
                path = ROOT / rel
                self.assertTrue(path.is_file(), f"missing {rel}")
                self.assertEqual(sha256_of(path), expected, f"{rel} content changed")


class TestSealedManifest(unittest.TestCase):
    """The 35-file sealed post-image manifest is well formed and complete."""

    def setUp(self) -> None:
        self.entries = parse_manifest(MANIFEST)

    def test_entry_count(self) -> None:
        self.assertEqual(len(self.entries), 35)

    def test_every_digest_is_lowercase_sha256(self) -> None:
        for name, digest in self.entries.items():
            with self.subTest(file=name):
                self.assertRegex(digest, SHA256_RE)

    def test_paths_are_relative_and_carry_no_private_data(self) -> None:
        for name in self.entries:
            with self.subTest(file=name):
                self.assertFalse(name.startswith("/"), "manifest paths must be relative")
                self.assertNotIn("..", name)
                self.assertNotIn("/home/", name)
                self.assertNotIn("/Users/", name)

    def test_the_five_r9_delta_files_match_the_image_labels(self) -> None:
        """Each R9 file's manifest digest equals the value in its image label."""
        for name, expected in R9_DELTA_FILES.items():
            with self.subTest(file=name):
                self.assertIn(name, self.entries, f"{name} absent from the sealed manifest")
                self.assertEqual(self.entries[name], expected)

    def test_the_dynamic_spec_decode_package_is_exactly_four_files(self) -> None:
        """depth_ladder.py is the file R9 adds; the other three are inherited."""
        dynamic = sorted(
            n for n in self.entries if n.startswith("vllm/v1/spec_decode/dynamic/")
        )
        self.assertEqual(
            dynamic,
            [
                "vllm/v1/spec_decode/dynamic/__init__.py",
                "vllm/v1/spec_decode/dynamic/acceptance_length.py",
                "vllm/v1/spec_decode/dynamic/depth_ladder.py",
                "vllm/v1/spec_decode/dynamic/utils.py",
            ],
        )

    def test_depth_ladder_is_the_only_added_file_in_that_package(self) -> None:
        delta = json.loads(RELEASE.read_text())["source_delta"]["files"]
        added = sorted(
            f["path"] for f in delta
            if f["change"] == "added" and f["path"].startswith("vllm/v1/spec_decode/dynamic/")
        )
        self.assertEqual(added, ["vllm/v1/spec_decode/dynamic/depth_ladder.py"])


class TestPatchApplier(unittest.TestCase):
    """The included applier is the real thing: parseable, attributed, and it
    declares exactly the five files this release claims to move."""

    def setUp(self) -> None:
        self.src = PATCH_13.read_text()

    def test_it_is_valid_python(self) -> None:
        import ast
        ast.parse(self.src, filename=str(PATCH_13))

    def test_it_preserves_upstream_attribution(self) -> None:
        self.assertIn("CosmicRaisins/glm-5.2-gb10", self.src)
        self.assertIn("600848707ce93fe42fedbc9dd4429116696e425d", self.src)

    def test_it_names_all_five_delta_files(self) -> None:
        for name in R9_DELTA_FILES:
            with self.subTest(file=name):
                self.assertIn(name, self.src)

    def test_it_documents_the_two_reconciliations(self) -> None:
        """The duplicate ladder parser and the merged candidate ranges are the
        two community-patch behaviours R9 deliberately does not carry."""
        self.assertIn("VLLM_ADAPTIVE_SPEC_DEPTHS", self.src)
        self.assertIn("depth_ladder.py", self.src)
        lowered = self.src.lower()
        self.assertIn("candidate range", lowered)


class TestImageGuard(unittest.TestCase):
    """The structural guard shipped here is the one the build ran."""

    def test_it_is_valid_python(self) -> None:
        import ast
        ast.parse(IMAGE_GUARD.read_text(), filename=str(IMAGE_GUARD))

    def test_it_asserts_the_v2_runner_exemption(self) -> None:
        src = IMAGE_GUARD.read_text()
        self.assertIn("use_v2_model_runner", src)
        self.assertIn("_maybe_override_dynamic_sd_cudagraph_mode", src)


class TestReleaseManifest(unittest.TestCase):
    """release-manifest.json agrees with everything else in the package."""

    def setUp(self) -> None:
        self.data = json.loads(RELEASE.read_text())

    def test_immutable_image_facts(self) -> None:
        img = self.data["image"]
        self.assertEqual(img["architecture"], "arm64")
        self.assertEqual(img["os"], "linux")
        self.assertEqual(img["size_bytes"], 20342958503)
        self.assertEqual(
            img["local_image_id"],
            "sha256:50261a39caf7109bcf49e33fa29b1ba9f7dd630f7ac9eebef72d7994aa98ea39",
        )
        self.assertFalse(img["model_weights_included"])

    def test_unknowns_are_null_not_guessed(self) -> None:
        img = self.data["image"]
        release = self.data["release"]
        self.assertIsNone(img["ghcr_digest"])
        self.assertIsNone(release["github_release_url"])
        self.assertIsNone(release["published_commit"])

    def test_delta_file_hashes_match_the_sealed_manifest(self) -> None:
        entries = parse_manifest(MANIFEST)
        for item in self.data["source_delta"]["files"]:
            with self.subTest(file=item["path"]):
                self.assertEqual(entries[item["path"]], item["sha256"])

    def test_measured_results_carry_denominators(self) -> None:
        for result in self.data["measured_results"]:
            with self.subTest(result=result["name"]):
                self.assertIn("caveat", result)
                self.assertTrue(result["caveat"].strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
