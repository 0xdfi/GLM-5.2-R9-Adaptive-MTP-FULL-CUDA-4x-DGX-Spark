# SPDX-License-Identifier: Apache-2.0
"""Publication hygiene: this repository must contain no secrets and no private
operational identifiers.

This is a real gate, not a formality. It runs in CI on every push and it is the
last thing between an internal receipt and a public repository.

Three scopes, deliberately different:

  * SECRET_PATTERNS      -- banned everywhere, no exceptions.
  * ADDRESS_PATTERNS     -- private/RFC1918/CGNAT addresses, banned everywhere,
                            no exceptions. There is no honest reason to print
                            one, not even to describe an audit finding.
  * IDENTIFIER_PATTERNS  -- internal hostnames, home paths, usernames,
                            transaction directories. Banned everywhere except
                            the audit documents, which have to describe the
                            findings they cleared.
  * BUILD_TAG_PATTERNS   -- the internal build tag and internal label
                            namespaces. These are DISCLOSED provenance and
                            belong in the docs; they are banned from anything
                            an operator copies and runs.

Run: python3 -m pytest tests -q        (or: python3 tests/test_no_private_data.py)
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Documents whose entire job is to describe what the audit found and cleared.
AUDIT_DOCS = {"IMAGE_AUDIT.md", "SECURITY.md", "VALIDATION.md"}

# This scanner necessarily contains the literal strings it bans (the username
# alternation and the known container-ID list). It is exempt from the
# IDENTIFIER and container-ID scans and from nothing else: it is still scanned
# for credentials and for private addresses, which it has no reason to contain.
SELF = "tests/test_no_private_data.py"

# Things an operator copies and runs. Nothing internal belongs in these.
OPERATIONAL_DIRS = ("runtime", "scripts")

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}


def repo_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        out.append(p)
    return sorted(out)


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


SECRET_PATTERNS = [
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("openai-style key", re.compile(r"\bsk-[A-Za-z0-9]{24,}")),
    ("anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("ssh public key material", re.compile(r"\bssh-(rsa|ed25519)\s+AAAA")),
    ("bearer header", re.compile(r"Authorization:\s*Bearer\s+\S+")),
    ("basic auth in url", re.compile(r"://[^/\s:<]+:[^/\s@<]+@")),
    ("docker registry auth blob", re.compile(r'"auths"\s*:\s*\{')),
]

ADDRESS_PATTERNS = [
    ("rfc1918 10/8 address", re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("rfc1918 192.168/16 address", re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")),
    ("rfc1918 172.16/12 address",
     re.compile(r"\b172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b")),
    ("cgnat/tailscale 100.64/10 address",
     re.compile(r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")),
]

# Hostnames and usernames get NO exemption anywhere. The audit documents describe
# these findings using redacted placeholders (<BUILD_HOST>, <user>,
# <OWNER_FIRST_NAME>), so there is no case in which a literal is needed.
IDENTITY_PATTERNS = [
    ("internal spark host alias", re.compile(r"\bspark-(nord|sud|ost|west)\b")),
    ("local username", re.compile(r"\b(daffi|donirwin)\b")),
]

# Path shapes. Exempt in the audit documents only, because describing a finding
# like "/home/<user>/exp1-build/..." unavoidably contains the path fragment.
PATH_PATTERNS = [
    ("macos home path", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    ("linux home path", re.compile(r"/home/[A-Za-z0-9._-]+")),
    ("internal transaction dir", re.compile(r"r9-cutover-\d{8}T\d{6}Z")),
    ("internal evidence/build path", re.compile(r"/exp1-(evidence|build)\b")),
]

IDENTIFIER_PATTERNS = IDENTITY_PATTERNS + PATH_PATTERNS

BUILD_TAG_PATTERNS = [
    ("internal build tag", re.compile(r"glm52-exp1-sm121a-368-canary")),
    ("internal label namespace", re.compile(r"\borg\.hermes\.")),
]

# Docker container IDs overlap with sha256 prefixes, which are legitimate here,
# so this checks for the specific known-internal IDs rather than the shape.
KNOWN_CONTAINER_IDS = [
    "f92880da4260", "85d55d09dfbf", "d206fe315281", "1c4c99e8cd1a",
    "22b9f9a3851b",
]


def _scan(testcase, patterns, files) -> None:
    for path in files:
        rel = path.relative_to(ROOT)
        text = read(path)
        if not text:
            continue
        for label, pattern in patterns:
            with testcase.subTest(file=str(rel), pattern=label):
                match = pattern.search(text)
                testcase.assertIsNone(
                    match,
                    f"{rel} contains a {label}: {match.group(0)[:40] if match else ''}",
                )


class TestNoSecrets(unittest.TestCase):
    def test_no_credential_shaped_strings_anywhere(self) -> None:
        _scan(self, SECRET_PATTERNS, repo_files())


class TestNoPrivateAddresses(unittest.TestCase):
    def test_no_private_ip_addresses_anywhere(self) -> None:
        """No exemption. Not even the audit documents may print one."""
        _scan(self, ADDRESS_PATTERNS, repo_files())


class TestNoPrivateIdentifiers(unittest.TestCase):
    def test_no_internal_hostnames_or_usernames_anywhere(self) -> None:
        """No exemption: the audit documents use redacted placeholders instead."""
        files = [p for p in repo_files() if str(p.relative_to(ROOT)) != SELF]
        _scan(self, IDENTITY_PATTERNS, files)

    def test_no_private_paths_outside_the_audit_docs(self) -> None:
        files = [
            p for p in repo_files()
            if p.name not in AUDIT_DOCS and str(p.relative_to(ROOT)) != SELF
        ]
        _scan(self, PATH_PATTERNS, files)

    def test_no_known_internal_container_ids_anywhere(self) -> None:
        for path in repo_files():
            rel = path.relative_to(ROOT)
            if str(rel) == SELF:
                continue
            text = read(path)
            if not text:
                continue
            for cid in KNOWN_CONTAINER_IDS:
                with self.subTest(file=str(rel), cid=cid):
                    self.assertNotIn(cid, text, f"{rel} leaks container ID {cid}")


class TestOperationalFilesAreClean(unittest.TestCase):
    def test_no_internal_tags_in_runtime_or_scripts(self) -> None:
        files = [
            p for p in repo_files()
            if p.relative_to(ROOT).parts and p.relative_to(ROOT).parts[0] in OPERATIONAL_DIRS
        ]
        self.assertTrue(files, "expected runtime/ and scripts/ to contain files")
        _scan(self, BUILD_TAG_PATTERNS + IDENTIFIER_PATTERNS, files)


class TestRuntimeTemplatesArePlaceholders(unittest.TestCase):
    def test_env_example_has_no_real_values(self) -> None:
        text = read(ROOT / "runtime" / "r9.env.example")
        for var in ("HEAD_NODE_IP", "NODE_IP", "API_BIND_ADDR", "API_PORT",
                    "RAY_PORT", "MODEL_DIR", "JIT_CACHE_DIR",
                    "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_HCA",
                    "CONTAINER_NAME"):
            with self.subTest(var=var):
                match = re.search(rf'^{var}="([^"]*)"$', text, re.MULTILINE)
                self.assertIsNotNone(match, f"{var} not found in the env example")
                assert match is not None
                value = match.group(1)
                self.assertTrue(
                    value.startswith("<") and value.endswith(">"),
                    f"{var} must be a bare <PLACEHOLDER>, got {value!r}",
                )

    def test_image_ref_is_digest_placeholder(self) -> None:
        text = read(ROOT / "runtime" / "r9.env.example")
        self.assertIn("@sha256:<GHCR_DIGEST>", text)

    def test_no_real_env_file_is_committed(self) -> None:
        self.assertFalse((ROOT / "runtime" / "r9.env").exists(),
                         "runtime/r9.env must never be committed")


class TestGitTrackedFilesOnly(unittest.TestCase):
    """If this is a git repo, the scan above must have covered the tracked set."""

    def test_tracked_files_are_a_subset_of_scanned_files(self) -> None:
        try:
            out = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files"],
                capture_output=True, text=True, check=True, timeout=30,
            ).stdout
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            self.skipTest("git not available or not a repository")
            return
        tracked = {line for line in out.splitlines() if line}
        if not tracked:
            self.skipTest("no files tracked yet")
            return
        scanned = {str(p.relative_to(ROOT)) for p in repo_files()}
        missing = tracked - scanned
        self.assertFalse(missing, f"tracked but not scanned: {sorted(missing)[:10]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
