# SPDX-License-Identifier: Apache-2.0
"""Behavioural tests for runtime/start-node.sh.

The launcher's whole value is that it refuses. These tests prove the refusals
actually fire, with the documented exit codes, and that a fully valid
configuration renders a launch line carrying the qualified invariants.

Nothing here starts a container: every case runs with RENDER_ONLY=1, and the
refusal cases exit before any docker invocation is reached.

Run: python3 -m pytest tests -q        (or: python3 tests/test_runtime_templates.py)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "runtime" / "start-node.sh"
EXAMPLE = ROOT / "runtime" / "r9.env.example"

# A complete, valid configuration. Addresses are documentation-range values
# (RFC 5737 TEST-NET-1) so no private address ever appears in this repository.
VALID = {
    "IMAGE_REF": "ghcr.io/0xdfi/glm-5.2-r9-adaptive-mtp-full-cuda-4x-dgx-spark@sha256:" + "0" * 64,
    "CONTAINER_NAME": "glm52-r9",
    "HEAD_NODE_IP": "203.0.113.10",
    "NODE_IP": "203.0.113.10",
    "RAY_PORT": "6379",
    "API_BIND_ADDR": "203.0.113.10",
    "API_PORT": "8210",
    "NCCL_SOCKET_IFNAME": "eth0",
    "GLOO_SOCKET_IFNAME": "eth0",
    "NCCL_IB_HCA": "mlx5_0",
    "MODEL_DIR": "",          # filled in per-test with a real temp dir
    "JIT_CACHE_DIR": "",      # filled in per-test with a real temp dir
    "R91_PROFILE": "balanced",
    "MAX_NUM_SEQS": "4",
    "MTP_K": "5",
    "VLLM_ADAPTIVE_SPEC_DEPTHS": "2,4,5",
    "ADAPTIVE_SPEC_WINDOW": "32",
    "CUDAGRAPH_SIZES": "6,12,18,24",
    "VLLM_MTP_INSTRUMENT": "1",
    "VLLM_MTP_INSTRUMENT_WINDOW": "32",
    "MAX_NUM_BATCHED_TOKENS": "1024",
    "LONG_PREFILL_TOKEN_THRESHOLD": "1024",
    "DECODE_PREFILL_TOKEN_BUDGET": "1024",
    "IDLE_PREFILL_TOKEN_BUDGET": "1024",
    "MAX_LONG_PREFILLS_PER_STEP": "1",
    "TENSOR_PARALLEL_SIZE": "4",
    "PIPELINE_PARALLEL_SIZE": "1",
    "DCP_COMM_BACKEND": "a2a",
    "SERVED_MODEL_NAME": "glm-5.2",
}

HAVE_BASH = shutil.which("bash") is not None
HAVE_DOCKER = shutil.which("docker") is not None


class LauncherCase(unittest.TestCase):
    def setUp(self) -> None:
        if not HAVE_BASH:
            self.skipTest("bash not available")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        (self.tmp / "models").mkdir()
        (self.tmp / "cache").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_env(self, **overrides: str) -> Path:
        cfg = dict(VALID)
        cfg["MODEL_DIR"] = str(self.tmp / "models")
        cfg["JIT_CACHE_DIR"] = str(self.tmp / "cache")
        cfg.update(overrides)
        path = self.tmp / "r9.env"
        path.write_text(
            "\n".join(f'{k}="{v}"' for k, v in cfg.items()) + "\n", encoding="utf-8"
        )
        return path

    def run_launcher(self, env_path: Path, role: str = "head", **extra_env: str):
        env = dict(os.environ)
        env.update({"NODE_ROLE": role, "RENDER_ONLY": "1"})
        env.update(extra_env)
        return subprocess.run(
            ["bash", str(SCRIPT), str(env_path)],
            capture_output=True, text=True, env=env, timeout=60,
        )


class TestSyntax(LauncherCase):
    def test_script_parses(self) -> None:
        proc = subprocess.run(["bash", "-n", str(SCRIPT)],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class TestFailClosed(LauncherCase):
    """Each documented exit code fires for its documented condition."""

    def assert_exit(self, code: int, **overrides: str) -> None:
        proc = self.run_launcher(self.write_env(**overrides))
        self.assertEqual(
            proc.returncode, code,
            f"expected exit {code}, got {proc.returncode}\n"
            f"stdout: {proc.stdout[:400]}\nstderr: {proc.stderr[:400]}",
        )
        self.assertIn("refusing to run", proc.stderr.lower())

    def test_11_placeholder_left_in_place(self) -> None:
        self.assert_exit(11, HEAD_NODE_IP="<HEAD_NODE_IP>")

    def test_11_empty_required_var(self) -> None:
        self.assert_exit(11, API_BIND_ADDR="")

    def test_11_bad_node_role(self) -> None:
        proc = self.run_launcher(self.write_env(), role="controller")
        self.assertEqual(proc.returncode, 11)

    def test_2_non_integer_mtp_k(self) -> None:
        self.assert_exit(2, MTP_K="five")

    def test_4_leading_zero_window_is_rejected(self) -> None:
        """'08' is valid bash and invalid JSON. It must not reach the server."""
        self.assert_exit(4, ADAPTIVE_SPEC_WINDOW="08")

    def test_4_non_integer_window(self) -> None:
        self.assert_exit(4, ADAPTIVE_SPEC_WINDOW="thirty-two")

    def test_5_wrong_ladder(self) -> None:
        self.assert_exit(5, VLLM_ADAPTIVE_SPEC_DEPTHS="2,3,4,5")

    def test_5_ladder_missing_a_rung(self) -> None:
        self.assert_exit(5, VLLM_ADAPTIVE_SPEC_DEPTHS="2,5")

    def test_6_ladder_top_rung_does_not_match_mtp_k(self) -> None:
        self.assert_exit(6, MTP_K="4")

    def test_7_enforce_eager(self) -> None:
        proc = self.run_launcher(self.write_env(), ENFORCE_EAGER="1")
        self.assertEqual(proc.returncode, 7, proc.stderr[:400])

    def test_7_capture_sizes_cannot_cover_the_shapes(self) -> None:
        self.assert_exit(7, CUDAGRAPH_SIZES="6,12")

    def test_8_telemetry_disabled(self) -> None:
        self.assert_exit(8, VLLM_MTP_INSTRUMENT="0")

    def test_8_malformed_instrument_window(self) -> None:
        self.assert_exit(8, VLLM_MTP_INSTRUMENT_WINDOW="0x20")

    def test_10_max_num_seqs_other_than_four(self) -> None:
        self.assert_exit(10, MAX_NUM_SEQS="2")

    def test_10_max_num_seqs_three_is_rejected(self) -> None:
        """R9 ran at C3; R9.1 requires C4. Three must now be refused."""
        self.assert_exit(10, MAX_NUM_SEQS="3")

    def test_10_max_num_seqs_zero_padded(self) -> None:
        self.assert_exit(10, MAX_NUM_SEQS="04")

    def test_12_unknown_profile(self) -> None:
        proc = self.run_launcher(self.write_env(R91_PROFILE="1m"))
        self.assertEqual(proc.returncode, 12, proc.stderr[:400])
        self.assertIn("matched triple", proc.stderr)

    def test_13_missing_model_dir(self) -> None:
        proc = self.run_launcher(self.write_env(MODEL_DIR=str(self.tmp / "nope")))
        self.assertEqual(proc.returncode, 13, proc.stderr[:400])


class TestValidRenderBalanced(LauncherCase):
    """A valid balanced (DCP2) configuration renders, and carries the invariants."""

    def setUp(self) -> None:
        super().setUp()
        if not HAVE_DOCKER:
            self.skipTest("docker not on PATH; the launcher requires it for host preflight")
        self.proc = self.run_launcher(self.write_env(R91_PROFILE="balanced"))
        self.assertEqual(self.proc.returncode, 0,
                         f"stderr: {self.proc.stderr[:600]}")
        self.out = self.proc.stdout

    def test_renders_the_qualified_profile(self) -> None:
        self.assertIn("--max-model-len 520000", self.out.replace("\\", ""))
        self.assertIn("--kv-cache-memory-bytes 8410000000", self.out.replace("\\", ""))

    def test_renders_dcp2_with_comm_backend(self) -> None:
        """Balanced is DCP2: the comm-backend and interleave flags must be present."""
        flat = self.out.replace("\\", "")
        self.assertIn("--decode-context-parallel-size 2", flat)
        self.assertIn("--dcp-comm-backend a2a", flat)
        self.assertIn("--dcp-kv-cache-interleave-size 1", flat)

    def test_renders_the_adaptive_speculative_config(self) -> None:
        flat = self.out.replace("\\", "")
        self.assertIn("adaptive_speculative_tokens_window", flat)
        self.assertIn("num_speculative_tokens", flat)
        self.assertIn("VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5", flat)

    def test_renders_full_graph_capture_sizes(self) -> None:
        flat = self.out.replace("\\", "")
        self.assertIn("cudagraph_capture_sizes", flat)
        self.assertIn("6,12,18,24", flat)

    def test_does_not_render_enforce_eager(self) -> None:
        self.assertNotIn("--enforce-eager", self.out)

    def test_renders_nvfp4_kv_cache_and_compressed_tensors_weights(self) -> None:
        flat = self.out.replace("\\", "")
        self.assertIn("--kv-cache-dtype nvfp4_ds_mla", flat)
        self.assertIn("--quantization compressed-tensors", flat)

    def test_mounts_the_model_read_only(self) -> None:
        self.assertIn(":/models:ro", self.out.replace("\\", ""))

    def test_worker_does_not_render_an_api_server(self) -> None:
        proc = self.run_launcher(self.write_env(R91_PROFILE="balanced"), role="worker")
        self.assertEqual(proc.returncode, 0, proc.stderr[:400])
        self.assertNotIn("api_server", proc.stdout)
        self.assertIn("ray start --address", proc.stdout.replace("\\", ""))

    def test_speculative_config_is_valid_json(self) -> None:
        """The rendered --speculative-config must parse; a launcher that emits
        invalid JSON fails at model load, an hour into a maintenance window."""
        flat = self.out.replace("\\", "")
        start = flat.index('{"model":"/models","method":"mtp"')
        depth = 0
        for i, ch in enumerate(flat[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = flat[start:i + 1]
                    break
        else:  # pragma: no cover
            self.fail("could not delimit the speculative-config JSON")
        cfg = json.loads(blob)
        self.assertEqual(cfg["num_speculative_tokens"], 5)
        self.assertEqual(cfg["adaptive_speculative_tokens_window"], 32)
        self.assertEqual(cfg["method"], "mtp")


class TestValidRenderFastDcp1Guard(LauncherCase):
    """The core R9.1 fix: the fast (DCP1) profile must OMIT the DCP comm flags
    that crash the engine on boot. This is the multi-process bug that R9.1 fixes."""

    def setUp(self) -> None:
        super().setUp()
        if not HAVE_DOCKER:
            self.skipTest("docker not on PATH; the launcher requires it for host preflight")

    def test_fast_renders_dcp1_without_comm_flags(self) -> None:
        proc = self.run_launcher(self.write_env(R91_PROFILE="fast"))
        self.assertEqual(proc.returncode, 0, proc.stderr[:600])
        flat = proc.stdout.replace("\\", "")
        self.assertIn("--decode-context-parallel-size 1", flat)
        self.assertIn("--max-model-len 319000", flat)
        self.assertIn("--kv-cache-memory-bytes 10233000000", flat)
        # The whole point of R9.1: DCP1 must NOT carry the comm flags.
        self.assertNotIn("--dcp-comm-backend", flat)
        self.assertNotIn("--dcp-kv-cache-interleave-size", flat)

    def test_fast_and_balanced_render_identical_capture_set(self) -> None:
        """Both profiles are C4 and must render the same twelve-shape set."""
        for profile in ("fast", "balanced"):
            with self.subTest(profile=profile):
                proc = self.run_launcher(self.write_env(R91_PROFILE=profile))
                self.assertEqual(proc.returncode, 0, proc.stderr[:400])
                flat = proc.stdout.replace("\\", "")
                self.assertIn("6,12,18,24", flat)
                self.assertIn("--max-num-seqs 4", flat)


class TestExampleEnvCoversEveryRequiredVar(LauncherCase):
    def test_example_defines_every_required_variable(self) -> None:
        import re
        example = EXAMPLE.read_text()
        for key in VALID:
            with self.subTest(var=key):
                self.assertIsNotNone(
                    re.search(rf"^{key}=", example, re.MULTILINE),
                    f"{key} missing from r9.env.example",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
