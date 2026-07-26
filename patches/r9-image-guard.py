#!/usr/bin/env python3
"""R9 image guard — the structural assertions Dockerfile.r9 Guard 2 runs.

    python3 patches/r9-image-guard.py <dist-packages-root>

`grep` proves a string is present. These are the checks that need the parse
tree: an ordering constraint, an absence-inside-a-function constraint, and the
one assertion that deliberately contradicts Dockerfile.r8.

Exits 0 and prints one line per check. Any failure exits 1 with the reason, so
the build fails rather than producing an image whose label claims a capability
its source does not have.

This lives in `patches/` and is COPYed into the image alongside the appliers,
so the guard that qualified the image is auditable inside it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CHECKS: list[str] = []


def _fail(message: str) -> None:
    print(f"R9 image guard FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def _parse(root: Path, name: str) -> ast.Module:
    path = root / name
    if not path.is_file():
        _fail(f"{name} is missing from the image")
    return ast.parse(path.read_text(), str(path))


def _function(tree: ast.Module, name: str, where: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    _fail(f"{where}: function {name}() not found")
    raise AssertionError  # unreachable


def check_v2_full_graph_exemption(root: Path) -> None:
    """THIS INVERTS DOCKERFILE.R8.

    Dockerfile.r8 asserts `"use_v2_model_runner" not in s` for this exact
    function. R9 asserts the opposite. The contradiction is the single most
    important line-level difference between the two images, and it is only
    sound because the coverage guarantee below holds.
    """
    tree = _parse(root, "vllm/config/vllm.py")
    source = ast.unparse(
        _function(tree, "_maybe_override_dynamic_sd_cudagraph_mode", "config/vllm.py")
    )
    if "use_v2_model_runner" not in source:
        _fail(
            "the R9 V2 full-CUDA-graph exemption is absent; this image would "
            "downgrade adaptive MTP to PIECEWISE exactly as R8 does"
        )
    if "uses_acceptance_length_adaptation" not in source:
        _fail("the exemption must be gated on acceptance-length adaptation")
    if "uses_batch_size_dynamic_speculative_decoding" not in source:
        _fail("the batch-size schedule must still be downgraded")
    if "PIECEWISE" not in source:
        _fail("the downgrade must survive for everything the exemption misses")
    CHECKS.append("V2 FULL-CUDA-graph exemption present and narrowly gated")


def check_single_ladder_parser(root: Path) -> None:
    """The community patch parses VLLM_ADAPTIVE_SPEC_DEPTHS twice, with two
    different results. Exactly one parser may exist."""
    for name in (
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/worker/gpu/cudagraph_utils.py",
    ):
        tree = _parse(root, name)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", None))
                == "getenv"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "VLLM_ADAPTIVE_SPEC_DEPTHS"
            ):
                _fail(
                    f"{name}:{node.lineno} re-parses VLLM_ADAPTIVE_SPEC_DEPTHS "
                    "instead of using depth_ladder.parse_adaptive_spec_depth_"
                    "ladder(); the schedulable and captured depth sets could "
                    "then diverge"
                )
        if "from vllm.v1.spec_decode.dynamic.depth_ladder import" not in (
            root / name
        ).read_text():
            _fail(f"{name} does not import the shared ladder parser")
    CHECKS.append("one ladder parser, imported by both call sites")


def check_stats_before_observe(root: Path) -> None:
    """Inherited from R8: `observe_batch` may move the depth on a window
    boundary, so the stats for the completed step must be read first."""
    tree = _parse(root, "vllm/v1/core/sched/scheduler.py")
    body = [
        ast.unparse(node)
        for node in _function(tree, "update_from_output", "scheduler.py").body
    ]
    try:
        stats = next(i for i, x in enumerate(body) if "current_num_spec_tokens" in x)
        observe = next(i for i, x in enumerate(body) if "observe_batch" in x)
    except StopIteration:
        _fail("update_from_output no longer records stats or observes batches")
    if not stats < observe:
        _fail("stats must be recorded before observe_batch moves the depth")
    CHECKS.append("stats recorded before the controller moves the depth")


def check_telemetry_adds_no_device_work(root: Path) -> None:
    """The brief: no forward pass and no GPU synchronization solely for
    telemetry."""
    tree = _parse(root, "vllm/v1/core/sched/scheduler.py")
    scheduler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    source = "".join(
        ast.unparse(node)
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_record_mtp")
    )
    if not source:
        _fail("Scheduler._record_mtp_instrumentation is missing")
    for forbidden in ("torch", "cuda", "synchronize", ".item()", ".cpu()"):
        if forbidden in source:
            _fail(f"telemetry must not touch {forbidden}")
    CHECKS.append("telemetry adds no device work")


def check_controller_policy(root: Path) -> None:
    text = (root / "vllm/v1/spec_decode/dynamic/acceptance_length.py").read_text()
    required = [
        "class AcceptanceLengthController",
        "if self.depth_ladder in ((2, 4), (2, 4, 5)):",
        "head_ratio >= 0.85",
        "tail_gain_23 >= 0.70",
        "tail_gain_23 >= 0.35",
        "position_4_gain >= 0.15",
        "self.num_spec_tokens = self.depth_ladder[0]",
        "self.observation_window // 2",
    ]
    for symbol in required:
        if symbol not in text:
            _fail(f"the 2/4/5 policy is incomplete: missing {symbol!r}")
    if "self.num_spec_tokens = max_num_spec_tokens" in text:
        _fail(
            "the controller still starts at max K; this is R8's ratchet, not "
            "R9's ladder"
        )
    CHECKS.append("2/4/5 depth-ladder policy present, R8 ratchet gone")


def check_multi_depth_graphs(root: Path) -> None:
    text = (root / "vllm/v1/worker/gpu/cudagraph_utils.py").read_text()
    for symbol in (
        "def adaptive_spec_decode_query_lens",
        "def _init_adaptive_spec_candidates",
        "def _assert_adaptive_spec_graph_coverage",
        "def _build_candidate_ranges",
        "no captured FULL CUDA graph",
        "requires FULL CUDA graphs",
    ):
        if symbol not in text:
            _fail(f"multi-depth CUDA-graph support is incomplete: {symbol!r}")

    # The fail-closed assertion must actually be reached from _init_candidates.
    tree = _parse(root, "vllm/v1/worker/gpu/cudagraph_utils.py")
    init = ast.unparse(_function(tree, "_init_candidates", "cudagraph_utils.py"))
    if "_assert_adaptive_spec_graph_coverage" not in init:
        _fail("_init_candidates does not call the fail-closed coverage check")
    CHECKS.append("multi-depth FULL descriptors + fail-closed coverage check")


def check_telemetry_schema(root: Path) -> None:
    text = (root / "vllm/v1/core/sched/scheduler.py").read_text()
    for symbol in (
        "MTP_WINDOW_JSON",
        '"acceptance_ratio_denominator": "drafts_attempted"',
        '"selected_k_run"',
        '"position_conditional_acceptance"',
        '"position_unconditional_gain"',
        '"tail_gain_23"',
        '"position_4_gain"',
        "adaptive_position_eligible",
        "async_tokens_to_discard == 0",
    ):
        if symbol not in text:
            _fail(f"the MTP window schema is incomplete: missing {symbol!r}")
    CHECKS.append("MTP_WINDOW_JSON schema complete")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <dist-packages-root>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    check_controller_policy(root)
    check_single_ladder_parser(root)
    check_multi_depth_graphs(root)
    check_v2_full_graph_exemption(root)
    check_stats_before_observe(root)
    check_telemetry_adds_no_device_work(root)
    check_telemetry_schema(root)

    for line in CHECKS:
        print(f"R9 image guard OK: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
