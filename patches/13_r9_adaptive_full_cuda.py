#!/usr/bin/env python3
"""R9 patch 13 — adaptive MTP 2/4/5 with FULL CUDA graphs retained.

Applies on top of the R8 post-image. The full chain is 03 -> 06 -> 12 -> 13.

    python3 patches/13_r9_adaptive_full_cuda.py <dist-packages-root>

What this does
--------------
R8 (patch 12) shipped the acceptance-length machinery but deliberately left two
things out, and R9 supplies exactly those two:

  1. The POLICY. R8 ships the generic v18 ratchet -- start at max K, move to
     `floor(mean accepted + 1.5)`, recover +1 per window. R9 replaces it with
     the production-tuned discrete `2 -> 4 -> 5` ladder from
     CosmicRaisins/glm-5.2-gb10 600848707ce93fe42fedbc9dd4429116696e425d
     (`adaptive-mtp/overlay/.../acceptance_length.py`), which starts at a safe
     k=2 floor, probes k4 only on an 0.85 head ratio, and judges k4/k5 by the
     unconditional marginal tokens the extra positions actually earn.

  2. FULL CUDA GRAPHS. R8's header says it plainly: enabling adaptive
     downgrades cudagraph_mode from FULL to PIECEWISE, because full-graph
     coverage for every adaptive K across the MAX_NUM_SEQS=3 shapes was not
     proven. R9 proves it by construction --
     `CudaGraphManager._init_candidates()` now captures a FULL decode
     descriptor for every (ladder depth, request count) pair -- and only then
     removes the downgrade, narrowly, for acceptance-length adaptation on the
     V2 runner.

Files this applier touches
--------------------------
  vllm/v1/spec_decode/dynamic/acceptance_length.py  REPLACED: the 2/4/5 policy
  vllm/v1/spec_decode/dynamic/depth_ladder.py       NEW: one ladder parser
  vllm/v1/worker/gpu/cudagraph_utils.py             multi-depth descriptors
  vllm/config/vllm.py                               the V2 FULL-graph exemption
  vllm/v1/core/sched/scheduler.py                   ladder + per-position counts
                                                    + MTP_WINDOW_JSON telemetry

Pre-images
----------
`cudagraph_utils.py` is untouched by R6, R7 and R8, so its anchors are the exact
e232d262369b8c918cf478a7a96a0fcf8127cf65 blob. `scheduler.py` and
`config/vllm.py` anchor on the R8 POST-image, i.e. the output of patch 03 ->
patch 06 -> patch 12. Run those first.

Deliberately NOT ported
-----------------------
* The community patch's second copy of the `VLLM_ADAPTIVE_SPEC_DEPTHS` parser.
  It disagrees with the scheduler's copy about whether `num_speculative_tokens`
  is always a rung, which is precisely the divergence that would let the
  scheduler pick an uncaptured depth. R9 ships one parser, `depth_ladder.py`,
  and both call sites import it.
* The community patch's re-partitioning of the CUDA-graph candidate ranges. It
  merges the per-depth descriptors into the base token-count buckets, which
  under the FULL_AND_PIECEWISE default silently removes mixed-batch coverage
  (a 4-token prefill lands in a bucket holding only uniform-5 decode
  descriptors and drops to eager). R9 layers instead: base ranges are built
  unchanged and the per-depth descriptors are prepended to them.
* DSpark, DFlash block capacity, CausalCascade, async draft output, variable
  width, `draft_gumbel_pos`, EPLB, local-argmax reduction -- still excluded,
  exactly as in R8.

Every edit is anchored on an exact string from its pre-image. A base bump that
moves an anchor fails loudly rather than silently skipping a hunk. Re-running on
an already-patched tree is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

CONTROLLER_PATH = "vllm/v1/spec_decode/dynamic/acceptance_length.py"
DEPTH_LADDER_PATH = "vllm/v1/spec_decode/dynamic/depth_ladder.py"

CONTROLLER_SOURCE = '''\
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# R9: the production-tuned 2 -> 4 -> 5 acceptance-length controller, taken
# verbatim from CosmicRaisins/glm-5.2-gb10 commit
# 600848707ce93fe42fedbc9dd4429116696e425d, file
# `adaptive-mtp/overlay/vllm/v1/spec_decode/dynamic/acceptance_length.py`.
#
# That overlay is itself a forward-port of the acceptance-length controller from
# local-inference-lab/vllm (Luke Alonso, feature commit
# d179dc83755ca7365a6c1b1294c74d7908106bc7) -- whose generic
# `floor(mean accepted + 1.5)` ratchet R8 shipped -- with the community's
# discrete 2/4/5 depth-ladder policy layered on top. R8's controller is
# REPLACED, not extended: its ratchet survives only as the `else` branch that
# runs when the ladder is not (2,4) or (2,4,5).
#
# The ONLY edit against the pinned overlay is the `__future__` import below,
# which lets this repository's Python 3.9 checks exec the file directly while
# the container keeps running it unchanged under Python 3.12. Everything from
# `from dataclasses import dataclass` onward is byte-identical to the overlay;
# tests/test_r9_controller_provenance.py asserts exactly that.
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence


@dataclass(frozen=True)
class AcceptanceLengthUpdate:
    previous_num_spec_tokens: int
    num_spec_tokens: int
    mean_num_accepted_tokens: float
    mean_num_draft_tokens: float
    raw_target_num_spec_tokens: int = 0
    acceptance_ratchet: bool = False
    decision_reason: str = ""
    position_conditional_acceptance: tuple[float, ...] = ()
    observation_window: int = 0
    tail_gain_23: float = 0.0
    position_4_gain: float = 0.0


class AcceptanceLengthController:
    """Adjust speculative depth from observed acceptance, using a depth ladder."""

    def __init__(
        self,
        max_num_spec_tokens: int,
        observation_window: int,
        depth_ladder: Sequence[int] | None = None,
    ) -> None:
        if max_num_spec_tokens <= 0:
            raise ValueError("max_num_spec_tokens must be greater than zero.")
        if observation_window <= 0:
            raise ValueError("observation_window must be greater than zero.")

        self.max_num_spec_tokens = max_num_spec_tokens
        self.observation_window = observation_window
        ladder = sorted(
            {
                int(depth)
                for depth in (depth_ladder or [])
                if 1 <= int(depth) <= max_num_spec_tokens
            }
        )
        if not ladder:
            ladder = list(range(1, max_num_spec_tokens + 1))
        if ladder[-1] != max_num_spec_tokens:
            ladder.append(max_num_spec_tokens)
        self.depth_ladder = tuple(ladder)
        self.num_spec_tokens = self.depth_ladder[0]

        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0
        self._position_eligible = [0] * max_num_spec_tokens
        self._position_accepted = [0] * max_num_spec_tokens

    def observe_batch(
        self,
        *,
        num_drafts: int,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        position_eligible: Sequence[int] | None = None,
        position_accepted: Sequence[int] | None = None,
    ) -> AcceptanceLengthUpdate | None:
        """Observe one scheduler step and occasionally update the depth."""
        if num_drafts < 0 or num_draft_tokens < 0 or num_accepted_tokens < 0:
            raise ValueError("Speculative decoding counts must be non-negative.")
        if num_accepted_tokens > num_draft_tokens:
            raise ValueError(
                "num_accepted_tokens must not exceed num_draft_tokens."
            )
        if num_drafts == 0:
            if num_draft_tokens or num_accepted_tokens:
                raise ValueError("Token counts require at least one draft.")
            return None

        self._num_observation_steps += 1
        self._num_drafts += num_drafts
        self._num_draft_tokens += num_draft_tokens
        self._num_accepted_tokens += num_accepted_tokens
        if position_eligible is not None or position_accepted is not None:
            if position_eligible is None or position_accepted is None:
                raise ValueError(
                    "Position eligibility and acceptance must be provided together."
                )
            if len(position_eligible) != len(position_accepted):
                raise ValueError(
                    "Position eligibility and acceptance lengths must match."
                )
            if len(position_eligible) > self.max_num_spec_tokens:
                raise ValueError("Position vectors exceed max_num_spec_tokens.")
            for i, (eligible, accepted) in enumerate(
                zip(position_eligible, position_accepted)
            ):
                if eligible < 0 or accepted < 0 or accepted > eligible:
                    raise ValueError("Invalid per-position acceptance counts.")
                self._position_eligible[i] += eligible
                self._position_accepted[i] += accepted

        # Stay conservative at the k=2 baseline, but resolve exploratory
        # k=4/k=5 probes quickly. The shorter probe window bounds the cost of
        # inheriting a high depth when workload character changes.
        effective_observation_window = (
            self.observation_window
            if self.num_spec_tokens == self.depth_ladder[0]
            else max(1, self.observation_window // 2)
        )
        if self._num_observation_steps < effective_observation_window:
            return None

        mean_num_accepted_tokens = self._num_accepted_tokens / self._num_drafts
        mean_num_draft_tokens = self._num_draft_tokens / self._num_drafts
        position_rates = tuple(
            accepted / eligible if eligible else 0.0
            for eligible, accepted in zip(
                self._position_eligible, self._position_accepted
            )
        )
        previous_num_spec_tokens = self.num_spec_tokens
        acceptance_ratchet = False
        decision_reason = "hold"

        tail_gain_23 = (
            (self._position_accepted[2] + self._position_accepted[3])
            / self._num_drafts
            if self._num_drafts and len(self._position_accepted) >= 4
            else 0.0
        )
        position_4_gain = (
            self._position_accepted[4] / self._num_drafts
            if self._num_drafts and len(self._position_accepted) >= 5
            else 0.0
        )

        # Production GLM policy: k=2 is the safe baseline. Head acceptance
        # only decides whether to probe k=4. Once above baseline, decisions
        # use the unconditional marginal tokens earned by the extra draft
        # positions, avoiding a second p0/p1 gate.
        if self.depth_ladder in ((2, 4), (2, 4, 5)):
            if self.num_spec_tokens == 2:
                head_ratio = (
                    mean_num_accepted_tokens / mean_num_draft_tokens
                    if mean_num_draft_tokens
                    else 0.0
                )
                target_num_spec_tokens = 4 if head_ratio >= 0.85 else 2
                decision_reason = (
                    "probe_k4"
                    if target_num_spec_tokens == 4
                    else "k2_baseline"
                )
                acceptance_ratchet = (
                    target_num_spec_tokens > self.num_spec_tokens
                )
            elif self.num_spec_tokens == 4:
                if (
                    self.depth_ladder == (2, 4, 5)
                    and tail_gain_23 >= 0.70
                ):
                    target_num_spec_tokens = 5
                    decision_reason = "probe_k5"
                    acceptance_ratchet = True
                elif tail_gain_23 >= 0.35:
                    target_num_spec_tokens = 4
                    decision_reason = "k4_hold"
                else:
                    target_num_spec_tokens = 2
                    decision_reason = "k4_tail_reject"
            else:
                if position_4_gain >= 0.15:
                    target_num_spec_tokens = 5
                    decision_reason = "k5_hold"
                elif tail_gain_23 >= 0.35:
                    target_num_spec_tokens = 4
                    decision_reason = "k5_p4_reject"
                else:
                    target_num_spec_tokens = 2
                    decision_reason = "k5_tail_reject"
        else:
            formula_target_num_spec_tokens = min(
                self.max_num_spec_tokens,
                max(1, floor(mean_num_accepted_tokens + 1.5)),
            )
            target_num_spec_tokens = max(
                (
                    depth
                    for depth in self.depth_ladder
                    if depth <= formula_target_num_spec_tokens
                ),
                default=self.depth_ladder[0],
            )
            decision_reason = "formula"

        self.num_spec_tokens = target_num_spec_tokens
        self._reset_window()
        return AcceptanceLengthUpdate(
            previous_num_spec_tokens=previous_num_spec_tokens,
            num_spec_tokens=self.num_spec_tokens,
            mean_num_accepted_tokens=mean_num_accepted_tokens,
            mean_num_draft_tokens=mean_num_draft_tokens,
            raw_target_num_spec_tokens=target_num_spec_tokens,
            acceptance_ratchet=acceptance_ratchet,
            decision_reason=decision_reason,
            position_conditional_acceptance=position_rates,
            observation_window=effective_observation_window,
            tail_gain_23=tail_gain_23,
            position_4_gain=position_4_gain,
        )

    def _reset_window(self) -> None:
        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0
        self._position_eligible = [0] * self.max_num_spec_tokens
        self._position_accepted = [0] * self.max_num_spec_tokens
'''

DEPTH_LADDER_SOURCE = '''\
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# R9: canonical parsing of the VLLM_ADAPTIVE_SPEC_DEPTHS depth ladder.
#
# The pinned community overlay (CosmicRaisins/glm-5.2-gb10
# 600848707ce93fe42fedbc9dd4429116696e425d,
# patches/adaptive-mtp-vllm-hooks.patch) parses this variable TWICE, once in
# `vllm/v1/core/sched/scheduler.py` and once in
# `vllm/v1/worker/gpu/cudagraph_utils.py`, with two different results: the
# scheduler unions the configured `num_speculative_tokens` into its snap points
# and the CUDA-graph side does not. Any ladder whose largest rung is below the
# configured maximum therefore lets the scheduler select a depth the graph layer
# never captured a descriptor for, which silently drops that step to eager.
#
# R9 keeps ONE parser and both call sites use it, so the scheduled depth set and
# the captured descriptor set are the same object by construction. This is a
# deliberate reconciliation of the pinned patch, recorded in
# evidence/r9-adaptive-full-implementation-report.md.
from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

ADAPTIVE_SPEC_DEPTHS_ENV = "VLLM_ADAPTIVE_SPEC_DEPTHS"
# The R9 production ladder. The community overlay's own default is "2,4"; R9
# ships the tuned 2/4/5 recipe from adaptive-mtp/README.md as the default so an
# operator cannot get a silently different ladder by forgetting the variable.
ADAPTIVE_SPEC_DEPTHS_DEFAULT = "2,4,5"


def parse_adaptive_spec_depth_ladder(
    max_num_spec_tokens: int,
    raw: str | None = None,
) -> tuple[int, ...]:
    """Resolve the effective adaptive speculative-depth ladder.

    `max_num_spec_tokens` (the configured `num_speculative_tokens`) is the hard
    upper bound and is ALWAYS a rung of the returned ladder: it is the depth the
    speculator's buffers, the draft-token store and the fixed-K fallback are all
    sized for, so leaving it uncaptured would be the one hole that cannot be
    recovered from at runtime.

    Returns a sorted, de-duplicated tuple of depths in `[1, max_num_spec_tokens]`.
    """
    if max_num_spec_tokens <= 0:
        raise ValueError("max_num_spec_tokens must be greater than zero.")

    if raw is None:
        raw = os.getenv(ADAPTIVE_SPEC_DEPTHS_ENV, ADAPTIVE_SPEC_DEPTHS_DEFAULT)

    def _parse(text: str) -> list[int] | None:
        try:
            return sorted({int(tok) for tok in text.split(",") if tok.strip()})
        except ValueError:
            return None

    depths = _parse(raw)
    if depths is None:
        logger.warning(
            "%s=%r is not a comma-separated list of ints; falling back to %r.",
            ADAPTIVE_SPEC_DEPTHS_ENV,
            raw,
            ADAPTIVE_SPEC_DEPTHS_DEFAULT,
        )
        depths = _parse(ADAPTIVE_SPEC_DEPTHS_DEFAULT) or []

    valid = [depth for depth in depths if 1 <= depth <= max_num_spec_tokens]
    if valid != depths:
        logger.warning(
            "%s=%r: dropped out-of-range depths (valid range [1, %d]); using %s.",
            ADAPTIVE_SPEC_DEPTHS_ENV,
            raw,
            max_num_spec_tokens,
            valid,
        )
    if not valid:
        logger.warning(
            "%s=%r left no usable depth; falling back to the configured "
            "maximum %d.",
            ADAPTIVE_SPEC_DEPTHS_ENV,
            raw,
            max_num_spec_tokens,
        )

    return tuple(sorted(set(valid) | {max_num_spec_tokens}))
'''

# path -> list of (anchor, replacement). Each anchor must occur exactly once in
# the pre-image; each replacement contains its anchor's surroundings so
# re-running is a no-op.
#
# NOTE: cudagraph_utils.py anchors on the raw e232d262 blob (no earlier patch
# touches it). scheduler.py and config/vllm.py anchor on the R8 post-image.
HUNKS: dict[str, list[tuple[str, str]]] = {
    'vllm/v1/core/sched/scheduler.py': [
        (
            '''\
import itertools
import time
from collections import defaultdict, deque
''',
            '''\
import itertools
import json
import os
import time
from collections import defaultdict, deque
''',
        ),
        (
            '''\
from vllm.v1.spec_decode.dynamic.acceptance_length import (
    AcceptanceLengthController,
)
from vllm.v1.spec_decode.dynamic.utils import''',
            '''\
from vllm.v1.spec_decode.dynamic.acceptance_length import (
    AcceptanceLengthController,
)
from vllm.v1.spec_decode.dynamic.depth_ladder import (
    parse_adaptive_spec_depth_ladder,
)
from vllm.v1.spec_decode.dynamic.utils import''',
        ),
        (
            '''\
        self.dynamic_sd_lookup: list[int] | None = None
        self.acceptance_length_controller: AcceptanceLengthController | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            if (
                observation_window := (
                    speculative_config.adaptive_speculative_tokens_window
                )
            ) is not None:
                self.acceptance_length_controller = AcceptanceLengthController(
                    max_num_spec_tokens=self.num_spec_tokens,
                    observation_window=observation_window,
                )
            if speculative_config.use_eagle():
''',
            '''\
        self.dynamic_sd_lookup: list[int] | None = None
        self.acceptance_length_controller: AcceptanceLengthController | None = None
        # R9: the effective depth ladder, resolved once from
        # VLLM_ADAPTIVE_SPEC_DEPTHS. Empty when adaptation is off.
        self.adaptive_spec_depth_ladder: tuple[int, ...] = ()
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            if (
                observation_window := (
                    speculative_config.adaptive_speculative_tokens_window
                )
            ) is not None:
                # Parsed by the SAME helper the CUDA-graph descriptor builder
                # uses, so every depth the scheduler can select is a depth a
                # FULL graph was captured for. See depth_ladder.py.
                self.adaptive_spec_depth_ladder = parse_adaptive_spec_depth_ladder(
                    self.num_spec_tokens
                )
                self.acceptance_length_controller = AcceptanceLengthController(
                    max_num_spec_tokens=self.num_spec_tokens,
                    observation_window=observation_window,
                    depth_ladder=self.adaptive_spec_depth_ladder,
                )
                logger.info(
                    "Adaptive speculative depth enabled: ladder=%s, "
                    "max_k=%d, baseline window=%d steps, exploratory window="
                    "%d steps.",
                    list(self.acceptance_length_controller.depth_ladder),
                    self.num_spec_tokens,
                    observation_window,
                    max(1, observation_window // 2),
                )
            if speculative_config.use_eagle():
''',
        ),
        (
            '''\
                self.num_lookahead_tokens = self.num_spec_tokens + 1

        # Create the KV cache manager.
''',
            '''\
                self.num_lookahead_tokens = self.num_spec_tokens + 1

        # ── R9: MTP instrumentation. ──────────────────────────────────────
        #
        # Every field below is derived from CPU-side counts the scheduler
        # already holds after verification (`scheduled_spec_decode_tokens` and
        # the sampled-token lists). No forward pass, no device read, no
        # synchronization is added for telemetry.
        self.mtp_instrument_enabled = os.getenv(
            "VLLM_MTP_INSTRUMENT", "0"
        ).strip().lower() not in {"", "0", "false", "no", "off"}
        try:
            self.mtp_instrument_window = max(
                1, int(os.getenv("VLLM_MTP_INSTRUMENT_WINDOW", "32"))
            )
        except ValueError:
            self.mtp_instrument_window = 32
        self._mtp_num_positions = max(1, self.num_spec_tokens)
        self._mtp_window_index = 0
        self._mtp_selected_k_run: dict[int, int] = {}
        self._reset_mtp_window()

        # Create the KV cache manager.
''',
        ),
        (
            '''\
        adaptive_num_drafts = 0
        adaptive_num_draft_tokens = 0
        adaptive_num_accepted_tokens = 0
        acceptance_length_controller = self.acceptance_length_controller
''',
            '''\
        adaptive_num_drafts = 0
        adaptive_num_draft_tokens = 0
        adaptive_num_accepted_tokens = 0
        # R9: per-position eligibility/acceptance, and the instrumentation
        # samples, are derived from the same CPU-side counts.
        adaptive_position_eligible = [0] * self._mtp_num_positions
        adaptive_position_accepted = [0] * self._mtp_num_positions
        mtp_samples: list[tuple[int, int, int, int]] = []
        acceptance_length_controller = self.acceptance_length_controller
''',
        ),
        (
            '''\
                # Skip a stale frame still pending discard (async_tokens_to_discard
                # > 0): its pre-reset rejection count would bias the controller.
                # Only the adaptive counters are gated here -- the R7 rejection
                # accounting below keeps its existing behaviour untouched.
                if (
                    acceptance_length_controller is not None
                    and request.async_tokens_to_discard == 0
                ):
                    adaptive_num_drafts += 1
                    adaptive_num_draft_tokens += num_draft_tokens
                    adaptive_num_accepted_tokens += num_accepted
                # num_computed_tokens represents the number of tokens
''',
            '''\
                # Skip a stale frame still pending discard (async_tokens_to_discard
                # > 0): its pre-reset rejection count would bias the controller.
                # Only the adaptive counters are gated here -- the R7 rejection
                # accounting below keeps its existing behaviour untouched.
                # R9 gates instrumentation on the same condition: a window that
                # counted stale frames would misreport the very acceptance
                # ratio the canary is meant to decide on.
                if request.async_tokens_to_discard == 0:
                    mtp_samples.append(
                        (
                            num_draft_tokens,
                            num_accepted,
                            request.num_computed_tokens,
                            num_sampled + num_accepted,
                        )
                    )
                    if acceptance_length_controller is not None:
                        adaptive_num_drafts += 1
                        adaptive_num_draft_tokens += num_draft_tokens
                        adaptive_num_accepted_tokens += num_accepted
                        # Conditional per-position acceptance from the same
                        # counts. Position p (1-based) is *eligible* when the
                        # batch drafted at least p tokens AND every earlier
                        # position was accepted -- rejection sampling stops at
                        # the first miss, so a later position is never reached.
                        for pos in range(1, self._mtp_num_positions + 1):
                            if num_draft_tokens < pos:
                                break
                            if pos == 1 or num_accepted >= pos - 1:
                                adaptive_position_eligible[pos - 1] += 1
                            if num_accepted >= pos:
                                adaptive_position_accepted[pos - 1] += 1
                # num_computed_tokens represents the number of tokens
''',
        ),
        (
            '''\
        # Recorded BEFORE observe_batch: these stats describe the step that has
        # just completed, and observe_batch may move the controller's depth on
        # a window boundary. Reading it afterwards would report the depth
        # selected for the *next* step against this step's acceptance numbers.
        if spec_decoding_stats is not None:
            spec_decoding_stats.current_num_spec_tokens = (
                scheduler_output.resolve_num_spec_tokens_to_schedule(
                    self.num_spec_tokens
                )
            )

        if acceptance_length_controller is not None:
            update = acceptance_length_controller.observe_batch(
                num_drafts=adaptive_num_drafts,
                num_draft_tokens=adaptive_num_draft_tokens,
                num_accepted_tokens=adaptive_num_accepted_tokens,
            )
            if (
                update is not None
                and update.previous_num_spec_tokens != update.num_spec_tokens
            ):
                logger.debug(
                    "Adaptive speculative depth changed from %d to %d "
                    "(mean accepted drafts: %.2f, mean attempted drafts: %.2f, "
                    "window: %d steps).",
                    update.previous_num_spec_tokens,
                    update.num_spec_tokens,
                    update.mean_num_accepted_tokens,
                    update.mean_num_draft_tokens,
                    acceptance_length_controller.observation_window,
                )

        # Remove the stopped requests from the running and waiting queues.
''',
            '''\
        # Recorded BEFORE observe_batch: these stats describe the step that has
        # just completed, and observe_batch may move the controller's depth on
        # a window boundary. Reading it afterwards would report the depth
        # selected for the *next* step against this step's acceptance numbers.
        # `resolve_...` returns the fixed K for synthetic outputs (warmup and
        # dummy runs never went through the scheduler).
        active_num_spec_tokens = scheduler_output.resolve_num_spec_tokens_to_schedule(
            self.num_spec_tokens
        )
        if spec_decoding_stats is not None:
            spec_decoding_stats.current_num_spec_tokens = active_num_spec_tokens

        update = None
        if acceptance_length_controller is not None:
            update = acceptance_length_controller.observe_batch(
                num_drafts=adaptive_num_drafts,
                num_draft_tokens=adaptive_num_draft_tokens,
                num_accepted_tokens=adaptive_num_accepted_tokens,
                position_eligible=adaptive_position_eligible,
                position_accepted=adaptive_position_accepted,
            )
            if (
                update is not None
                and update.previous_num_spec_tokens != update.num_spec_tokens
            ):
                logger.info(
                    "Adaptive speculative depth changed from %d to %d "
                    "(mean accepted drafts: %.2f, mean attempted drafts: %.2f, "
                    "window: %d steps, reason: %s, tail gain p2+p3: %.2f, "
                    "p4 gain: %.2f, conditional position acceptance: %s).",
                    update.previous_num_spec_tokens,
                    update.num_spec_tokens,
                    update.mean_num_accepted_tokens,
                    update.mean_num_draft_tokens,
                    update.observation_window,
                    update.decision_reason,
                    update.tail_gain_23,
                    update.position_4_gain,
                    ",".join(
                        f"{rate:.2f}"
                        for rate in update.position_conditional_acceptance
                    ),
                )

        self._record_mtp_instrumentation(
            mtp_samples, update, active_num_spec_tokens
        )

        # Remove the stopped requests from the running and waiting queues.
''',
        ),
        (
            '''\
        return spec_decoding_stats

    def shutdown(self) -> None:
''',
            '''\
        return spec_decoding_stats

    # ── R9: MTP instrumentation ────────────────────────────────────────────

    def _reset_mtp_window(self) -> None:
        self._mtp_window_steps = 0
        self._mtp_window_batches = 0
        self._mtp_window_draft_tokens = 0
        self._mtp_window_accepted_tokens = 0
        self._mtp_window_output_tokens = 0
        self._mtp_window_started_ns = 0
        self._mtp_window_context_min: int | None = None
        self._mtp_window_context_max: int | None = None
        self._mtp_window_position_eligible = [0] * self._mtp_num_positions
        self._mtp_window_position_accepted = [0] * self._mtp_num_positions
        self._mtp_window_selected_k: dict[int, int] = {}
        self._mtp_window_active_k_first: int | None = None

    def _record_mtp_instrumentation(
        self,
        samples: list[tuple[int, int, int, int]],
        update: Any,
        active_num_spec_tokens: int,
    ) -> None:
        """Accumulate one scheduler step and emit MTP_WINDOW_JSON on a boundary.

        `samples` carries `(num_draft_tokens, num_accepted, num_computed_tokens,
        num_output_tokens)` per verified request, all CPU-side values the
        scheduler already computed. Nothing here reads a device tensor.
        """
        if not self.mtp_instrument_enabled:
            return

        self._mtp_window_steps += 1
        if self._mtp_window_active_k_first is None:
            self._mtp_window_active_k_first = active_num_spec_tokens
        self._mtp_window_selected_k[active_num_spec_tokens] = (
            self._mtp_window_selected_k.get(active_num_spec_tokens, 0) + 1
        )
        self._mtp_selected_k_run[active_num_spec_tokens] = (
            self._mtp_selected_k_run.get(active_num_spec_tokens, 0) + 1
        )

        if samples:
            if self._mtp_window_started_ns == 0:
                self._mtp_window_started_ns = time.monotonic_ns()
            self._mtp_window_batches += len(samples)
            for draft, accepted, context_tokens, output_tokens in samples:
                self._mtp_window_draft_tokens += draft
                self._mtp_window_accepted_tokens += accepted
                self._mtp_window_output_tokens += output_tokens
                if self._mtp_window_context_min is None:
                    self._mtp_window_context_min = context_tokens
                    self._mtp_window_context_max = context_tokens
                else:
                    self._mtp_window_context_min = min(
                        self._mtp_window_context_min, context_tokens
                    )
                    self._mtp_window_context_max = max(
                        self._mtp_window_context_max, context_tokens
                    )
                for pos in range(1, self._mtp_num_positions + 1):
                    if draft < pos:
                        break
                    if pos == 1 or accepted >= pos - 1:
                        self._mtp_window_position_eligible[pos - 1] += 1
                    if accepted >= pos:
                        self._mtp_window_position_accepted[pos - 1] += 1

        if self._mtp_window_steps < self.mtp_instrument_window:
            return

        batches = self._mtp_window_batches
        attempted = self._mtp_window_draft_tokens
        accepted = self._mtp_window_accepted_tokens
        eligible = self._mtp_window_position_eligible
        pos_accepted = self._mtp_window_position_accepted
        record = {
            "schema": 2,
            "window_index": self._mtp_window_index,
            # ── configuration ──
            "configured_max_k": self.num_spec_tokens,
            "depth_ladder": list(self.adaptive_spec_depth_ladder),
            "adaptive_enabled": self.acceptance_length_controller is not None,
            # ── depth actually in force ──
            "active_k": active_num_spec_tokens,
            "active_k_at_window_start": self._mtp_window_active_k_first,
            "controller_previous_k": getattr(
                update, "previous_num_spec_tokens", None
            ),
            "controller_next_k": getattr(update, "num_spec_tokens", None),
            "decision_reason": getattr(update, "decision_reason", None)
            if update is not None
            else ("no_decision" if self.acceptance_length_controller else "fixed"),
            "controller_window": getattr(update, "observation_window", None),
            "acceptance_ratchet": getattr(update, "acceptance_ratchet", None),
            # ── volumes. `draft_batches` is the denominator for every
            #    *_per_batch and *_gain field below. ──
            "verification_steps": self._mtp_window_steps,
            "draft_batches": batches,
            "drafts_attempted": attempted,
            "drafts_accepted": accepted,
            # aggregate accepted-draft ratio; denominator is drafts_attempted,
            # NOT draft_batches and NOT configured_max_k * draft_batches.
            "acceptance_ratio": (accepted / attempted) if attempted else 0.0,
            "acceptance_ratio_denominator": "drafts_attempted",
            "mean_attempted_per_batch": (attempted / batches) if batches else 0.0,
            "mean_accepted_per_batch": (accepted / batches) if batches else 0.0,
            # ── per-position, p0..p{max_k-1} ──
            "position_eligible": list(eligible),
            "position_accepted": list(pos_accepted),
            "position_conditional_acceptance": [
                (pos_accepted[i] / eligible[i]) if eligible[i] else 0.0
                for i in range(self._mtp_num_positions)
            ],
            "position_unconditional_gain": [
                (pos_accepted[i] / batches) if batches else 0.0
                for i in range(self._mtp_num_positions)
            ],
            # The two policy inputs, recomputed here from the window's own
            # counts so a window record can be audited without the controller.
            "tail_gain_23": (
                (pos_accepted[2] + pos_accepted[3]) / batches
                if batches and self._mtp_num_positions >= 4
                else 0.0
            ),
            "position_4_gain": (
                pos_accepted[4] / batches
                if batches and self._mtp_num_positions >= 5
                else 0.0
            ),
            "controller_tail_gain_23": getattr(update, "tail_gain_23", None),
            "controller_position_4_gain": getattr(update, "position_4_gain", None),
            # ── throughput / context envelope ──
            "output_tokens_emitted": self._mtp_window_output_tokens,
            "window_elapsed_ms": (
                (time.monotonic_ns() - self._mtp_window_started_ns) / 1e6
                if self._mtp_window_started_ns
                else 0.0
            ),
            "context_tokens_min": self._mtp_window_context_min,
            "context_tokens_max": self._mtp_window_context_max,
            # ── selected-K distribution, in scheduler steps ──
            "selected_k_window": {
                str(k): v for k, v in sorted(self._mtp_window_selected_k.items())
            },
            "selected_k_run": {
                str(k): v for k, v in sorted(self._mtp_selected_k_run.items())
            },
        }
        logger.info("MTP_WINDOW_JSON %s", json.dumps(record, sort_keys=True))
        self._mtp_window_index += 1
        self._reset_mtp_window()

    def shutdown(self) -> None:
''',
        ),
    ],
    'vllm/config/vllm.py': [
        (
            '''\
    def _maybe_override_dynamic_sd_cudagraph_mode(self) -> None:
        speculative_config = self.speculative_config
        if (
            speculative_config is None
            or not speculative_config.uses_dynamic_speculative_decoding()
            or not self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            return

        logger.warning_once(
''',
            '''\
    def _maybe_override_dynamic_sd_cudagraph_mode(self) -> None:
        speculative_config = self.speculative_config
        if (
            speculative_config is None
            or not speculative_config.uses_dynamic_speculative_decoding()
            or not self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            return

        # ── R9: the V2 full-CUDA-graph exemption. ──────────────────────────
        #
        # e232d262 downgrades ANY dynamic speculative decoding to PIECEWISE
        # because the target verification length moves at runtime, and a
        # descriptor set captured for one uniform query length cannot serve
        # another. R8 kept that verbatim, which made adaptive MTP a trade of
        # graph mode for draft depth.
        #
        # R9 removes the trade for acceptance-length adaptation on the V2
        # runner ONLY, because it also removes the premise:
        # CudaGraphManager._init_candidates() now captures a FULL decode
        # descriptor for every rung of the VLLM_ADAPTIVE_SPEC_DEPTHS ladder at
        # every request count up to max_num_seqs, and the scheduler can only
        # select depths from that same ladder (both read
        # vllm/v1/spec_decode/dynamic/depth_ladder.py). Every reachable
        # verification shape therefore has a captured graph.
        #
        # The exemption is deliberately narrow and fails closed:
        #   * V1 runner            -> still downgraded (V1 has no multi-depth
        #                             descriptor path and reads
        #                             num_spec_tokens_to_schedule unresolved).
        #   * batch-size schedule  -> still downgraded (its K comes from the
        #                             batch size, not the ladder, so the
        #                             captured set does not bound it).
        if (
            speculative_config.uses_acceptance_length_adaptation()
            and not (
                speculative_config.uses_batch_size_dynamic_speculative_decoding()
            )
            and self.use_v2_model_runner
        ):
            logger.info_once(
                "Acceptance-length adaptive speculative decoding is keeping "
                "cudagraph_mode=%s: the V2 runner captures a FULL decode "
                "graph for every depth on the adaptive ladder.",
                self.compilation_config.cudagraph_mode.name,
            )
            return

        logger.warning_once(
''',
        ),
    ],
    'vllm/v1/worker/gpu/cudagraph_utils.py': [
        (
            '''\
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
''',
            '''\
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.dynamic.depth_ladder import (
    parse_adaptive_spec_depth_ladder,
)
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
''',
        ),
        (
            '''\
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
''',
            '''\
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            # R9: fail closed BEFORE this early return, not after it.
            #
            # `cudagraph_mode` is falsy for CUDAGraphMode.NONE (value 0), which
            # is what `--enforce-eager` and every backend-driven downgrade to
            # eager resolve to, and `cudagraph_capture_sizes` is empty when the
            # compilation config captures nothing at all. In both cases every
            # reachable adaptive depth would execute eagerly. Returning here
            # used to skip `_assert_adaptive_spec_graph_coverage()` entirely,
            # so the FULL-mode rejection below was unreachable for exactly the
            # mode it most needed to reject.
            self._assert_adaptive_spec_graphs_possible(capture_sizes)
            return

        capture_sizes = sorted(capture_sizes)
''',
        ),
        (
            '''\
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)

        if not descs_by_token_lora:
            return

        all_token_counts = sorted({k[0] for k in descs_by_token_lora})
        current_range_start = 0
        for token_cg_size in all_token_counts:
            for i in range(current_range_start, token_cg_size + 1):
                for num_active_loras in self.lora_capture_cases:
                    staging_key = (token_cg_size, num_active_loras)
                    if staging_key in descs_by_token_lora:
                        self._candidates[(i, num_active_loras)] = descs_by_token_lora[
                            staging_key
                        ]
            current_range_start = token_cg_size + 1

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

    def needs_capture(self) -> bool:
''',
            '''\
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)

        # R9: FULL decode descriptors for every other rung of the adaptive
        # speculative-depth ladder. Additive -- nothing above is changed.
        adaptive_descs_by_token_lora = self._init_adaptive_spec_candidates(
            capture_sizes,
            max_decode_tokens,
            decode_mode,
            separate_decode_routine,
            descs_by_mode,
        )

        if not descs_by_token_lora:
            # The same failure one step later: a resolved mode that produced no
            # descriptor for any capture size cannot serve any adaptive depth.
            self._assert_adaptive_spec_graphs_possible(capture_sizes)
            return

        self._candidates = self._build_candidate_ranges(descs_by_token_lora)

        # Adaptive descriptors are PREPENDED to the base candidate list for
        # their own (tighter) token-count range, never substituted for it.
        #
        # This is a deliberate reconciliation of the pinned community patch,
        # which merges both descriptor sets into one `descs_by_token_lora` and
        # so re-partitions the token-count ranges. Under the FULL_AND_PIECEWISE
        # default that silently REMOVES mixed-batch coverage: introducing a
        # bucket at, say, 5 tokens makes a 4-token prefill resolve to a bucket
        # that holds only uniform-5 decode descriptors, which no mixed batch is
        # compatible with, so it drops to eager instead of the PIECEWISE graph
        # at 6 it used to get. Layering keeps every base lookup reachable:
        # `dispatch` walks the list in order and the base entries are still
        # there, just after the exact-depth ones.
        for key, adaptive_descs in self._build_candidate_ranges(
            adaptive_descs_by_token_lora
        ).items():
            base_descs = self._candidates.get(key, ())
            merged = list(adaptive_descs)
            merged.extend(desc for desc in base_descs if desc not in merged)
            self._candidates[key] = merged

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

        self._assert_adaptive_spec_graph_coverage(separate_decode_routine)

    def _assert_adaptive_spec_graphs_possible(
        self, capture_sizes: list[int] | None
    ) -> None:
        """Fail closed on the paths that never reach descriptor construction.

        `_init_candidates` has two early returns that predate R9: one for a
        falsy/NONE `cudagraph_mode` or an empty capture-size list, one for a
        descriptor set that came out empty anyway. Neither can produce a
        captured graph for any depth, so if this manager owns an adaptive
        verification shape the run must abort here rather than proceed to
        execute the ladder eagerly.
        """
        decode_query_lens = self.adaptive_spec_decode_query_lens()
        if not decode_query_lens:
            return
        raise RuntimeError(
            "Adaptive speculative depth requires FULL CUDA graphs, but this "
            "run captures no CUDA graphs at all: cudagraph_mode="
            f"{getattr(self.cudagraph_mode, 'name', self.cudagraph_mode)}, "
            f"cudagraph_capture_sizes={list(capture_sizes or [])}. Every depth "
            f"on the ladder ({decode_query_lens} verification query lengths) "
            "would run eagerly. Remove --enforce-eager, restore a non-empty "
            "cudagraph_capture_sizes, or disable adaptive speculative depth."
        )

    def _assert_adaptive_spec_graph_coverage(
        self, separate_decode_routine: bool
    ) -> None:
        """Fail closed when adaptive depth would run without FULL graphs.

        R9's whole premise is that the adaptive ladder and the captured
        descriptor set are the same set. An environment variable saying so is
        not evidence -- `cudagraph_mode` is resolved from the attention
        backend's capabilities at load time and can land on PIECEWISE or NONE
        for reasons no launch script can see. This check runs against the
        RESOLVED mode and the descriptors actually built, and raises rather
        than serving a silently degraded candidate.

        There is deliberately NO opt-out. An earlier revision honoured
        `VLLM_ADAPTIVE_SPEC_ALLOW_DEGRADED_GRAPHS=1`, which meant one
        environment variable -- settable by anything that can reach the
        container environment -- could turn a proven R9 launch back into a
        silently-eager one after every launch-script guard had passed. The
        variable is now inert.
        """
        decode_query_lens = self.adaptive_spec_decode_query_lens()
        if not decode_query_lens:
            return

        if not self.cudagraph_mode.has_full_cudagraphs():
            raise RuntimeError(
                "Adaptive speculative depth requires FULL CUDA graphs, but "
                f"cudagraph_mode resolved to {self.cudagraph_mode.name}. Every "
                "depth on the ladder would run without a captured graph. Fix "
                "the attention backend or compilation config, or disable "
                "adaptive speculative depth."
            )
        if not separate_decode_routine:
            # A mixed FULL descriptor carries `uniform_token_count=None` and is
            # compatible with every uniform query length, so there is no
            # per-depth set to check.
            return

        covered = {
            (d.num_tokens, d.num_reqs, d.uniform_token_count, d.num_active_loras)
            for d in self._capture_descs.get(CUDAGraphMode.FULL, ())
        }
        missing = [
            (query_len, num_reqs)
            for query_len, num_reqs, num_active_loras in product(
                decode_query_lens,
                range(1, min(self.max_num_reqs, 32) + 1),
                self.lora_capture_cases,
            )
            if (
                query_len * num_reqs,
                num_reqs,
                query_len,
                num_active_loras,
            )
            not in covered
        ]
        if missing:
            raise RuntimeError(
                "Adaptive speculative depth has no captured FULL CUDA graph "
                f"for {len(missing)} reachable verification shape(s) "
                f"(query_len, num_reqs): {sorted(set(missing))}. The scheduler "
                "could select those depths and would fall back to eager "
                "execution. Raise max_cudagraph_capture_size to at least "
                f"{max(q * n for q, n in missing)}, shorten "
                "VLLM_ADAPTIVE_SPEC_DEPTHS, or lower max_num_seqs."
            )

        logger.info(
            "Adaptive speculative depth: FULL CUDA graph coverage verified for "
            "query lengths %s across request counts 1..%d.",
            decode_query_lens,
            min(self.max_num_reqs, 32),
        )

    def _build_candidate_ranges(
        self,
        descs_by_token_lora: dict[tuple[int, int], list[BatchExecutionDescriptor]],
    ) -> dict[tuple[int, int], list[BatchExecutionDescriptor]]:
        """Map every token count to the descriptors of the smallest bucket
        that can still hold it. Extracted verbatim from `_init_candidates`."""
        candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        all_token_counts = sorted({k[0] for k in descs_by_token_lora})
        current_range_start = 0
        for token_cg_size in all_token_counts:
            for i in range(current_range_start, token_cg_size + 1):
                for num_active_loras in self.lora_capture_cases:
                    staging_key = (token_cg_size, num_active_loras)
                    if staging_key in descs_by_token_lora:
                        candidates[(i, num_active_loras)] = descs_by_token_lora[
                            staging_key
                        ]
            current_range_start = token_cg_size + 1
        return candidates

    def adaptive_spec_decode_query_lens(self) -> list[int]:
        """Uniform verification query lengths reachable under adaptive MTP.

        One entry per rung of the `VLLM_ADAPTIVE_SPEC_DEPTHS` ladder, each
        `depth + num_new_sampled_tokens_per_step` wide -- the same arithmetic
        `GPUModelRunner` uses to derive `decode_query_len` from the fixed K.

        Empty (and therefore a no-op) unless acceptance-length adaptation is
        configured AND this manager owns the verification shape. The draft
        decode manager is built with `decode_query_len == 1`: it drafts one
        token per step at every depth, so it has nothing to follow.

        Once adaptation IS configured and this manager DOES own the shape, an
        empty or unrecognised result is a failure, not a no-op: it would let
        every caller conclude "nothing to cover" and serve the ladder with no
        captured graph behind it. Those cases raise.
        """
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None or self.decode_query_len <= 1:
            return []
        if (
            getattr(speculative_config, "adaptive_speculative_tokens_window", None)
            is None
        ):
            return []

        # Past this point adaptation is configured and this manager owns the
        # verification shape. Nothing below may answer "no depths to cover".
        max_num_spec_tokens = self.vllm_config.num_speculative_tokens
        if max_num_spec_tokens <= 0:
            raise RuntimeError(
                "Adaptive speculative depth is configured "
                "(adaptive_speculative_tokens_window="
                f"{speculative_config.adaptive_speculative_tokens_window}) but "
                f"num_speculative_tokens resolved to {max_num_spec_tokens}. "
                "There is no ladder to capture graphs for and no way to tell "
                "which verification shapes the scheduler can produce."
            )
        num_new_sampled_tokens_per_step = self.decode_query_len - max_num_spec_tokens
        if num_new_sampled_tokens_per_step < 1:
            # Not the `K + new sampled tokens` verification shape this port was
            # derived against. Capturing a shape whose meaning is unproven is
            # wrong, and so is capturing nothing: the scheduler would still
            # select ladder depths and run them eagerly.
            raise RuntimeError(
                "Adaptive speculative depth is configured but decode_query_len "
                f"({self.decode_query_len}) is not greater than "
                f"num_speculative_tokens ({max_num_spec_tokens}), so the "
                "per-depth verification shape cannot be derived. This is not "
                "the 'K + newly sampled tokens' verifier layout R9 captures "
                "graphs for; refusing to run adaptive depth against an "
                "unrecognised verifier shape."
            )

        decode_query_lens = [
            depth + num_new_sampled_tokens_per_step
            for depth in parse_adaptive_spec_depth_ladder(max_num_spec_tokens)
        ]
        if not decode_query_lens:
            raise RuntimeError(
                "Adaptive speculative depth is configured but the depth ladder "
                f"parsed to an empty set for num_speculative_tokens="
                f"{max_num_spec_tokens}. Refusing to report 'no depths to "
                "cover' for a run that will still adapt its depth."
            )
        return decode_query_lens

    def _init_adaptive_spec_candidates(
        self,
        capture_sizes: list[int],
        max_decode_tokens: int,
        decode_mode: CUDAGraphMode,
        separate_decode_routine: bool,
        descs_by_mode: dict[CUDAGraphMode, list[BatchExecutionDescriptor]],
    ) -> dict[tuple[int, int], list[BatchExecutionDescriptor]]:
        """Add a FULL decode descriptor per (adaptive depth, request count).

        The request count is enumerated exactly rather than rounded up to a
        capture size, which is what makes the coverage claim checkable: for
        `max_num_seqs = R` and ladder `D`, the captured set is exactly
        `{(d + s) * n for d in D for n in 1..R}`, every uniform verification
        shape the scheduler can produce.
        """
        adaptive_descs_by_token_lora: dict[
            tuple[int, int], list[BatchExecutionDescriptor]
        ] = defaultdict(list)

        decode_query_lens = self.adaptive_spec_decode_query_lens()
        if not (decode_query_lens and separate_decode_routine and decode_mode):
            return adaptive_descs_by_token_lora

        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size
        # Exact request counts, mirroring `small_decode_sizes` in
        # `CompilationConfig.adjust_cudagraph_sizes_for_spec_decode`.
        max_captured_reqs = min(self.max_num_reqs, 32)
        if self.max_num_reqs > max_captured_reqs:
            logger.warning(
                "Adaptive speculative depth: capturing per-depth decode "
                "graphs for request counts 1..%d only; batches with more than "
                "%d requests will fall back to the fixed-K descriptors.",
                max_captured_reqs,
                max_captured_reqs,
            )

        existing = {
            (d.num_tokens, d.num_reqs, d.uniform_token_count, d.num_active_loras)
            for d in descs_by_mode[decode_mode]
        }
        skipped: list[tuple[int, int]] = []
        for decode_query_len, num_reqs, num_active_loras in product(
            decode_query_lens,
            range(1, max_captured_reqs + 1),
            self.lora_capture_cases,
        ):
            num_tokens = decode_query_len * num_reqs
            if num_tokens > max_decode_tokens or (
                max_cg_capture_size is not None and num_tokens > max_cg_capture_size
            ):
                skipped.append((decode_query_len, num_reqs))
                continue

            key = (num_tokens, num_reqs, decode_query_len, num_active_loras)
            if key in existing:
                # Already captured by the fixed-K path above. Capturing it
                # twice would trip the `desc not in self.graphs` assertion.
                continue
            existing.add(key)

            desc = BatchExecutionDescriptor(
                cg_mode=decode_mode,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform_token_count=decode_query_len,
                num_active_loras=num_active_loras,
            )
            descs_by_mode[decode_mode].append(desc)
            adaptive_descs_by_token_lora[(num_tokens, num_active_loras)].append(desc)

        if skipped:
            logger.warning(
                "Adaptive speculative depth: %d (query_len, num_reqs) decode "
                "shapes exceed max_cudagraph_capture_size=%s / "
                "max_decode_tokens=%d and were NOT captured: %s.",
                len(skipped),
                max_cg_capture_size,
                max_decode_tokens,
                sorted(set(skipped)),
            )
        else:
            logger.info(
                "Adaptive speculative depth: captured decode query lengths %s "
                "for request counts 1..%d (fixed decode_query_len=%d).",
                decode_query_lens,
                max_captured_reqs,
                self.decode_query_len,
            )
        return adaptive_descs_by_token_lora

    def needs_capture(self) -> bool:
''',
        ),
    ],
}


NEW_FILES = {
    CONTROLLER_PATH: CONTROLLER_SOURCE,
    DEPTH_LADDER_PATH: DEPTH_LADDER_SOURCE,
}


class PatchError(RuntimeError):
    pass


def apply_hunks(path: Path, name: str, hunks: list[tuple[str, str]]) -> bool:
    if not path.is_file():
        raise PatchError(f"{name}: missing from the target tree")

    text = path.read_text()
    original = text
    for index, (anchor, replacement) in enumerate(hunks, start=1):
        if replacement in text:
            # Already applied. Verify the hunk is not also present unpatched.
            if text.count(anchor) > text.count(replacement):
                raise PatchError(
                    f"{name}: hunk {index} is both applied and unapplied; "
                    "refusing to patch a partially-modified tree"
                )
            continue
        occurrences = text.count(anchor)
        if occurrences != 1:
            raise PatchError(
                f"{name}: hunk {index} anchor matched {occurrences} times, "
                "expected exactly 1. The base tree is not the R8 post-image "
                "(patch 03 -> 06 -> 12) over e232d262369b8c918cf478a7a96a0fcf"
                "8127cf65."
            )
        text = text.replace(anchor, replacement, 1)

    if text == original:
        return False
    path.write_text(text)
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <dist-packages-root>", file=sys.stderr)
        return 2

    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    changed = []
    try:
        for name, hunks in HUNKS.items():
            if apply_hunks(root / name, name, hunks):
                changed.append(name)

        for name, source in NEW_FILES.items():
            target = root / name
            if not target.parent.is_dir():
                raise PatchError(
                    f"{name}: {target.parent} is missing. The base must "
                    "already carry the batch-size dynamic-SD helpers."
                )
            existing = target.read_text() if target.is_file() else None
            if existing != source:
                target.write_text(source)
                changed.append(name)
    except PatchError as error:
        print(str(error), file=sys.stderr)
        return 1

    for name in changed:
        print(f"patched {name}")
    if not changed:
        print("already applied; nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
