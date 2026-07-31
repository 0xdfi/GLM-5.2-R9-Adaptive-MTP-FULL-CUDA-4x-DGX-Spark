# Runtime templates

Public, placeholder-only launch templates for the four-node GLM-5.2 R13 runtime.

R13 supersedes R9. The change set: two named profiles (`fast` / `balanced`),
both at C4 concurrency (four lanes, twelve-shape FULL-graph capture set
`6,12,18,24`), and a DCP-comm-arg guard that lets the DCP1 `fast` profile boot
clean. See [`../RELEASE_NOTES.md`](../RELEASE_NOTES.md) for the full changelog and
[`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) for the measured tradeoffs.

| File | Purpose |
|---|---|
| `r9.env.example` | every knob, with `<...>` placeholders and the reasoning next to each one |
| `start-node.sh` | fail-closed launcher; renders or launches for `NODE_ROLE=head` / `NODE_ROLE=worker` |

## Use

```bash
cp runtime/r9.env.example runtime/r9.env
$EDITOR runtime/r9.env                              # replace every <...>

# Inspect what would run, without running it:
RENDER_ONLY=1 NODE_ROLE=head   ./runtime/start-node.sh runtime/r9.env
RENDER_ONLY=1 NODE_ROLE=worker ./runtime/start-node.sh runtime/r9.env

# Bring up the three workers first, then the head node:
NODE_ROLE=worker ./runtime/start-node.sh runtime/r9.env    # on nodes 1..3
NODE_ROLE=head   ./runtime/start-node.sh runtime/r9.env    # on node 0
```

`runtime/r9.env` is gitignored. Keep it that way — it will contain your cluster's
addresses and interface names.

## These templates fail closed

`start-node.sh` renders nothing and launches nothing unless every invariant of the
qualified runtime holds. The exit codes mirror the production launcher so receipts are
comparable:

| Exit | Condition |
|---|---|
| 2 | `MTP_K` is not a bare positive integer |
| 4 | `ADAPTIVE_SPEC_WINDOW` is not a bare positive integer — `08` is valid bash and invalid JSON, so a leading zero is rejected on purpose |
| 5 | `VLLM_ADAPTIVE_SPEC_DEPTHS` does not normalise to exactly `2,4,5` |
| 6 | the ladder's top rung is not `MTP_K`, or `MTP_K != 5` |
| 7 | `ENFORCE_EAGER=1`, or `CUDAGRAPH_SIZES` is not the qualified `6,12,18,24`, or the largest reachable shape exceeds the largest capture size |
| 8 | MTP telemetry disabled, or a malformed instrumentation window |
| 10 | `MAX_NUM_SEQS` is anything other than exactly `4` |
| 11 | any required variable is unset, empty, or still contains a `<...>` placeholder |
| 12 | `R13_PROFILE` is not one of the authorized named profiles (`fast` / `balanced`) |
| 13 | a host precondition fails (missing model directory, missing cache directory, no `docker`) |

Note: `--decode-context-parallel-size` is derived from the profile (`fast`=1,
`balanced`=2). The launcher emits `--dcp-comm-backend` / `--dcp-kv-cache-interleave-size`
**only** at DCP>1; at DCP1 (fast) those flags are omitted because passing them
crashes the R13 engine on boot. This is the core R13 fix.

None of these is stylistic. `MAX_NUM_SEQS` is the multiplier that defines the
twelve-shape FULL-graph coverage set, and the server's own load-time assertion will refuse
to start if the launcher and the graph layer disagree — so the launcher refuses first,
before a maintenance window has been opened.

## What these templates deliberately do NOT do

The production launcher additionally owns container preservation under timestamped
rollback names, a rollback transaction, and a deadline watchdog that restores the
previous runtime on any failure or signal. **None of that is here.** Writing untestable
activation orchestration into a public template would add risk without adding evidence.
If you are cutting over a live service, write and review that orchestration for your own
environment first.

Also absent by design: any hostname, IP address, interface name, HCA name, filesystem
path, container name, image tag or username from the deployment these templates were
derived from. Every one of those is a placeholder.

## Security

The vLLM OpenAI-compatible server has **no authentication, no authorization and no
TLS** in this configuration. `API_BIND_ADDR` has no default and must be a private
address. Ray's cluster ports and NCCL/RoCE traffic must stay on a trusted, isolated
segment. `--trust-remote-code` executes model-repository Python as root inside the
container. Read [`../SECURITY.md`](../SECURITY.md) before exposing anything.

## Envelope changes

Context window, KV budget and DCP size are a matched triple and are selected by
**named profile**, not passed independently. R13 ships two profiles:

| Profile | `max_model_len` | KV budget / rank | DCP | Context ceiling | Character |
|---|---|---|---|---|---|
| `fast` | 319000 | 10.23 GB | 1 (none) | 319k | Short-context throughput. Highest prefill and C4-aggregate; lowest prose-decode C1. |
| `balanced` | 520000 | 8.41 GB | 2 (`a2a`) | 520k | Long-context capacity. Faster prose-decode C1; lower C4-aggregate. |

Both are C4. The R9 `520k`/`550k` pair and the R9 image (`sha256:50261a39…`) are
superseded — `550k` was the envelope under which the head node was OOM-killed by the
Linux kernel. Read [`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) §5 for the history.

Raising `MAX_NUM_BATCHED_TOKENS`, `kv_cache_memory_bytes` or `MAX_NUM_SEQS` on
unified-memory GB10 nodes with no swap is not a tuning exercise; the measured host
memory floor during a 500K-token request was 484.6 MiB. Re-measure before you change
any of them.
