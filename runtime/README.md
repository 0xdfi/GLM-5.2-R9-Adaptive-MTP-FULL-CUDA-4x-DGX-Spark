# Runtime templates

Public, placeholder-only launch templates for the four-node GLM-5.2 R9 runtime.

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
| 7 | `ENFORCE_EAGER=1`, or `CUDAGRAPH_SIZES` is not the qualified `6,12,18`, or the largest reachable shape exceeds the largest capture size |
| 8 | MTP telemetry disabled, or a malformed instrumentation window |
| 10 | `MAX_NUM_SEQS` is anything other than exactly `3` |
| 11 | any required variable is unset, empty, or still contains a `<...>` placeholder |
| 12 | `R9_PROFILE` is not one of the authorized named `max_model_len`/`kv_cache_memory_bytes` pairs |
| 13 | a host precondition fails (missing model directory, missing cache directory, no `docker`) |

None of these is stylistic. `MAX_NUM_SEQS` is the multiplier that defines the
nine-shape FULL-graph coverage set, and the server's own load-time assertion will refuse
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

`max_model_len` and `kv_cache_memory_bytes` are a matched pair and are selected by
**named profile**, not passed independently. `520k` is the qualified, currently-live
envelope. `550k` is the prior envelope, under which the head node was OOM-killed by the
Linux kernel with the nine-shape FULL graph set — read
[`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) §5 before choosing it.

Raising `MAX_NUM_BATCHED_TOKENS`, `kv_cache_memory_bytes` or `MAX_NUM_SEQS` on
unified-memory GB10 nodes with no swap is not a tuning exercise; the measured host
memory floor during a 500K-token request was 484.6 MiB. Re-measure before you change
any of them.
