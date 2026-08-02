# Training-run orchestration on Modal — folder-as-state with a KV lease

Status: v2 draft for review · Scope: single-tenant PyTorch training runs, one Modal Volume + one modal.Dict

---

## 0. Purpose and guarantees

Each training job runs as a Modal function ("worker", "attempt"), one attempt
per run folder, with resume, cancel, and a live dashboard. Guarantees:

- **G1 — Single writer.** At most one attempt writes a run folder's shared
  files at any instant. Enforced by mechanism (mutex + lease + commit gate),
  not convention.
- **G2 — Derived status.** Every durable fact lives in the run folder; run
  status is recomputed at the moment of decision from live authorities and is
  never stored anywhere.
- **G3 — Exact resume.** A resumed run continues from the latest durable
  checkpoint; it never trusts memory or uncommitted files, and never regresses
  committed history.
- **G4 — Bounded loss.** Crash, preemption, OOM, cancel, or fence eviction
  loses at most one checkpoint interval of work. Committed history survives
  every failure mode in §13.
- **G5 — Auditability.** Every attempt — including ones that abort without
  training — leaves its own log trail; every lease grant is mirrored into logs.

---

## 1. The three substrates and their division of labor

The design uses three storage/authority systems, each for exactly the property
it actually has. Mixing their roles is the root cause of every race this spec
defends against.

| substrate | property used | holds | never holds |
|---|---|---|---|
| **modal.Dict** | read-after-write consistency; single-key atomicity | the lease: *who may write, now* | history, status, anything durable |
| **Modal control plane** | ground truth on process liveness (`FunctionCall` state) | *who is alive, now* | — (queried, not written) |
| **modal.Volume** | durable bulk storage | *what happened*: config, checkpoints, metrics, logs | the lease, status, coordination state |

One-line model: **Dict = who may write; Modal = who is alive; Volume = what
happened.** The launcher (§5) is the only place all three are reconciled; the
commit gate (§7.2) is where the reconciliation is enforced.

### 1.1 modal.Dict semantics (load-bearing facts)

- Every `get`/`put`/`pop` is a server round-trip: reads are always fresh.
  There is no snapshot, no commit, no reload. This is the property a lease
  needs and the property the Volume cannot provide.
- **Single-key operations are atomic** server-side. In particular
  `put(key, value, skip_if_exists=True)` is an atomic put-if-absent returning
  `True` iff the key was added — a real per-key acquisition primitive (§5.2).
- **There is no compare-and-swap on value.** Any read-decide-write sequence
  (`get` → decide → `pop`/`put`) is NOT atomic and must be externally
  serialized (§5.1).
- **Entries expire after 7 days of inactivity** (no reads or writes). The Dict
  is therefore structurally incapable of being a system of record; it may hold
  only facts that must be *fresh*, not facts that must be *durable* (§4.4).
- A Dict call can also **fail to answer** (network blip, service event,
  timeout). Every consumer must therefore handle three outcomes — yes, no,
  unknown — never two (§6).

### 1.2 modal.Volume semantics (load-bearing facts)

- **Snapshot reads:** a container sees the volume as of its mount (or last
  `reload()`); others' commits are invisible until then.
- **Batched last-writer-wins commits:** local writes are invisible until
  `volume.commit()`; concurrent commits to one path resolve silently to the
  later commit. No conflict signal exists.
- **Corollary:** the Volume cannot host coordination state. A volume lock file
  is check-then-act over stale snapshots — two containers can both observe
  "no lock", both create it, both commit, and neither learns of the collision.
  This is *why* the lease lives in the Dict.

With the lease off the Volume, the worker never needs to read another
process's volume data mid-run — so the worker **never calls `reload()`** after
boot, and the entire class of reload-ordering constraints disappears from the
protocol.

---

## 2. Directory layout

```
runs/
  <run_id>/                            # e.g. 20260731T142211.483_alpha
    config.json                        # immutable — model, optimizer, schedule
    train.jsonl                        # append-only per-step metrics ledger
    checkpoints/
      ckpt_00001200.pt                 # write-once, monotonic step-numbered
      ckpt_00001200.pt.tmp             # transient; never survives a boundary
    logs/
      fc-abc123.../                    # one folder per attempt, named by call id
        attempt.log
      fc-def456.../
        attempt.log
```

There is deliberately **no lease file** anywhere in the tree. Ownership lives
solely in the Dict (§4); a second copy on the Volume would be a second source
of truth that lags takeovers and answers wrongly under exactly the failure
conditions that matter (§13/R15).

### 2.1 `run_id`

`{YYYYMMDDTHHMMSS.mmm}_{name}` — timestamp with milliseconds plus a human
label; the folder name is the immutable primary key. Millisecond resolution
plus atomic acquisition (§5.2) makes same-instant collisions a deterministic
loser-no-ops, never a race.

### 2.2 `config.json` — immutable run definition

Written exactly once by the launcher at folder creation; never edited; resumes
always run under the original (config arguments passed to a resume are
ignored). Contents:

```json
{
  "model":        { "...architecture / init source..." },
  "optimizer":    { "...type, lr, schedule..." },
  "total_steps":  120000,
  "ckpt_every":   500,
  "seed":         1234,
  "created_ts":   1785500531.483,
  "config_sha256": "hex digest of the canonical JSON of the fields above"
}
```

`config_sha256` is computed over the canonicalized body (sorted keys, no
whitespace) excluding itself. Every attempt recomputes it at startup and
refuses to train on mismatch — hand-edits become loud failures, not silent
divergence.

### 2.3 `train.jsonl` — append-only metrics ledger

One JSON object per line, one line per training step, written only by the
current lease holder. Rows may be appended to the container-local file at any
time; they become real only at a boundary commit (§7.3). Schema:

```json
{ "step": 1200, "loss": 2.4173, "lr": 3.0e-4, "ts": 1785500912.0,
  "call_id": "fc-abc123...", "ckpt": "ckpt_00001200.pt" }
```

- `step`: 1-based, strictly increasing within committed history.
- `call_id`: which attempt produced the row — with §9 logs, the full
  multi-resume history of a run is reconstructible attempt by attempt.
- `ckpt`: present only on boundary rows; a committed `ckpt` reference always
  names an existing checkpoint (§7.3 ordering).

**Reader contract:** tail = seek last ~8 KB, walk backward to the last
parseable line (a live append can tear the final line; skip silently). If
duplicate `step` values appear (possible only under §8.3), last occurrence
wins.

### 2.4 `checkpoints/`

Full training state (model + optimizer + RNG + step), named
`ckpt_{step:08d}.pt`. Write-once and monotonic: created atomically via
tmp+rename (§7.3), never rewritten, never renamed. Pruning keeps the newest
`KEEP_CKPTS = 2`, performed only by the lease holder in the same commit that
lands a newer checkpoint. Invariant for audit: any file under `checkpoints/`
was written by a confirmed owner — the commit gate (§7.2) blocks even the
tmp write for an unconfirmed one.

### 2.5 `logs/<call_id>/`

The private namespace of one attempt. Only that attempt writes here; the
folder name is its globally unique call id, so no two attempts can collide.
This is what makes it safe for even an *evicted* attempt to flush its final
log without holding the lease. G1's "shared files" excludes these folders by
design.

---

## 3. Ownership and write discipline

| path / key | writer | discipline |
|---|---|---|
| `runs/<run_id>/` (name) | launcher | immutable primary key |
| `config.json` | launcher, at creation | write-once, self-hashed |
| Dict `lease:<run_id>` | **launcher only** | §4; grant/takeover only, atomic or mutex-serialized |
| `train.jsonl` | current lease holder | append-only, boundary-committed |
| `checkpoints/ckpt_*.pt` | current lease holder | write-once tmp+rename; holder-pruned |
| `logs/<call_id>/**` | attempt `<call_id>` only | append-only, private namespace |

The worker **never writes the Dict**. An attempt cannot promote, renew, or
release itself; eviction and grant are exclusively launcher acts. Two facts
deliberately have no storage anywhere: run *status* (recomputed, §10) and any
done/failed marker (derivable; a stored copy could disagree with the
derivation).

---

## 4. The lease

### 4.1 Shape

One Dict key per run:

```
"lease:<run_id>"  →  { "call_id": "fc-...", "granted_ts": 1785500531.4, "attempt": 3 }
```

`call_id` is the lease. The other fields are **immutable at grant time** and
therefore safe to store (they cannot go stale): `granted_ts` for debugging
lease age, `attempt` for resume count without scanning log folders.

### 4.2 What the value must never contain: status

Status changes without anyone writing — OOM, preemption, timeout, console
cancel all transition a run with no process available to update a stored
field. The victim of the most important transition (crash) is by definition
unable to record it. A stored status is therefore stale from the moment of
writing, in the dangerous direction, and is an attractive nuisance: its only
modes are ignored (dead weight) or trusted (a bug). Status is recomputed at
decision time from Modal + folder (§10); the Dict stores only the pointer
(`call_id`) that makes the recomputation possible.

### 4.3 Registered ≠ alive

The Dict answers "who was last *granted*" — never "who is *running*". A lease
pointing at a crashed attempt is not an anomaly; it is the normal state of
every failed run awaiting resume (nothing ever cleans keys — no unlock
exists). Every ownership decision therefore consults **two oracles in
sequence**: Dict for the grant (*who*), then Modal `status()` on that call id
(*alive?*). Revocation only ever follows a fresh `status()` call — never a
clock (§11).

### 4.4 Expiry and durability

Dict entries expire after 7 idle days. Consequences, all benign by
construction: active runs' entries are touched by every fence check and
dashboard scan, so they cannot expire while anyone is running or watching; a
run nobody launches *or looks at* for a week loses its key and reads as
`orphaned` (§10) — the correct answer for it anyway; and the Dict can lose
*everything* without losing any training state, because the Volume is the
sole system of record. For audit durability, the launcher mirrors every grant
(`ts, run_id, call_id, attempt`) into the new attempt's log folder — an
informational record, **never consulted by any fence or launch decision**
(§13/R15 explains why a consulted copy would be unsafe).

---

## 5. The launcher

All resuming — and takeover of any existing folder — flows through one Modal
function, `launch`, with `max_containers=1` and single-input concurrency:
Modal's scheduler executes every such request sequentially in one container.
This is the mutex, and it is the **only** true mutual-exclusion primitive
available (§1.2 corollary rules out the Volume; §1.1 rules out Dict
compositions).

### 5.1 What needs the mutex and what doesn't

Single-key Dict operations are atomic; read-decide-write sequences are not.

| operation | atomic via Dict alone? | needs mutex? |
|---|---|---|
| fresh grant — `put(skip_if_exists=True)` | yes | no |
| liveness gate — `get` lease, then Modal `status()` | no (two calls, two systems) | yes |
| takeover — `pop` then `put` | no | yes |
| worker fence read — single `get` + local compare | yes | no |

### 5.2 Decision table (`launch(run_id?, name?, config?)`)

Executed top to bottom; first match wins.

| # | condition | action |
|---|---|---|
| L1 | no `run_id` given | mint one (§2.1), create folder tree + `config.json`, spawn worker, `put("lease:<id>", grant, skip_if_exists=True)` → must return `True`; return `{spawned}` |
| L2 | fresh mint collides (put returns `False`) | should be impossible with ms timestamps under the mutex; if it fires, another launcher path granted first (§5.4) — abort this spawn path, return the existing grant as "already running/owned". Backstop, not a normal path |
| L3 | folder exists, `tail(train.jsonl).step ≥ total_steps` | no-op `{"reason": "already complete"}` — decided from the **Volume alone**; Dict and Modal are never consulted for a complete run |
| L4 | folder exists, incomplete, lease present, `status(call_id) == running` | no-op `{"reason": "already running", call_id}` |
| L5 | folder exists, incomplete, lease present, `status(call_id) != running` (done / failed / crashed / timed_out / expired) | **takeover**: spawn worker, `pop` + `put(new grant)`, mirror grant to logs, return `{spawned}` |
| L6 | folder exists, incomplete, **no lease key** | no owner — identical action to L5, minus the `pop`. Absence does not imply a hidden runner: only the launcher writes keys and always before returning, so a missing key means no legitimate grant; any illegitimate bypass worker self-evicts via its own fence (§6, unknown branch) without the launcher's help |
| L7 | folder exists, `config.json` missing/unparseable | refuse; folder is quarantined `orphaned` — never guessed at |

Ordering within a grant: **spawn first, then write the lease** (the call id
does not exist before spawn). The write is immediate and globally visible on
return — no commit latency exists for the worker to wait out, so the fence-in
retry (§6.1) covers only this microscopic spawn-before-put window plus
transport time.

### 5.3 Crash window

If the launcher dies between spawn and `put`, the spawned worker's fence-in
never sees its own id → exhausts retries → aborts with zero shared writes; the
folder remains resumable. No durable damage (§13/R7).

### 5.4 Scope of the mutex — deployed app only

`max_containers=1` serializes one *app instance*. An ephemeral instance (a
`modal run` of the module, `with app.run()`) has its own pool and does **not**
serialize against the deployed launcher. Every caller — notebook, CLI,
dashboard, cron — must resolve the deployed `launch` via
`Function.from_name(...)`. If the rule is violated, two independent
protections remain: put-if-absent makes concurrent *fresh* grants pick exactly
one winner (L2 is that backstop firing), and the fence bounds any takeover
race to one dropped interval — degraded, never corrupted.

---

## 6. The fence: three branches, not two

The worker knows its `run_id` and its own `call_id`
(`modal.current_function_call_id()`). Its ownership check is a single fresh
Dict read — but the read has **three** outcomes, and conflating any two of
them is a design error in a specific, known direction:

| outcome | meaning | authority | action |
|---|---|---|---|
| **MATCH** — value's `call_id` == mine | I am the owner, now | Dict, fresh | proceed |
| **EXPLICIT MISMATCH** — key holds another id | someone else was deliberately granted this run | Dict, fresh — a positive fact | abort **immediately**: no retry (retrying a "no" is hoping the truth changes), flush own log, write nothing shared |
| **UNKNOWN** — key absent, or the call raised/timed out | ownership currently unprovable | none — silence, not denial | bounded retry (§6.2); abort only after exhaustion |

Absence routes to UNKNOWN, never to MISMATCH: an absent key carries no
Modal-server assertion the way a concrete other id does. Its causes — Dict
outage, 7-day expiry of a paused run, a never-granted bypass — none of them
mean "another worker owns this". Treating silence as denial makes a
correlated Dict blip fell the entire fleet (§13/R12); treating it as
confirmation would let a partitioned zombie write (§13/R11). Bounded patience
is the only policy whose errors are conservative in both directions.

### 6.1 Fence-in (startup)

Before **any** shared write — before creating even its own log folder is
required, though the log folder is private and technically safe:

1. Verify `config_sha256` (§2.2); refuse on mismatch.
2. Fence check per the table. UNKNOWN retries here cover the
   spawn-before-put window (§5.2) plus transients; a few seconds of budget
   suffices (`FENCE_IN_RETRIES × backoff`).
3. On MATCH: locate resume point (§8), begin training.
4. On abort: create `logs/<my_id>/`, flush buffered startup log +
   reason, `volume.commit()` (private namespace — safe leaseless), exit.

### 6.2 The commit gate (every checkpoint boundary) — the load-bearing check

The single point where the Dict protects the Volume. **No confirmed MATCH ⇒
no shared-namespace activity at all**: not the ledger append, not the commit,
and not the checkpoint `.tmp` write either (§2.4's audit invariant).

UNKNOWN policy at the gate — fail-closed with bounded patience:

- The dangerous act is *committing*, not *computing*: everything since the
  last boundary is in memory or uncommitted local files, discardable by
  design (G4). So block the boundary, retry with backoff + jitter
  (`GATE_RETRIES = N`, e.g. 1s → 2s → … capped), logging each attempt.
- While blocked, compute at most ~one further interval ahead, then pause the
  GPU loop too — training past that manufactures work that may be discarded,
  inflating loss beyond G4's bound.
- Patience is safe against legitimate takeover: the launcher's gate (L4)
  consults **Modal**, and a Dict-partitioned worker is still `running` per
  Modal — so its lease cannot legitimately move during the blip. When
  connectivity returns, MATCH resumes the run in place, zero loss. Only a
  forced manual override can reassign meanwhile, and the no-commit rule
  covers exactly that.
- After exhaustion: exit via the eviction path — flush own log, nothing
  shared, return. Uncommitted local ledger rows die with the container;
  nothing to clean.
- Economics of N: exiting costs one interval + restart (boot + checkpoint
  load); waiting costs idle GPU. Dict failures are fleet-correlated, so
  N×backoff is also the knob between "mass idle" and "synchronized resume
  herd" — a herd the idempotent launcher absorbs safely, one request at a
  time.

W = 0 (abort on first UNKNOWN) is *correct* but twitchy: it converts routine
transients — a run making per-minute checks over 20 h issues ~1,200 RPCs;
occasional failure is expected, not exceptional — into interval-loss +
restart, fleet-wide, simultaneously. Recommended default: at least one retry
(sleep ~5 s, ask once more); it absorbs the overwhelming majority of real
transients at near-zero complexity.

### 6.3 Advisory checks (optional)

Between boundaries the worker may `get` the lease per step or per K steps for
early eviction detection — supersession latency drops from `ckpt_every` to
~one step. These carry **zero safety weight**: errors are ignored, only an
EXPLICIT MISMATCH acts (immediate exit). Only §6.2 gates writes.

---

## 7. The worker

### 7.1 Loop shape

Boot → fence-in (§6.1) → resume point (§8) → loop: train one step; append the
metric row to the container-local `train.jsonl`; log freely to
`logs/<my_id>/`. Local writes are invisible and non-durable until committed —
and if the attempt aborts, they die with the container, which is exactly the
cleanup required.

### 7.2 Top-level exception wrap (mandatory)

The worker body is wrapped: any escaped `Exception` is logged to the attempt
folder and re-raised as `RuntimeError(...) from e`. Reason: `status()`
classifies a re-raised builtin `TimeoutError` as `running` (§10.1) — an
uncaught socket timeout would pin a dead run at `in_progress` forever. The
wrap guarantees every escape classifies as `crashed` (§13/R14).

### 7.3 Boundary protocol

At every `ckpt_every` steps and at `total_steps`, in order:

1. **Commit gate** (§6.2). MATCH or exit — nothing below runs unconfirmed.
2. **Checkpoint atomically:** serialize to `ckpt_{step:08d}.pt.tmp`, fsync,
   `os.rename` to final. (Atomic on the container FS; the volume commit only
   ever uploads the final name — no reader can observe a torn checkpoint.)
3. **Boundary ledger row:** the row for this step carries
   `"ckpt": "ckpt_{step:08d}.pt"`. (2-before-3 is what makes every committed
   `ckpt` reference resolve.)
4. **Prune:** delete checkpoints beyond the newest `KEEP_CKPTS`.
5. **`volume.commit()`** — the **commit unit**: one checkpoint, its metric
   rows, its pruning, and the interval's logs land together. Durable state is
   always a complete boundary; a torn interval is unrepresentable.

No `reload()` appears anywhere: the worker reads no one else's volume data
after boot (§1.2). Completion is not an event — when the final boundary
commits, §10 derives `completed` from the tail; the worker simply returns.

### 7.4 Cancellation

Cancel = `FunctionCall.cancel()` on the lease's call id (dashboard/CLI reads
the Dict to find it). Modal's signal interrupts a synchronous loop; the call
transitions to a failed state within seconds. Work since the last boundary is
lost — G4 applied deliberately. No cooperative stop-file exists.

---

## 8. Resume discipline

### 8.1 Invariant

Metrics commit only alongside their checkpoint (§7.3), so committed state
always satisfies **`tail(train.jsonl).step` == newest checkpoint's step**.
Resume: load newest checkpoint, continue at `step + 1`. No duplicates, no
gaps — under the invariant.

### 8.2 What resume trusts

Checkpoint **files**, not the ledger: the resume point is the newest
`ckpt_*.pt` that deserializes successfully, walking backward on failure (a
corrupt newest checkpoint costs one more interval; `KEEP_CKPTS = 2` exists
for exactly this). Fresh folder, no checkpoints: initialize from
`config.json` at step 0. The ledger is for display and audit; the checkpoint
is the training state.

### 8.3 Defensive rule when the invariant is violated

Partial commit or tampering could leave `tail.step ≠ ckpt.step`:

- `tail.step > ckpt.step` → trust the checkpoint; resume at `ckpt.step + 1`;
  re-run steps append duplicate `step` rows — permitted; readers take last
  occurrence (§2.3).
- `tail.step < ckpt.step` → trust the checkpoint; accept the metrics gap; log
  loudly.

Reconciliation is always one-directional: checkpoints are ground truth for
state, the ledger for progress display.

---

## 9. Logs and audit

Every attempt leaves `logs/<call_id>/attempt.log`: startup, config-hash
check, fence-in outcome, resume point, per-boundary summaries, every UNKNOWN
retry (so a blocked gate is visible — "fence indeterminate, retry 7, 210 s" —
not a silent hang), grant mirror (if this attempt's grant, written by the
launcher), and exit reason. With `train.jsonl.call_id`, the complete
multi-resume history is reconstructible.

Pruning of old attempt folders: launcher only, inside the mutex, only beyond
the newest `KEEP_ATTEMPT_LOGS`, never the folder of the current lease's call
id. Attempts never delete anything.

---

## 10. Status derivation

Recomputed per dashboard request; stored nowhere. Two orthogonal signals:

- **Progress** (Volume): `tail(train.jsonl).step` vs `total_steps`;
  `complete` short-circuits everything else.
- **Liveness** (Dict → Modal): `get` the lease; feed `call_id` into the
  pinned `status()`; consulted **only when not complete** — call results are
  GC'd by Modal eventually, and a finished run must read finished forever
  from the folder alone.

| progress | liveness | status |
|---|---|---|
| complete | *(never consulted)* | **completed** |
| partial | `running` | **in_progress** |
| partial | `done` / `failed` / `crashed` / `timed_out` | **failed** |
| partial | `expired`, or **no lease key** | **orphaned** |
| no rows yet | `running` | **starting** |
| no rows yet | `done` / `failed` / `crashed` / `timed_out` | **failed** |
| no rows yet | `expired`, or no lease key | **orphaned** |

- `partial × done` (returned with steps remaining) is an abnormal end —
  the worker only returns at completion or eviction; eviction is visible in
  its log. Classified failed.
- Cancel surfaces as `failed`; no special state.
- `starting` = lease granted, call running, no committed boundary; natural
  duration ≈ boot + one `ckpt_every`. Far beyond that ⇒ wedged in init;
  cancellable.
- `orphaned` is the honest cannot-know bucket (expired outcome, expired key,
  out-of-band folder). Always resumable — the launcher re-verifies inside the
  mutex regardless of what the dashboard believed.
- Dashboard scan is async: lease `get`s and `status()` calls gathered
  concurrently over all not-complete runs — one round-trip of latency, not
  one per run. Side effect: scan reads keep active leases perpetually fresh
  against the 7-day expiry (§4.4).
- Actions by status: `in_progress`, `starting` → Cancel; `failed`,
  `orphaned` → Resume (through the deployed `launch`, never a direct worker
  spawn); `completed` → nothing. Poll responses: `Cache-Control: no-store`
  and client-side cache bypass — a heuristically cached poll silently
  freezes the dashboard.

### 10.1 Liveness mapping (pinned)

One async `status(call)` is the single source of truth for
`FunctionCall.get(timeout=0)` → state, verified against the pinned modal
version and its test suite; a second (sync) implementation is prohibited —
it will drift. States: `running` (builtin `TimeoutError`), `done`, `expired`
(`OutputExpiredError`), `timed_out` (`FunctionTimeoutError`), `failed` (other
modal `Error`, incl. cancel — `detail` carries the exception), `crashed` (any
other re-raised exception). `unknown` is produced at the wrapper level when
there is no call id to ask about. Version fragility: the mapping's catch
order assumes modal's `TimeoutError` is not the builtin; any modal upgrade
must re-run the pinning test.

---

## 11. Lease lifecycle — deliberately no expiry mechanism

There is no TTL, no renewal, no heartbeat on the lease, because the standard
renewal pattern is structurally unavailable: only the launcher writes the
Dict, so "time since last write" measures time since *grant*, not liveness —
a healthy worker 10 h into a 20 h run and a worker that crashed 10 h ago are
byte-identical to any clock. A timer would revoke both, and revoking the
first is precisely the false positive that produces two writers.

"Expiry" is instead a **query, evaluated lazily**: a lease is eligible for
reassignment the instant `status(call_id) != running`, discovered at the next
`launch(run_id=X)` — the resume-time liveness gate *is* the expiry check,
consulted fresh at the moment of decision. The safety property, stated once:
**revocation only ever follows a fresh `status()` call.** Never a clock.

Legitimate time-based components are *watchdogs* that trigger the query, not
expiries that bypass it:

- **Stalled watchdog:** `tail.ts` older than `N × ckpt_every` ⇒ surface
  `stalled` on the dashboard; prompts a human or cron to call `launch`, which
  performs the real gated check.
- **Reaper cron (optional):** periodically resume `failed`/`orphaned` runs
  (idempotent — L3/L4 make blind retries safe) and/or `.cancel()` wedged
  `running` calls to stop paying for them. Still gated by `status()` at
  decision time.

The Dict's own 7-day storage GC is cleanup of unread keys, not a correctness
mechanism (§4.4).

---

## 12. API surface

| operation | call | semantics |
|---|---|---|
| new run | `launch(name=, config=)` | L1: mint, create, spawn, atomic grant |
| resume | `launch(run_id=)` | L3–L6 inside the mutex; grant iff not complete and not running |
| idempotent retry | `launch(run_id=)` on running/complete | no-op with stated reason — blind retries always safe |
| cancel | Dict `get` → `.cancel()` | interrupt now; ≤ 1 interval lost |
| status | scan (§10) | read-only; no side effects |

All of it resolves to the **deployed** app (§5.4). There is no other write
path to either the Dict or the Volume's shared files.

---

## 13. Race and failure inventory

| # | scenario | resolution |
|---|---|---|
| R1 | two launches race a **fresh** `run_id` | atomic `put(skip_if_exists=True)` — exactly one `True`, loser no-ops (no mutex needed) |
| R2 | two launches race a **resume/takeover** | serialized launcher: the read-decide-write sequence runs alone (§5.1) |
| R3 | ephemeral app instance forks the mutex | rule: deployed `launch` only (§5.4); backstops: put-if-absent for fresh grants, fence bounds takeover races to one dropped interval |
| R4 | worker boots before the launcher's `put` lands | fence-in UNKNOWN retries (§6.1) — window is spawn-to-put, microscopic; no commit latency exists |
| R5 | direct worker spawn bypasses the launcher | never granted ⇒ fence-in UNKNOWN → exhaust → abort, zero shared writes; self-evicting without launcher help (§5.2/L6) |
| R6 | zombie: superseded/partitioned worker keeps computing | commit gate: EXPLICIT MISMATCH before any shared write ⇒ interval dropped, tail never regresses (§6.2, §13-note) |
| R7 | launcher dies between spawn and grant | worker fence-in exhausts → aborts clean; folder resumable (§5.3) |
| R8 | Modal-level retry re-runs the worker input | same call id ⇒ fence MATCH ⇒ legitimate continuation from last boundary |
| R9 | ledger row references a missing checkpoint | ordering: ckpt rename → boundary row → single commit (§7.3) |
| R10 | newest checkpoint corrupt | resume walks back; `KEEP_CKPTS ≥ 2` (§8.2) |
| R11 | Dict unreachable, worker assumes ownership (fail-open) | prohibited: UNKNOWN never confirms; no commit without MATCH (§6.2) |
| R12 | Dict unreachable, worker aborts instantly (hair-trigger) — correlated fleet suicide | bounded-patience UNKNOWN policy; N×backoff knob; idempotent launcher absorbs the resume herd (§6.2) |
| R13 | absent key treated as eviction | absence routes to UNKNOWN, not MISMATCH — silence is not denial (§6) |
| R14 | worker code lets builtin `TimeoutError` escape ⇒ classified `running` forever | mandatory top-level wrap ⇒ `crashed` (§7.2) |
| R15 | lease snapshot/copy on the Volume consulted as fallback | prohibited: a snapshot always lags takeovers and tells the *old owner* it still owns — confident and wrong exactly under failure. Volume copies are audit-only, never consulted (§4.4) |
| R16 | stored status field goes stale | prohibited: no status is ever stored; Dict value holds only grant-time-immutable fields (§4.2) |
| R17 | timer-based lease expiry revokes a healthy long run | prohibited: no TTL; revocation only follows fresh `status()` (§11) |
| R18 | torn final `train.jsonl` line under a live reader | tail walks back to last parseable line (§2.3) |
| R19 | same-instant `run_id` collision | ms timestamps + R1's atomic acquisition |
| R20 | config hand-edited mid-run | `config_sha256` mismatch ⇒ attempt refuses (§2.2) |
| R21 | log pruning deletes live data | launcher-only, mutex-only, current lease's folder excluded (§9) |
| R22 | stale dashboard | per-request fresh Dict reads + Volume scan; `no-store` on polls (§10) |
| R23 | worker wedged but `running` (hung dataloader) | no matrix signal exists by design; stalled watchdog surfaces it; human/cron cancels then resumes (§11) |

Note on R6's residual window: between a takeover grant (L5) and the old
worker's next gate check, old and new attempts may *compute* concurrently —
but never *commit* concurrently: the old attempt's next gate reads the new
id and aborts before writing. Coexistence of processes is permitted;
coexistence of writers is not — and only the latter matters for G1.

---

## 14. Operational parameters

| knob | default | governs |
|---|---|---|
| `ckpt_every` | ≈30–60 s of steps | G4's loss bound; dashboard progress granularity; commit rate |
| `KEEP_CKPTS` | 2 | corruption survival depth vs space (§8.2) |
| `FENCE_IN_RETRIES × backoff` | ~3 × 2 s | spawn-to-put window + transients (§6.1) |
| `GATE_RETRIES (N) × backoff` | ≥1 retry; N ≈ 5–10 min total for expensive GPUs | UNKNOWN patience: interval-loss+restart cost vs idle-GPU cost; fleet herd synchronization (§6.2) |
| `KEEP_ATTEMPT_LOGS` | 10 | audit depth vs clutter |
| worker `timeout=` | > longest expected run | beyond it: `timed_out` → failed → resumable |
| worker retries | 0 explicit | Modal retries are safe (R8) but redundant with resume; off for legibility |
| dashboard poll | ~5 s | lease/liveness freshness is now instant; only metrics lag by `ckpt_every` |
| stalled threshold | `N × ckpt_every` | R23 detection latency vs false alarms |

---

## 15. Non-goals and known limits

- **One writer per folder is the model.** Distributed data parallelism must
  present itself as one attempt; intra-run multi-writer is out of scope.
- **The mutex is one deployed function.** §5.4 is discipline enforced by
  fencing and put-if-absent, not by construction across app instances.
- **Metrics visibility lags by one interval.** Between boundaries, steps live
  in container memory/local files; the dashboard shows the last committed
  boundary. Ownership/liveness, by contrast, is now real-time. Streaming
  metrics out-of-band (W&B/MLflow) is the sanctioned escape hatch; the folder
  ledger remains the correctness record.
- **`orphaned` is irreducible.** When Modal has expired an outcome and the
  Dict has no key, no oracle exists; the system reports honestly and offers
  resume.
- **New dependency:** the Dict service sits on the commit path. Its failure
  mode is availability (blocked gates, bounded aborts), never integrity —
  §6.2's policy is what makes that claim true.
- Assumes the documented semantics in §1.1–1.2 under the pinned modal
  version; re-verify both on upgrade (the liveness test suite covers §10.1;
  Dict atomicity of `skip_if_exists` should get its own pin test).
