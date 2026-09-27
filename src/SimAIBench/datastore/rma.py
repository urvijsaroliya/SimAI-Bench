"""
One-sided MPI ring-queue transport — a bounded queue inside the job's memory.

This is the third transport mechanism of the taxonomy in
`docs/coupling_model_theory_and_experiments_v3.md` §15.2: not a store (no
external process, no namespace) and not point-to-point (no producer that must
name a receiver for every message), but a **queue** — fixed-capacity ring
buffers living in pre-allocated MPI RMA windows, addressed by ticket.
`datastore/mpi.py` is the direct path this is compared against; everything that
file says about a single launch and a shared communicator applies here too, and
more strictly (see "Collective by construction").

The design is the RDQ of v3 §15 reduced to what two ranks were measured doing
on this platform (build plan step 2c, job 4242310, `EXPERIMENTS.fritz.md` M33).
It is deliberately *less* than §15.1 describes: one ring per (producer,
consumer, class) instead of three rings per instance, the descriptor folded into
the slot header instead of a ring of its own, and polling atomics instead of the
MCS-derived notification protocol. Each of those is named where it is made.


WHAT THE 2c GATE FORCED (job 4242310; M33)
------------------------------------------
Two measurements changed the shape of this backend, and both are about *which
component serves a window*, not about RMA in general:

1. **Rings on the world communicator get `osc/ucx`, and on this build UCX RMA
   has no shared-memory lane.** Open MPI serves a window with `osc/sm` only when
   every rank of that window's communicator shares a node. A coupled job spans
   nodes, so its world-communicator windows get `osc/ucx`, where every intra-node
   put, get and atomic goes over `rc_mlx5` — the HCA looped back. Per atomic
   within a node that is **0.0120-0.0128 ms against `osc/sm`'s 0.0038-0.0047**
   (3.2x on `Fetch_and_op`, 2.7x on `Compare_and_swap`), and the per-message
   control path is **0.0248 ms against 0.0085**. Window creation costs
   **0.107 s against 0.040** for one 1.28 MB slot.
   `ring_comm` is the knob for this and it defaults to `"world"` — the plain
   world communicator, i.e. the expensive column, because that is the
   communicator a coupled job's windows actually live on.
   `ring_comm="node"` puts the rings on `Split_type(COMM_TYPE_SHARED)` and
   recovers most of it, but it can only carry pairs whose two ends share a node,
   so it is a control arm for colocated patterns and not a default. The tiered
   arrangement (node-local rings under a cross-node ring) is the author's open
   option and is **not** implemented; `ring_comm="tier"` raises rather than
   pretending.

2. **A producer cannot overlap a push within a node; a consumer can overlap a
   pop.** Cross-node `MPI_Put` returns in 2.8 us and the bytes move in the flush
   (0.1073 of 0.1101 ms). Within a node the put is **94% copy-in inside the
   call** (0.2440 of 0.2603 ms) under `osc/ucx` and *is* the memcpy under
   `osc/sm`. The direction that overlaps within a node is the get: 0.0302 ms of
   call, 0.2010 ms of flush.
   So the default is a **producer-owned ring popped by the consumer with `Get`**
   (`ring_owner="producer"`): the producer writes its payload with a plain local
   store into its own window memory — no RMA on the push path at all — and the
   consumer pays a remote `Get` it can overlap with local work.
   `ring_owner="consumer"` is the other arm (producer `Put`, consumer local
   load), which is v3 §15.8 (i)'s zero-copy pop; the gate never timed a
   consumer-local window, so that arm is run (d)'s measurement, not a claim.

Budget from M33's `nosm` column within a node and its `default` column across
nodes, never from the within-node default pass. Per message at 1.28 MB the
protocol below issues, on the consumer side, one HEAD atomic + one 128 B header
`Get` + one payload `Get` + one release CAS — which from M33's per-operation
numbers is ~0.29 ms within a node (`osc/ucx`) and ~0.19 ms across nodes. That is
arithmetic over isolated operations, exactly the kind F13 warns does not survive
coupling; it is a sizing input, not a prediction of a coupled run.


THE PROTOCOL, IN FULL
---------------------
One ring per (producer, consumer, class). A ring is **single-producer,
single-consumer**, which is what lets it run on two monotone uint64 counters
with no per-slot state machine:

    HEAD   records committed by the producer            (producer writes, consumer reads)
    TAIL   records released by the consumer             (consumer writes, producer reads)

    ring is non-empty for the consumer   iff  HEAD > TAIL
    producer may commit record HEAD      iff  HEAD - TAIL < capacity

`capacity` is M_q, v3 §15.5's explicit configuration constant. The bound holds at
**every payload size**, which is the one thing the direct path cannot give: there
the send window binds only above the rendezvous switch, 7 194 B within a node and
4 111 B across it (F20 revised, job 4241434). That is the discriminating claim of
step 11 and it is structural here, not measured.

HEAD, TAIL and CLOSED live in the ring **owner's** control window and are touched
only with accumulate-class operations, never with a load or a store, even by the
owner — MPI leaves a load racing an accumulate undefined in both memory models,
and this is the state the whole queue rests on:

    commit  (producer)   Fetch_and_op(SUM,     1)        on HEAD
    read    (consumer)   Fetch_and_op(NO_OP,   0)        on HEAD
    release (consumer)   Compare_and_swap(t+1, t)        on TAIL
    read    (producer)   Fetch_and_op(NO_OP,   0)        on TAIL
    close   (producer)   Fetch_and_op(REPLACE, HEAD+1)   on CLOSED

The release is a compare-and-swap rather than the cheaper fetch-and-add on
purpose: in a single-consumer ring the two are equivalent, so the CAS's return
value is a free assertion that the ring really had one consumer. A mismatch
increments `cas_mismatch` and raises rather than silently corrupting the queue.
`release_op="faa"` drops the assertion for a cell that wants to price it.

**Both sides cache the other's counter**, so the steady-state atomic count per
message is one on each side, not four: the producer re-reads TAIL only when its
cached value says the ring is full, and the consumer re-reads HEAD only when its
cached value says the ring is empty. `head_atomics` and `tail_atomics` against
`writes` / `pops` is how a run shows that held.

A slot is `header + key + payload`, fixed size, 64-byte aligned:

    [0 : 64)                 header, 8 x uint64
    [64 : 64 + key_max)      the key, NUL-padded  (exact bytes, not a hash)
    [POFF : POFF + nbytes)   the payload          POFF = align64(64 + key_max)

    header word 0  MAGIC        0x52445131; a slot never written reads 0
                1  ticket       record index in this ring
                2  keylen       bytes of key that are the key
                3  nbytes       payload bytes actually written
                4  producer     producer's world rank
                5  kind         0 raw buffer, 1 pickled object
                6  ticket_echo  == word 1; a lapped or torn slot fails here
                7  reserved     0

The key travels **verbatim**, not hashed. `mpi.py` has to hash keys into a tag
and raise on a collision (`mpi.py:_tag`); a ring has somewhere to put the key,
so that whole hazard class is absent here.

**How a consumer knows a slot is full:** `HEAD > TAIL`, one atomic — never by
inspecting the slot's bytes, which would cost a payload `Get` per poll. The
header's MAGIC and `ticket_echo` are then checked as instrumentation, not as the
readiness test.

**Push** (`stage_write`): refresh TAIL if the cached value says full; block or
drop if it still is; write header, key and payload into the slot; commit. Under
`ring_owner="producer"` "write" is a local store followed by `Win.Sync()` (the
unified-model memory barrier that publishes the private copy into the public one)
and "commit" is a local atomic on HEAD. Under `ring_owner="consumer"` it is one
`Put` of `POFF + nbytes` bytes, a `Flush`, then a remote atomic on HEAD. The
record becomes visible at the commit and not before.

**Pop** (`poll_staged_data` / `stage_read`): read HEAD; peek the next record's
header and key (a POFF-byte `Get`, or a local load when the consumer owns the
ring); if the key matches what the caller asked for, `Get` the payload and
release; if it does not, pop the record into a read-ahead dict and look again.

**Key bridging, and why the read-ahead dict is not optional.** The queue is FIFO
by ticket and `stage_read(key)` names a record. In the driver's normal path they
agree — the trainer asks for `input_<src>_<idx>` with `idx` increasing, in exactly
the order the sim wrote them. They disagree in the driver's teardown, which polls
for `simdone_<src>` while unread `input` records are still queued: on the direct
path `simdone` is a separate tag and can overtake, in a ring it cannot. So a poll
whose key is not the head record **drains** records into `_readahead` until it
finds the key, and `stage_read` serves from that dict first. That drain is v3
§15.5's O(versions) cost made explicit and `readahead_pops` counts it. It is
bounded twice: `poll_max_pops` per call, and `readahead_max` entries in total —
whose default is *derived* (2 x this rank's total consumed capacity, at least 16)
rather than a constant, because a producer cannot run more than `capacity` records
ahead, so that is the most that can be in front of a key in the first place. A
parked record is a whole payload, so this bound is a memory bound.

**The peek is cached, so poll + read costs one header Get, not two.** `_peek_header`
caches the header and key against the ticket and `_release` invalidates it, so the
driver's usual `poll(k)` then `stage_read(k)` issues the small `Get` once. That is
why `peek="on"` is the default and costs nothing over `peek="off"` on the path that
matters; `peek="off"` saves the header `Get` only for a poll that hits and is never
followed by a read of that key, and gives up telling the truth to do it.

**Full ring.** Lossless keys block; keys matching `drop_prefixes` drop, exactly
as `mpi.py` does and for the same reason — a keep-last-1 stream that blocks on a
bounded transport deadlocks a closed loop, because the blocked producer stops
consuming the reverse direction. The *new* record is the one discarded (newest
dropped, not oldest) and a publish is dropped for **every** consumer at once, so
`dropped` counts publishes and `coupled.py`'s `_dropped()` reads the same number
it reads off `mpi.py` (M27). v3 §15.5 is still right that the ring has no native
keep-last-1: a consumer that wants only the newest version must drain to it, and
`readahead_pops` is what that costs.
**Dropping the NEW record puts a HOLE in the index sequence, so a consumer that
polls for a CONTIGUOUS index cannot drain to the newest version — it can only
drain past it.** `coupled.py`'s barrier counts `input_<src>_<base+got>`, so a
dropped index is one `poll_staged_data` can never match: the scan pops and parks
every LATER record instead, `next_ticket` never reaches the hole, and `_park`
raises once `readahead_max` is reached. That is why `coupled.py` REFUSES
`--forward-policy keep-last` with `--backend rma` rather than serving it. A ring
that dropped the OLDEST record instead would keep the sequence contiguous and
make the combination servable; that is a chunk-1 change, and it is not made here.
Lossless and keep-last keys never share a ring's capacity — there are **two
classes of ring per pair** when `drop_prefixes` is non-empty, for the reason
`mpi.py` keeps two pending lists: occupancy from one class must not charge the
other's window.

**Nothing spins without a bound.** Every wait has a deadline and a counter:
`_block_for_slot` (`full_timeout_s`, `blocked_writes`, `full_timeouts`),
`stage_read` (the caller's `timeout`, `read_timeouts`), the release CAS
(`cas_max_retries`, `cas_giveups`), `clean` (`clean_drain_s`,
`clean_unreleased`). A producer that closes its ring publishes CLOSED, so a
consumer waiting for a record that will never come fails by name instead of
waiting out a 30-second timeout.

**No buffer-ownership hazard.** `MPI_Put` + `Flush` completes the origin buffer
before the call returns, and the producer-owned arm has no origin buffer at all,
so M12's "rotate max_outstanding + 2 buffers" discipline and M28's
rewrite-in-flight corruption cannot arise on this path. That is a semantic
difference from the direct path, not an implementation detail.


COLLECTIVE BY CONSTRUCTION — READ THIS BEFORE USING IT
------------------------------------------------------
`MPI_Win_allocate` is collective over the ring communicator, so **constructing
this backend is collective**, and so is freeing its windows. Three consequences:

- Every rank of the communicator must construct a `DataStoreRMA`, and the
  geometry exchange is a mandatory `allgather` — not the opt-in `verify_peers`
  of `mpi.py`. A consumer sizes its `Get` from its own config, so a
  producer/consumer mismatch corrupts records rather than raising; since setup is
  collective anyway, checking it costs nothing that was not already required.
- A process that builds a **second** store (as `coupled.py` does on the sim
  ranks in the closed loop, `infer = Simulation(...)`) must not allocate a second
  pair of windows, or it deadlocks against the ranks that built one. So
  `windows="singleton"` is the default: the windows are a per-process singleton,
  a second store with the same geometry signature joins them, and one with a
  *different* signature raises instead of deadlocking. A harness whose every rank
  builds the same sequence of differently-shaped stores (the two-rank unit test)
  sets `windows="per-config"` and gets one window pair per signature.
- `free_windows` defaults to **False**: `MPI_Win_free` is collective, and the
  driver's closed loop leaves one store per sim rank uncleaned, so a refcount
  would not reach zero symmetrically. `clean()` therefore ends the epoch
  (`Unlock_all`, which is *not* collective) on the first call in the process and
  leaves the windows to `MPI_Finalize`. Set `free_windows=True` only where every
  rank constructs and cleans exactly once. **After any `clean()` no store sharing
  those windows may be used again.**

`device="cpu"` only in v1. Build plan OQ3 (CUDA-aware `Put`) is unanswered rather
than answered: Fritz has no GPU, so the gate's `cuda_put` is null in all 15
results and the question returns to JUPITER.
"""
import logging as logging_
import os
import resource
import socket
import time
from typing import Any

import numpy as np

from .base import BaseDataStore, BaseServerManager

try:
    import cloudpickle as _pickle
except ImportError:                                  # pragma: no cover
    import pickle as _pickle

try:
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    MPI4PY_AVAILABLE = True
except ImportError:                                  # pragma: no cover
    MPI4PY_AVAILABLE = False


# ---- slot and control layout ------------------------------------------------
MAGIC = 0x52445131          # "RDQ1"; a slot that was never written reads 0
HDR_WORDS = 8
HDR_BYTES = HDR_WORDS * 8
H_MAGIC, H_TICKET, H_KEYLEN, H_NBYTES, H_PRODUCER, H_KIND, H_ECHO, H_RSV = range(8)

CTL_WORDS_PER_RING = 8      # padded to 64 B so two rings never share a cache line
W_HEAD, W_TAIL, W_CLOSED = 0, 1, 2

CLS_MAIN, CLS_DROP = 0, 1
KIND_RAW, KIND_PICKLE = 0, 1

# Per process. MPI_Win_allocate is collective, so a second DataStoreRMA in one
# process must join the first one's windows or the job deadlocks; see
# "Collective by construction" in the module docstring.
_WINDOWS = {}               # signature -> dict(win_pl, win_ct, refs, locked, ...)


def _align64(n: int) -> int:
    return (int(n) + 63) & ~63


def _mapped_files():
    """Basenames of every file mapped into this process, "(deleted)" stripped.

    Diffed across window creation, what appears is the window's own backing:
    Open MPI names those `osc_sm.*`, `osc_rdma.*`, `osc_ucx.*`. Anonymous window
    memory (`osc/ucx`) leaves nothing, which is itself the answer. Copied from
    `probes/rma_probe.py` so a coupled run reports the component the gate did.
    """
    out = set()
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/"):
                    out.add(os.path.basename(
                        parts[5].strip().replace(" (deleted)", "")))
    except OSError:
        pass
    return out


def _memlock_kb():
    """`ulimit -l`, because a window that cannot be registered is the failure.

    A login node's 8192 kB will not let UCX register IB memory at all (§5 of
    `EXPERIMENTS.fritz.md`), and M33 observation 3 puts window creation under
    `osc/ucx` at 2.8x `osc/sm` because it *is* IB registration. At M_q slots x
    N ranks that cost is untested, so it is recorded rather than assumed.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (ValueError, OSError):                    # pragma: no cover
        return (None, None)
    inf = resource.RLIM_INFINITY
    return (-1 if soft == inf else soft // 1024,
            -1 if hard == inf else hard // 1024)


class _Ring(object):
    """Geometry and cached counters of one (producer, consumer, class) ring.

    Every rank derives the identical list from one allgather, so a ring's
    control-word base and payload byte base need no agreement beyond the geometry
    signature the setup already checks.
    """

    __slots__ = ("rid", "owner", "producer", "consumer", "cls", "capacity",
                 "slot_bytes", "payload_bytes", "pay_base", "ctl_base",
                 "head", "tail", "next_ticket", "peek_ticket", "peek_key",
                 "peek_nbytes", "peek_kind", "closed", "committed", "misses")

    def __init__(self, rid, owner, producer, consumer, cls, capacity,
                 slot_bytes, payload_bytes, pay_base, ctl_base):
        self.rid = rid
        self.owner = owner
        self.producer = producer
        self.consumer = consumer
        self.cls = cls
        self.capacity = capacity
        self.slot_bytes = slot_bytes
        self.payload_bytes = payload_bytes
        self.pay_base = pay_base        # bytes into the owner's payload window
        self.ctl_base = ctl_base        # uint64 words into the owner's control window
        self.head = 0                   # consumer's cached HEAD
        self.tail = 0                   # producer's cached TAIL
        self.next_ticket = 0            # consumer: the ticket it will claim next
        self.peek_ticket = -1           # consumer: which ticket's header is cached
        self.peek_key = None
        self.peek_nbytes = 0
        self.peek_kind = KIND_RAW
        self.closed = 0                 # producer's final count + 1, 0 while open
        self.committed = 0              # producer: records it has committed (== HEAD)
        self.misses = 0                 # consumer: consecutive empty polls

    def bytes_total(self):
        return self.capacity * self.slot_bytes

    def slot_offset(self, ticket):
        return self.pay_base + (int(ticket) % self.capacity) * self.slot_bytes

    def __repr__(self):
        return ("<ring %d %d->%d cls%d cap%d slot%d pay@%d ctl@%d>"
                % (self.rid, self.producer, self.consumer, self.cls,
                   self.capacity, self.slot_bytes, self.pay_base, self.ctl_base))


class DataStoreRMA(BaseDataStore):
    """RMA ring-queue DataStore. See the module docstring for the contract."""

    # ---- construction -------------------------------------------------------
    def __init__(self, name: str, server_info, logging: bool = False,
                 log_level: int = logging_.INFO, is_colocated: bool = False):
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is required for the rma backend")
        if not MPI.Is_initialized():
            raise RuntimeError("MPI must be initialized before the rma backend")

        super().__init__(name, server_info, logging, log_level, is_colocated)

        if isinstance(server_info, dict):
            cfg = server_info.get("config", server_info)
        else:
            cfg = BaseServerManager.deserialize(server_info)["config"]
        self.config = dict(cfg)
        c = self.config

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # ---- wire format: identical on every rank ----------------------------
        self.consumers = [int(r) for r in c.get("consumers", [])]
        self.shape = tuple(c.get("shape", [319488]))
        self.dtype = np.dtype(c.get("dtype", "float32"))
        self.shapes = {str(k): tuple(v)
                       for k, v in (c.get("shapes", {}) or {}).items()}
        self.bulk_prefixes = tuple(c.get("bulk_prefixes", ["input"]))
        self.drop_prefixes = tuple(c.get("drop_prefixes", []))
        self.key_rank_field = int(c.get("key_rank_field", 1))
        self.key_max_bytes = int(c.get("key_max_bytes", 64))
        self.control_max_bytes = int(c.get("control_max_bytes", 4096))
        self.device = str(c.get("device", "cpu"))

        # ---- geometry: per producer, allgathered ------------------------------
        self.capacity = int(c.get("capacity", 8))
        self.drop_capacity = int(c.get("drop_capacity", 0)) or self.capacity
        self.max_window_bytes = int(c.get("max_window_bytes", 1 << 30))

        # ---- mechanism knobs -------------------------------------------------
        self.ring_comm_kind = str(c.get("ring_comm", "world"))
        self.ring_owner = str(c.get("ring_owner", "producer"))
        self.release_op = str(c.get("release_op", "cas"))
        self.peek = str(c.get("peek", "on"))
        self.fetch_on_poll = bool(c.get("fetch_on_poll", False))
        self.sync_mode = str(c.get("sync_mode", "sync"))
        self.windows_mode = str(c.get("windows", "singleton"))
        self.free_windows = bool(c.get("free_windows", False))
        self.win_info = dict(c.get("win_info", {}) or {})

        # ---- bounds: no wait without a deadline and a counter ------------------
        self.full_timeout_s = float(c.get("full_timeout_s", 300.0))
        self.stall_warn_s = float(c.get("stall_warn_s", 30.0))
        self.poll_max_pops = int(c.get("poll_max_pops", 64))
        # 0 = derive it in _setup_client from the rings this rank consumes. A
        # producer cannot run more than `capacity` records ahead, so that is the
        # most that can ever sit in front of a key; a constant here would either
        # raise spuriously under fan-in or stop bounding memory.
        self.readahead_max = int(c.get("readahead_max", 0))
        self.cas_max_retries = int(c.get("cas_max_retries", 64))
        self.clean_drain_s = float(c.get("clean_drain_s", 5.0))
        self.spin_before_sleep = int(c.get("spin_before_sleep", 200))
        self.sleep_max_s = float(c.get("sleep_max_s", 1e-3))
        self.closed_check_every = int(c.get("closed_check_every", 64))
        self.progress_poke = bool(c.get("progress_poke", True))

        if self.device != "cpu":
            raise NotImplementedError(
                "rma backend v1 is device='cpu' only. Build plan OQ3 (a Put "
                "from a CUDA buffer) is unanswered, not answered: the 2c gate "
                "ran on a machine with no GPU and its cuda_put is null in all "
                "15 results, so nothing here has been tried on device memory.")
        if self.ring_comm_kind == "tier":
            raise NotImplementedError(
                "rma backend: ring_comm='tier' (node-local rings under a "
                "cross-node ring) is the author's open option from the 2c gate "
                "and is not implemented. Use 'world' (the default, osc/ucx) or "
                "'node' (osc/sm, intra-node pairs only).")
        if self.ring_comm_kind not in ("world", "node"):
            raise ValueError("rma backend: ring_comm must be 'world' or 'node'")
        if self.ring_owner not in ("producer", "consumer"):
            raise ValueError("rma backend: ring_owner must be 'producer' or 'consumer'")
        if self.release_op not in ("cas", "faa"):
            raise ValueError("rma backend: release_op must be 'cas' or 'faa'")
        if self.peek not in ("on", "off"):
            raise ValueError("rma backend: peek must be 'on' or 'off'")
        if self.sync_mode not in ("sync", "none"):
            raise ValueError("rma backend: sync_mode must be 'sync' or 'none'")
        if self.windows_mode not in ("singleton", "per-config"):
            raise ValueError("rma backend: windows must be 'singleton' or 'per-config'")
        if self.capacity < 1 or self.drop_capacity < 1:
            raise ValueError("rma backend: capacity must be >= 1 (it is M_q)")
        if self.key_max_bytes < 16 or self.key_max_bytes % 8:
            raise ValueError(
                "rma backend: key_max_bytes must be >= 16 and a multiple of 8")

        self._init_counters()

        # scratch reused across calls, allocated and touched once so no timed
        # operation pays a first-touch fault (M19) and UCX registers each origin
        # buffer once rather than per call
        self._opnd = np.zeros(1, dtype=np.uint64)
        self._res = np.zeros(1, dtype=np.uint64)
        self._cmp = np.zeros(1, dtype=np.uint64)
        self._new = np.zeros(1, dtype=np.uint64)

        self._readahead = {}        # key -> decoded record, popped out of order
        self._closed_flag = False
        self._wins = None

        self._setup_client()

    def _init_counters(self):
        # producer
        self.writes = 0
        self.write_bytes = 0
        self.commits = 0
        self.dropped = 0            # plain attribute: coupled.py reads .backend.dropped (M27)
        self.blocked_writes = 0
        self.block_s_total = 0.0
        self.block_s_max = 0.0
        self.full_timeouts = 0
        self.stall_warns = 0
        self.progress_pokes = 0
        self.tail_atomics = 0
        self.tail_cache_hits = 0
        self.local_store_s = 0.0
        self.put_s = 0.0
        self.sync_calls = 0
        self.sync_s = 0.0
        self.pickled_writes = 0
        self.pickle_max_bytes = 0
        # consumer
        self.polls = 0
        self.poll_hits = 0
        self.poll_misses = 0
        self.poll_cap_hits = 0
        self.head_atomics = 0
        self.head_cache_hits = 0
        self.peeks = 0
        self.peek_bytes = 0
        self.peek_s = 0.0
        self.pops = 0
        self.pop_bytes = 0
        self.get_s = 0.0
        # The CONSUMER's MPI_Win_sync time, kept apart from `sync_s` (the PRODUCER's
        # publish sync) so neither is charged to the other side's per-message cost:
        # analysis/rdq.py's push/msg is (put_s + sync_s + local_store_s)/writes and
        # its ctl/msg is (peek_s + release_s + pop_sync_s)/pops. Under
        # ring_owner='consumer' the producer Puts and does not Sync at all, so every
        # Win_sync in the run is the consumer's - and folding it into sync_s made a
        # consumer cost appear in a producer column.
        self.pop_sync_s = 0.0
        self.releases = 0
        self.release_s = 0.0
        self.cas_retries = 0
        self.cas_giveups = 0
        self.cas_mismatch = 0
        self.readahead_pops = 0
        self.readahead_hits = 0
        self.readahead_peak = 0
        self.cleaned = 0
        self.read_timeouts = 0
        self.closed_detected = 0
        self.closed_checks = 0
        # correctness tripwires: any non-zero invalidates the run
        self.magic_mismatch = 0
        self.ticket_mismatch = 0
        self.size_mismatch = 0
        self.key_mismatch = 0
        # first-use costs, so memory registration stays visible
        self.first_atomic_s = None
        self.first_peek_s = None
        self.first_get_s = None
        self.first_put_s = None
        # teardown
        self.clean_unreleased = 0
        self.unread_records = 0

    # ---- setup --------------------------------------------------------------
    def _wire_format(self):
        """Everything a peer must already agree on for a pop to decode.

        A consumer sizes its destination buffer from its *own* config before it
        has seen the record, so a mismatch truncates or corrupts silently rather
        than raising - the hazard `mpi.py:_verify_peers` names. Here the check is
        mandatory, not opt-in, because the window allocation is collective anyway
        and has already imposed the constraint that made it optional there.
        """
        return (str(self.dtype), tuple(self.shape),
                tuple(sorted((k, tuple(v)) for k, v in self.shapes.items())),
                tuple(self.bulk_prefixes), tuple(self.drop_prefixes),
                self.key_rank_field, self.key_max_bytes,
                self.control_max_bytes, self.ring_owner,
                self.ring_comm_kind, self.device)

    def _classes(self):
        return (CLS_MAIN, CLS_DROP) if self.drop_prefixes else (CLS_MAIN,)

    def _class_of_prefix(self, pfx: str) -> int:
        if self.drop_prefixes and pfx.startswith(self.drop_prefixes):
            return CLS_DROP
        return CLS_MAIN

    def _class_payload_bytes(self):
        """Slot payload bytes per class, from config alone, so every rank agrees.

        A ring carries whatever keys its class selects, so its slot must hold the
        largest of them: every bulk prefix assigned to the class, and
        `control_max_bytes`, because a pickled key can land in either class (a
        non-bulk key goes to CLS_MAIN unless its own prefix is a drop prefix) and
        4 KB against a 1.28 MB payload slot is not worth a special case.
        """
        item = self.dtype.itemsize
        default_bytes = int(np.prod(self.shape)) * item
        out = {}
        for cls in self._classes():
            biggest = self.control_max_bytes
            for pfx in self.bulk_prefixes:
                if self._class_of_prefix(pfx) != cls:
                    continue
                shp = self.shapes.get(pfx)
                nb = int(np.prod(shp)) * item if shp else default_bytes
                if nb > biggest:
                    biggest = nb
            out[cls] = biggest
        return out

    def _setup_client(self):
        bad = [r for r in self.consumers if not 0 <= r < self.size]
        if bad:
            raise ValueError(
                "rma backend: consumer rank(s) %s lie outside COMM_WORLD "
                "(size %d). Producer and consumer must be ranks of one launch; "
                "a window cannot be created across two MPI_COMM_WORLDs."
                % (bad, self.size))

        poff = _align64(HDR_BYTES + self.key_max_bytes)
        pay_bytes = self._class_payload_bytes()
        self.payload_offset = poff
        self.payload_bytes_by_class = pay_bytes
        self.slot_bytes_by_class = {cls: poff + _align64(nb)
                                    for cls, nb in pay_bytes.items()}

        # ---- one mandatory allgather: peers, wire format, per-producer caps ----
        ident = {"rank": self.rank, "host": socket.gethostname(),
                 "pid": os.getpid(), "wire": self._wire_format(),
                 "consumers": list(self.consumers),
                 "capacity": self.capacity, "drop_capacity": self.drop_capacity}
        peers = self.comm.allgather(ident)
        mine = ident["wire"]
        wrong = [(p["rank"], p["host"], p["wire"])
                 for p in peers if p["wire"] != mine]
        if wrong:
            detail = "; ".join("rank %d on %s has %s" % w for w in wrong[:4])
            raise ValueError(
                "rma backend: rank %d has wire format %s but %s. A consumer "
                "sizes its Get from its own config, so this corrupts records "
                "rather than raising. Every rank must build the same config "
                "except `consumers`, `capacity` and `drop_capacity`."
                % (self.rank, mine, detail))
        self._peers = peers
        self._hosts = [p["host"] for p in peers]

        # ---- the ring communicator -------------------------------------------
        if self.ring_comm_kind == "node":
            self.rcomm = self.comm.Split_type(MPI.COMM_TYPE_SHARED, key=self.rank)
            self._rmap = {w: i for i, w in enumerate(self.rcomm.allgather(self.rank))}
        else:
            self.rcomm = self.comm
            self._rmap = dict((r, r) for r in range(self.size))

        # ---- geometry, computed identically on every rank ----------------------
        spec = []
        for p in range(self.size):
            for cons in peers[p]["consumers"]:
                cons = int(cons)
                if cons == p:
                    continue
                if self.ring_comm_kind == "node" and self._hosts[p] != self._hosts[cons]:
                    raise ValueError(
                        "rma backend: ring_comm='node' but the declared pair "
                        "%d -> %d does not share a node (%s vs %s). A "
                        "node-local communicator cannot carry a cross-node "
                        "ring - that is the whole reason the default is "
                        "'world' and pays osc/ucx (M33). Use ring_comm='world' "
                        "for this topology."
                        % (p, cons, self._hosts[p], self._hosts[cons]))
                for cls in self._classes():
                    cap = (peers[p]["drop_capacity"] if cls == CLS_DROP
                           else peers[p]["capacity"])
                    spec.append((p, cons, cls, int(cap)))

        owner_pay, owner_ctl = {}, {}
        self.rings = []
        self._ring_index = {}
        for rid, (p, cons, cls, cap) in enumerate(spec):
            owner = p if self.ring_owner == "producer" else cons
            sb = self.slot_bytes_by_class[cls]
            pb = owner_pay.get(owner, 0)
            cb = owner_ctl.get(owner, 0)
            ring = _Ring(rid, owner, p, cons, cls, cap, sb,
                         pay_bytes[cls], pb, cb)
            owner_pay[owner] = pb + cap * sb
            owner_ctl[owner] = cb + CTL_WORDS_PER_RING
            self.rings.append(ring)
            self._ring_index[(p, cons, cls)] = ring

        self.my_pay_bytes = owner_pay.get(self.rank, 0)
        self.my_ctl_words = owner_ctl.get(self.rank, 0)
        self.win_bytes_max = max([0] + list(owner_pay.values()))
        if self.my_pay_bytes > self.max_window_bytes:
            raise ValueError(
                "rma backend: rank %d would register %.1f MiB of window "
                "(%d rings owned x capacity x slot); max_window_bytes is "
                "%.1f MiB. Window creation is IB memory registration on this "
                "build - M33 observation 3 puts it at 0.107 s against osc/sm's "
                "0.040 for a single 1.28 MB slot, and at M_q slots x N ranks it "
                "is untested - and `ulimit -l` here is %s kB. Refused up front "
                "rather than failing inside MPI."
                % (self.rank, self.my_pay_bytes / 2.0 ** 20,
                   sum(1 for r in self.rings if r.owner == self.rank),
                   self.max_window_bytes / 2.0 ** 20, _memlock_kb()[0]))

        self._produce = [r for r in self.rings if r.producer == self.rank]
        self._consume = [r for r in self.rings if r.consumer == self.rank]
        self._my_rings = [r for r in self.rings if r.owner == self.rank]

        if self.readahead_max <= 0:
            self.readahead_max = max(
                16, 2 * sum(r.capacity for r in self._consume))
        self.readahead_max_bytes = self.readahead_max * max(
            [0] + [r.payload_bytes for r in self._consume])

        self._open_windows()
        self._alloc_scratch()

        if self.logger:
            self.logger.debug(
                "rma rank %d: %d rings (%d produced, %d consumed, %d owned), "
                "slots %s B, capacity %d/%d, window %d B + %d ctl words, "
                "comm=%s(%d) owner=%s osc=%s model=%s memlock=%s kB"
                % (self.rank, len(self.rings), len(self._produce),
                   len(self._consume), len(self._my_rings),
                   self.slot_bytes_by_class, self.capacity, self.drop_capacity,
                   self.my_pay_bytes, self.my_ctl_words, self.ring_comm_kind,
                   self.rcomm.Get_size(), self.ring_owner, self.osc_maps,
                   self.win_model, self.memlock_kb[0]))

    def _signature(self):
        """What lets two stores in one process share one pair of windows."""
        return (self.ring_comm_kind, self.ring_owner, self._wire_format(),
                tuple(self.consumers), self.capacity, self.drop_capacity,
                self.my_pay_bytes, self.my_ctl_words, self.size)

    def _open_windows(self):
        """Allocate (or join) the windows and open one passive-target epoch.

        The whole run is one epoch: `Lock_all` costs 0.0049-0.0056 ms and
        `Unlock_all` 0.0072-0.0079 (job 4242310), so paying it per message would
        be a third of the control path for nothing. The epoch is what makes every
        `Fetch_and_op`, `Compare_and_swap`, `Put` and `Get` below legal.
        """
        sig = self._signature()
        entry = _WINDOWS.get(sig)
        if entry is None:
            if _WINDOWS and self.windows_mode == "singleton":
                raise RuntimeError(
                    "rma backend: this process already holds RMA windows with a "
                    "different geometry, and MPI_Win_allocate is collective - "
                    "allocating a second pair here would deadlock against the "
                    "ranks that built one (coupled.py builds a second store on "
                    "the sim ranks only, in the closed loop). Give every "
                    "DataStoreRMA in a process the same config, or set "
                    "windows='per-config' if EVERY rank builds the same "
                    "sequence of stores.")
            before = _mapped_files()
            info, own_info = MPI.INFO_NULL, False
            if self.win_info:
                info, own_info = MPI.Info.Create(), True
                for k, v in self.win_info.items():
                    info.Set(str(k), str(v))
            t0 = time.perf_counter()
            try:
                win_pl = MPI.Win.Allocate(self.my_pay_bytes, 1, info, self.rcomm)
                t1 = time.perf_counter()
                win_ct = MPI.Win.Allocate(self.my_ctl_words * 8, 8, info, self.rcomm)
                t2 = time.perf_counter()
            finally:
                if own_info:
                    info.Free()
            new_maps = sorted(_mapped_files() - before)
            for w in (win_pl, win_ct):
                # failures become Python exceptions carrying our own message,
                # rather than an abort with no context
                w.Set_errhandler(MPI.ERRORS_RETURN)
            # first-touch every page now, on the numactl-bound domain, so no
            # timed store or Get pays a fault later (M19)
            if self.my_pay_bytes:
                np.frombuffer(win_pl.tomemory(), dtype=np.uint8)[:] = 0
            if self.my_ctl_words:
                np.frombuffer(win_ct.tomemory(), dtype=np.uint64)[:] = 0
            win_pl.Lock_all()
            win_ct.Lock_all()
            model = win_pl.Get_attr(MPI.WIN_MODEL)
            entry = {
                "win_pl": win_pl, "win_ct": win_ct, "refs": 0, "locked": True,
                "alloc_s": {"payload": t1 - t0, "control": t2 - t1},
                "maps_new": new_maps[:40],
                "osc_maps": sorted(set(c for c in ("osc_sm", "osc_rdma", "osc_ucx")
                                       for m in new_maps if m.startswith(c + "."))),
                "win_model": ("unified" if model == MPI.WIN_UNIFIED
                              else "separate" if model == MPI.WIN_SEPARATE
                              else str(model)),
            }
            _WINDOWS[sig] = entry

        entry["refs"] += 1
        self._wins = entry
        self._sig = sig
        self.win_pl = entry["win_pl"]
        self.win_ct = entry["win_ct"]
        self.win_alloc_s = dict(entry["alloc_s"])
        self.osc_maps = list(entry["osc_maps"])
        self.win_model = entry["win_model"]
        self.memlock_kb = _memlock_kb()

        if (self.ring_owner == "producer" and self.sync_mode == "sync"
                and self.win_model != "unified"):
            raise RuntimeError(
                "rma backend: ring_owner='producer' publishes a payload with a "
                "local store plus MPI_Win_sync, which is defined for the "
                "UNIFIED memory model; this window reports %r. Use "
                "ring_owner='consumer' (the producer then Puts; coupled.py's "
                "--ring-owner consumer), or say so explicitly with "
                "sync_mode='none' (--ring-sync none) and accept that a remote "
                "Get may read stale bytes - which shows up as magic_mismatch, "
                "loudly, rather than as a wrong number. This is a property of "
                "the MPI build, not of the run: sbatch/cpu/rdq.sbatch's "
                "preflight reads MPI_WIN_MODEL from probes/rma_probe.py in two "
                "minutes so that a whole campaign cannot be spent finding it "
                "out." % self.win_model)

        # local views of this rank's own window memory. MPI_Win_allocate returns
        # aligned memory and every slot offset is a multiple of 64, so the uint64
        # view is safe.
        if self.my_pay_bytes:
            self._pl_u8 = np.frombuffer(self.win_pl.tomemory(), dtype=np.uint8)
            self._pl_u64 = np.frombuffer(self.win_pl.tomemory(), dtype=np.uint64)
        else:
            self._pl_u8 = np.zeros(0, dtype=np.uint8)
            self._pl_u64 = np.zeros(0, dtype=np.uint64)

    def _alloc_scratch(self):
        """Origin buffers, allocated and touched once.

        UCX registers an origin buffer on first use - which is why the gate's
        `first` column is an upper bound rather than a measurement. One buffer per
        ring keeps that cost at setup instead of once per read, and
        `first_peek_s` / `first_get_s` / `first_put_s` record what it was.
        """
        self._peek_buf = {}
        self._stage_buf = {}
        for r in self._consume:
            self._peek_buf[r.rid] = np.zeros(self.payload_offset, dtype=np.uint8)
        if self.ring_owner == "consumer":
            for r in self._produce:
                self._stage_buf[r.rid] = np.zeros(r.slot_bytes, dtype=np.uint8)

    # ---- key contract -------------------------------------------------------
    def _src(self, key: str) -> int:
        """Producer rank, read from field `key_rank_field` of the key.

        A ticket queue has no directory either, and `stage_read` has no source
        argument, so the source comes from the key exactly as in `mpi.py`; the
        driver's key convention is unchanged.
        """
        try:
            return int(key.split("_")[self.key_rank_field])
        except (IndexError, ValueError):
            raise ValueError(
                "rma backend reads the producer rank from field %d of the key "
                "(0-based, split on '_'), so keys must look like "
                "<prefix>_<rank>_... - got %r. Set key_rank_field if your keys "
                "are shaped differently." % (self.key_rank_field, key))

    def _is_bulk(self, key: str) -> bool:
        return key.startswith(self.bulk_prefixes)

    def _class_of(self, key: str) -> int:
        if self.drop_prefixes and key.startswith(self.drop_prefixes):
            return CLS_DROP
        return CLS_MAIN

    def _shape_for(self, key: str) -> tuple:
        for pfx, shp in self.shapes.items():
            if key.startswith(pfx):
                return shp
        return self.shape

    def _ring_out(self, key: str, consumer: int) -> _Ring:
        ring = self._ring_index.get((self.rank, consumer, self._class_of(key)))
        if ring is None:                             # pragma: no cover
            raise KeyError(
                "rma backend: rank %d has no ring to consumer %d for %r; "
                "`consumers` at setup did not list it."
                % (self.rank, consumer, key))
        return ring

    def _ring_in(self, key: str) -> _Ring:
        src = self._src(key)
        ring = self._ring_index.get((src, self.rank, self._class_of(key)))
        if ring is None:
            raise KeyError(
                "rma backend: rank %d consumes no ring from producer %d for "
                "%r. The producer must list this rank in `consumers`: a queue is "
                "addressed by ticket, but the rings are still built from the "
                "declared topology." % (self.rank, src, key))
        return ring

    def _key_bytes(self, key: str) -> bytes:
        kb = key.encode()
        if len(kb) > self.key_max_bytes:
            raise ValueError(
                "rma backend: key %r is %d bytes and a slot carries %d. The key "
                "travels verbatim in the slot (no hash, so no collision class); "
                "raise key_max_bytes, which changes the slot size and therefore "
                "the window." % (key, len(kb), self.key_max_bytes))
        return kb

    # ---- atomics ------------------------------------------------------------
    def _target(self, owner: int) -> int:
        return self._rmap[owner]

    def _atomic(self, ring: _Ring, word: int, op, val=0) -> int:
        """One accumulate-class op on a control word, and its Flush.

        Always flushed: the result is read on the next line, and an atomic whose
        flush is deferred has not happened. This is the operation the gate timed
        as `fetch_and_op` - 0.0038-0.0047 ms under `osc/sm`, 0.0120-0.0128 under
        `osc/ucx` within a node, and the cross-node figure of M33's default pass.
        """
        t0 = time.perf_counter()
        self._opnd[0] = int(val)
        tgt = self._target(ring.owner)
        self.win_ct.Fetch_and_op([self._opnd, MPI.UINT64_T],
                                 [self._res, MPI.UINT64_T],
                                 tgt, ring.ctl_base + word, op)
        self.win_ct.Flush(tgt)
        dt = time.perf_counter() - t0
        if self.first_atomic_s is None:
            self.first_atomic_s = dt
        return int(self._res[0])

    def _cas(self, ring: _Ring, word: int, expect: int, new: int) -> int:
        """Compare-and-swap a control word; returns the value found there."""
        self._cmp[0] = int(expect)
        self._new[0] = int(new)
        tgt = self._target(ring.owner)
        self.win_ct.Compare_and_swap([self._new, MPI.UINT64_T],
                                     [self._cmp, MPI.UINT64_T],
                                     [self._res, MPI.UINT64_T],
                                     tgt, ring.ctl_base + word)
        self.win_ct.Flush(tgt)
        return int(self._res[0])

    def _read_head(self, ring: _Ring) -> int:
        self.head_atomics += 1
        ring.head = self._atomic(ring, W_HEAD, MPI.NO_OP, 0)
        return ring.head

    def _read_tail(self, ring: _Ring) -> int:
        self.tail_atomics += 1
        ring.tail = self._atomic(ring, W_TAIL, MPI.NO_OP, 0)
        return ring.tail

    def _read_closed(self, ring: _Ring) -> int:
        self.closed_checks += 1
        ring.closed = self._atomic(ring, W_CLOSED, MPI.NO_OP, 0)
        return ring.closed

    # ---- push ---------------------------------------------------------------
    def _has_slot(self, ring: _Ring) -> bool:
        """Free slot for the next commit? One atomic only when the cache says no."""
        if ring.committed - ring.tail < ring.capacity:
            self.tail_cache_hits += 1
            return True
        self._read_tail(ring)
        return ring.committed - ring.tail < ring.capacity

    def _block_for_slot(self, ring: _Ring, key: str):
        """Bounded wait for `HEAD - TAIL < capacity`, or raise.

        A lossless producer either gets a slot or says why it did not: an
        unbounded stall is what turns a wrong message count into a job that burns
        its whole walltime with no output, and `mpi.py` paid for that twice
        (jobs 1418787, 1418788).
        """
        self.blocked_writes += 1
        started = time.perf_counter()
        deadline = started + self.full_timeout_s
        warned, spins = False, 0
        while True:
            self._read_tail(ring)
            if ring.committed - ring.tail < ring.capacity:
                break
            now = time.perf_counter()
            if not warned and now - started > self.stall_warn_s:
                warned = True
                self.stall_warns += 1
                msg = ("rma backend: rank %d has waited %.0fs for a free slot on "
                       "%r (ring %d -> %d, capacity %d, head %d tail %d). The "
                       "ring is bounded at every payload size, so this is real "
                       "backpressure - but it clears only if the consumer pops. "
                       "Check that it pops as many records as this rank writes."
                       % (self.rank, self.stall_warn_s, key, ring.producer,
                          ring.consumer, ring.capacity, ring.committed, ring.tail))
                if self.logger:
                    self.logger.warning(msg)
                else:
                    print(msg, flush=True)
            if now > deadline:
                self.full_timeouts += 1
                raise TimeoutError(
                    "rma backend: no free slot for %r within %.0fs (ring %d -> "
                    "%d, capacity %d, head %d tail %d). Either the consumer "
                    "stopped popping, or it pops fewer records than this rank "
                    "writes." % (key, self.full_timeout_s, ring.producer,
                                 ring.consumer, ring.capacity, ring.committed,
                                 ring.tail))
            # Spin first so the common short wait pays nothing, then back off. A
            # flat sleep is what cost 5x on read latency on the direct path
            # (0.051 -> 0.259 ms per 1 MB, job 1619893); here the hazard differs
            # but is no smaller - whether an `osc/ucx` atomic needs progress on
            # the target rank is the gate's untested question (v3 section 15.8 ii;
            # `rma_probe.py --owner-poll` exists and was never run), so a
            # sleeping producer may be a consumer's stall. Poke the progress
            # engine while waiting rather than only sleeping.
            if spins < self.spin_before_sleep:
                spins += 1
            else:
                if self.progress_poke:
                    self.comm.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG)
                    self.progress_pokes += 1
                time.sleep(min(1e-5 * 2 ** ((spins - self.spin_before_sleep) // 50),
                               self.sleep_max_s))
                spins += 1
        waited = time.perf_counter() - started
        self.block_s_total += waited
        if waited > self.block_s_max:
            self.block_s_max = waited

    def _encode(self, key: str, data):
        """(kind, flat uint8 view of the payload) for one record."""
        if self._is_bulk(key):
            arr = np.ascontiguousarray(data, dtype=self.dtype)
            return KIND_RAW, arr.reshape(-1).view(np.uint8)
        blob = _pickle.dumps(data)
        self.pickled_writes += 1
        if len(blob) > self.pickle_max_bytes:
            self.pickle_max_bytes = len(blob)
        return KIND_PICKLE, np.frombuffer(blob, dtype=np.uint8)

    def _write_slot(self, ring: _Ring, ticket: int, kb: bytes, kind: int, pay):
        """Header, key and payload into the slot; does NOT commit."""
        off = ring.slot_offset(ticket)
        nbytes = int(pay.nbytes)
        if self.ring_owner == "producer":
            t0 = time.perf_counter()
            hw = off // 8
            u64, u8 = self._pl_u64, self._pl_u8
            u64[hw + H_MAGIC] = MAGIC
            u64[hw + H_TICKET] = ticket
            u64[hw + H_KEYLEN] = len(kb)
            u64[hw + H_NBYTES] = nbytes
            u64[hw + H_PRODUCER] = self.rank
            u64[hw + H_KIND] = kind
            u64[hw + H_ECHO] = ticket
            u64[hw + H_RSV] = 0
            ks = off + HDR_BYTES
            u8[ks:ks + len(kb)] = np.frombuffer(kb, dtype=np.uint8)
            if len(kb) < self.key_max_bytes:
                u8[ks + len(kb):off + self.payload_offset] = 0
            ps = off + self.payload_offset
            u8[ps:ps + nbytes] = pay
            self.local_store_s += time.perf_counter() - t0
            if self.sync_mode == "sync":
                # publish the private copy into the public one BEFORE the commit
                # makes the record visible; the consumer's Get reads the public copy
                t1 = time.perf_counter()
                self.win_pl.Sync()
                self.sync_s += time.perf_counter() - t1
                self.sync_calls += 1
            return
        buf = self._stage_buf[ring.rid]
        t0 = time.perf_counter()
        hdr = buf[:HDR_BYTES].view(np.uint64)
        hdr[H_MAGIC] = MAGIC
        hdr[H_TICKET] = ticket
        hdr[H_KEYLEN] = len(kb)
        hdr[H_NBYTES] = nbytes
        hdr[H_PRODUCER] = self.rank
        hdr[H_KIND] = kind
        hdr[H_ECHO] = ticket
        hdr[H_RSV] = 0
        buf[HDR_BYTES:HDR_BYTES + len(kb)] = np.frombuffer(kb, dtype=np.uint8)
        buf[HDR_BYTES + len(kb):self.payload_offset] = 0
        buf[self.payload_offset:self.payload_offset + nbytes] = pay
        self.local_store_s += time.perf_counter() - t0
        tgt = self._target(ring.owner)
        t1 = time.perf_counter()
        self.win_pl.Put(buf[:self.payload_offset + nbytes], tgt, off)
        self.win_pl.Flush(tgt)
        dt = time.perf_counter() - t1
        self.put_s += dt
        if self.first_put_s is None:
            self.first_put_s = dt
        # Put + Flush completes the origin buffer, so this one buffer is reusable
        # on the next write: M12's rotate-M+2 discipline and M28's
        # rewrite-in-flight corruption do not exist on this path.

    def stage_write(self, key: str, data: Any, persistant: bool = True,
                    client_id: int = 0, is_local: bool = False):
        """Push one record onto the ring of every declared consumer."""
        if self._closed_flag:
            raise RuntimeError("rma backend: stage_write after clean()")
        kb = self._key_bytes(key)
        drop = bool(self.drop_prefixes) and key.startswith(self.drop_prefixes)
        kind, pay = self._encode(key, data)
        nbytes = int(pay.nbytes)

        rings = []
        for dest in self.consumers:
            if dest == self.rank:
                continue
            ring = self._ring_out(key, dest)
            if nbytes > ring.payload_bytes:
                raise ValueError(
                    "rma backend: %r is %d bytes and a class-%d slot carries "
                    "%d. Slot size is fixed at window creation - that is what "
                    "makes the queue bounded at every payload size - so this "
                    "needs a larger `shape`/`shapes` entry (bulk) or a larger "
                    "`control_max_bytes` (pickled), and a new window."
                    % (key, nbytes, ring.cls, ring.payload_bytes))
            rings.append(ring)
        if not rings:
            return

        if drop:
            # keep-last-1 on a bounded queue drops rather than stalls, exactly as
            # `mpi.py` does: the version being held up is about to be superseded,
            # and in a closed loop blocking here deadlocks - the producer stops
            # consuming the reverse direction, the consumer's own publishes back
            # up, and it can no longer reach the read that would have drained
            # this ring. A publish is dropped for EVERY consumer at once, so
            # `dropped` counts publishes and coupled.py's wdone arithmetic
            # (wversion - w_dropped) stays exact per consumer.
            if not all(self._has_slot(r) for r in rings):
                self.dropped += 1
                return
        else:
            for ring in rings:
                if not self._has_slot(ring):
                    self._block_for_slot(ring, key)

        for ring in rings:
            ticket = ring.committed
            self._write_slot(ring, ticket, kb, kind, pay)
            # commit: the record is visible exactly here. THE RETURN VALUE IS THE
            # PRODUCER'S HALF OF THE INVARIANT AND IT IS FREE. Fetch_and_op(SUM, 1)
            # returns the OLD HEAD, which on a single-producer ring must be this
            # ticket; `_atomic` has already flushed it and read it back into
            # `self._res`, so checking it costs an integer compare. This is the exact
            # mirror of `_release`'s CAS, whose return value is kept for the same
            # reason ("a free assertion that the ring really had one consumer"). It
            # catches a second producer on one ring, and a ctl_base/pay_base
            # collision, HERE - at the end that caused it, before the slot's bytes are
            # overwritten - instead of at the consumer's next pop, where the same
            # fault surfaces as a ticket_mismatch that names the wrong half of the
            # ring.
            old = self._atomic(ring, W_HEAD, MPI.SUM, 1)
            if old != ticket:
                self.ticket_mismatch += 1
                raise RuntimeError(
                    "rma backend: committing ticket %d to ring %d -> %d found HEAD "
                    "at %d, not %d. On a single-producer ring the fetched HEAD is "
                    "the ticket being committed, so either a second rank is "
                    "producing into this ring or its control words overlap another "
                    "ring's (ctl_base %d, capacity %d)."
                    % (ticket, ring.producer, ring.consumer, old, ticket,
                       ring.ctl_base, ring.capacity))
            ring.committed = ticket + 1
            self.commits += 1
        self.writes += 1
        self.write_bytes += nbytes * len(rings)

    # ---- pop ----------------------------------------------------------------
    def _peek_header(self, ring: _Ring) -> str:
        """Header + key of the next unclaimed record. Does not claim it.

        Under `ring_owner="producer"` this is a `payload_offset`-byte remote Get
        (128 B by default) - more than an `MPI_Iprobe`'s 0.0123 ms, and the price
        of `poll_staged_data` telling the truth. Under `ring_owner="consumer"` it is
        a local load behind an `MPI_Win_sync`, which is cheap but NOT free and NOT
        untimed: the sync goes into `pop_sync_s` and the access into `peek_s`, and
        `peeks` counts both arms. Leaving that arm untimed made analysis/rdq.py's
        ctl/msg and pay/msg columns print 0.0000 for the whole consumer-owned half of
        the owner sweep - a table saying the pop is free when it is a Win_sync plus a
        1.28 MB memcpy. `peek_bytes` stays a WIRE count and is not incremented here.
        """
        t = ring.next_ticket
        if ring.peek_ticket == t and ring.peek_key is not None:
            return ring.peek_key
        if self.ring_owner == "consumer":
            if self.sync_mode == "sync":
                t0 = time.perf_counter()
                self.win_pl.Sync()
                self.pop_sync_s += time.perf_counter() - t0
                self.sync_calls += 1
            t0 = time.perf_counter()
            off = ring.slot_offset(t)
            raw = self._pl_u8[off:off + self.payload_offset]
            dt = time.perf_counter() - t0
            self.peek_s += dt
            if self.first_peek_s is None:
                self.first_peek_s = dt
            self.peeks += 1
        else:
            buf = self._peek_buf[ring.rid]
            tgt = self._target(ring.owner)
            t0 = time.perf_counter()
            self.win_pl.Get(buf, tgt, ring.slot_offset(t))
            self.win_pl.Flush(tgt)
            dt = time.perf_counter() - t0
            self.peek_s += dt
            if self.first_peek_s is None:
                self.first_peek_s = dt
            self.peeks += 1
            self.peek_bytes += self.payload_offset
            raw = buf
        hdr = raw[:HDR_BYTES].view(np.uint64)
        if int(hdr[H_MAGIC]) != MAGIC:
            self.magic_mismatch += 1
            raise RuntimeError(
                "rma backend: slot for ticket %d of ring %d -> %d has magic "
                "0x%x, not 0x%x. HEAD said the record was committed, so either "
                "the commit was ordered before the payload (sync_mode) or the "
                "two ends disagree about the geometry."
                % (t, ring.producer, ring.consumer, int(hdr[H_MAGIC]), MAGIC))
        if int(hdr[H_TICKET]) != t or int(hdr[H_ECHO]) != t:
            self.ticket_mismatch += 1
            raise RuntimeError(
                "rma backend: slot for ticket %d of ring %d -> %d carries "
                "ticket %d (echo %d). A producer lapped a slot the consumer had "
                "not released, which the capacity rule HEAD - TAIL < %d is "
                "supposed to make impossible."
                % (t, ring.producer, ring.consumer, int(hdr[H_TICKET]),
                   int(hdr[H_ECHO]), ring.capacity))
        klen = int(hdr[H_KEYLEN])
        if not 0 < klen <= self.key_max_bytes:
            self.magic_mismatch += 1
            raise RuntimeError(
                "rma backend: slot for ticket %d of ring %d -> %d declares a "
                "%d-byte key against key_max_bytes %d."
                % (t, ring.producer, ring.consumer, klen, self.key_max_bytes))
        ring.peek_ticket = t
        ring.peek_key = bytes(raw[HDR_BYTES:HDR_BYTES + klen]).decode()
        ring.peek_nbytes = int(hdr[H_NBYTES])
        ring.peek_kind = int(hdr[H_KIND])
        return ring.peek_key

    def _fetch_payload(self, ring: _Ring, key: str):
        """Get the peeked record's payload and decode it. Does not release."""
        t = ring.next_ticket
        nbytes, kind = ring.peek_nbytes, ring.peek_kind
        off = ring.slot_offset(t) + self.payload_offset
        if kind == KIND_RAW:
            shape = self._shape_for(key)
            dest = np.empty(shape, dtype=self.dtype)
            if int(dest.nbytes) != nbytes:
                self.size_mismatch += 1
                raise ValueError(
                    "rma backend: %r carries %d payload bytes but this rank "
                    "sizes it at %d from shape %s dtype %s. A consumer sizes its "
                    "Get from its own config, so this is a wire mismatch the "
                    "setup check should have caught."
                    % (key, nbytes, dest.nbytes, shape, self.dtype))
            view = dest.reshape(-1).view(np.uint8)
        else:
            view = np.empty(nbytes, dtype=np.uint8)
            dest = None

        if self.ring_owner == "consumer":
            # Local, and the one place "local" is expensive: this is an nbytes memcpy
            # out of our own window (1.28 MB in every measured cell). It is timed into
            # get_s exactly as the producer's mirror-image local store is timed into
            # local_store_s, so pay/msg means the same thing in both owner arms; the
            # Win_sync goes to pop_sync_s, not to the producer's sync_s.
            if self.sync_mode == "sync":
                t0 = time.perf_counter()
                self.win_pl.Sync()
                self.pop_sync_s += time.perf_counter() - t0
                self.sync_calls += 1
            t0 = time.perf_counter()
            view[:] = self._pl_u8[off:off + nbytes]      # local: the zero-copy arm
            dt = time.perf_counter() - t0
            self.get_s += dt
            if self.first_get_s is None:
                self.first_get_s = dt
        else:
            tgt = self._target(ring.owner)
            t0 = time.perf_counter()
            self.win_pl.Get(view, tgt, off)
            self.win_pl.Flush(tgt)
            dt = time.perf_counter() - t0
            self.get_s += dt
            if self.first_get_s is None:
                self.first_get_s = dt
        self.pop_bytes += nbytes
        return dest if kind == KIND_RAW else _pickle.loads(bytes(view))

    def _release(self, ring: _Ring):
        """Free the slot just read.

        After the payload has landed, never before: the producer's rule is
        `HEAD - TAIL < capacity`, so bumping TAIL early is exactly what would let
        it overwrite a slot with a Get in flight.
        """
        t = ring.next_ticket
        t0 = time.perf_counter()
        if self.release_op == "cas":
            found = self._cas(ring, W_TAIL, t, t + 1)
            tries = 0
            while found != t:
                # a single-consumer ring makes this unreachable, which is why it
                # is the assertion: a second consumer would show up here rather
                # than by corrupting the queue
                self.cas_mismatch += 1
                self.cas_retries += 1
                tries += 1
                if tries > self.cas_max_retries:
                    self.cas_giveups += 1
                    raise RuntimeError(
                        "rma backend: TAIL of ring %d -> %d would not move from "
                        "%d after %d compare-and-swap attempts (found %d). The "
                        "ring is single-consumer by construction, so a second "
                        "consumer on it is a topology error."
                        % (ring.producer, ring.consumer, t, tries, found))
                found = self._cas(ring, W_TAIL, t, t + 1)
        else:
            self._atomic(ring, W_TAIL, MPI.SUM, 1)
        self.release_s += time.perf_counter() - t0
        self.releases += 1
        ring.next_ticket = t + 1
        ring.peek_ticket = -1
        ring.peek_key = None

    def _pop_one(self, ring: _Ring):
        """Peek, fetch, release. Returns (key, object)."""
        key = self._peek_header(ring)
        obj = self._fetch_payload(ring, key)
        self._release(ring)
        self.pops += 1
        return key, obj

    def _park(self, key: str, obj):
        if len(self._readahead) >= self.readahead_max:
            raise RuntimeError(
                "rma backend: %d records are parked in the read-ahead dict, the "
                "derived readahead_max (2 x this rank's consumed capacity, "
                "%.1f MiB of payload). A producer cannot run more than capacity "
                "records ahead, so reaching this means a consumer is polling for "
                "keys it never reads. Parked: %s..."
                % (len(self._readahead), self.readahead_max_bytes / 2.0 ** 20,
                   sorted(self._readahead)[:4]))
        if key in self._readahead:
            # THE ONE LOSS PATH THE CORRECTNESS LEDGER CANNOT SEE. The read-ahead is
            # a dict keyed by the record's key, so a second parked record with the
            # same key would silently replace the first: `commits` would still equal
            # `pops`, `readahead_left` would be one short rather than wrong, and a
            # record would be gone with no counter moving. It is raised as
            # key_mismatch - an existing tripwire the gate and analysis/rdq.py
            # already treat as voiding the run - because a silent loss in a queue
            # whose whole claim is exactly-once delivery is worse than a dead rank.
            # Nothing in coupled.py reaches it today: forward records carry distinct
            # indices, and the weights ring holds same-key records but never parks
            # (the peek always matches, including under fetch_on_poll, where each
            # park is consumed by the following stage_read). A per-key deque would
            # make it servable; that is a chunk-1 change and it is not made here.
            self.key_mismatch += 1
            raise RuntimeError(
                "rma backend: %r is already parked in the read-ahead and a second "
                "copy would overwrite it - a record lost with no counter moving. "
                "The read-ahead is keyed by key, so two UNREAD versions of one key "
                "cannot be parked at once: either a consumer polls for keys it "
                "never reads, or a producer publishes the same key twice without "
                "the consumer reading in between. Parked: %s"
                % (key, sorted(self._readahead)[:8]))
        self._readahead[key] = obj
        if len(self._readahead) > self.readahead_peak:
            self.readahead_peak = len(self._readahead)

    def _nonempty(self, ring: _Ring) -> bool:
        if ring.head > ring.next_ticket:
            self.head_cache_hits += 1
            return True
        self._read_head(ring)
        return ring.head > ring.next_ticket

    def poll_staged_data(self, key: str, client_id: int = 0,
                         is_local: bool = False) -> bool:
        """Is `key` available? One atomic when the ring is empty.

        Readiness is `HEAD > TAIL`, never an inspection of the slot's own bytes.
        With `peek="on"` (the default) the head record's key is then checked
        exactly, so a poll for a key queued *behind* other records answers
        honestly - which every driver path depends on: the trainer's teardown
        polls for `simdone_<src>` while unread `input` records are still in the
        ring, and a poll that said "yes, something is here" would send it into a
        read that cannot complete. With `peek="off"` the poll costs one atomic and
        trusts FIFO; `key_mismatch` then counts every time that trust was wrong.
        """
        self.polls += 1
        if key in self._readahead:
            self.readahead_hits += 1
            self.poll_hits += 1
            return True
        ring = self._ring_in(key)
        for _ in range(max(1, self.poll_max_pops)):
            if not self._nonempty(ring):
                ring.misses += 1
                self.poll_misses += 1
                if (self.closed_check_every
                        and ring.misses % self.closed_check_every == 0
                        and self._read_closed(ring)
                        and ring.next_ticket >= ring.closed - 1):
                    self.closed_detected += 1
                return False
            ring.misses = 0
            if self.peek == "off":
                self.poll_hits += 1
                return True
            if self._peek_header(ring) == key:
                if self.fetch_on_poll:
                    # moves the payload Get from read_tot to poll_tot; exists so
                    # the analysis can check that attribution, not as a default
                    k, obj = self._pop_one(ring)
                    self._park(k, obj)
                self.poll_hits += 1
                return True
            k, obj = self._pop_one(ring)             # v3 section 15.5's O(versions)
            self.readahead_pops += 1
            self._park(k, obj)
            if key in self._readahead:
                self.poll_hits += 1
                return True
        self.poll_cap_hits += 1
        self.poll_misses += 1
        return False

    def stage_read(self, key: str, client_id: int = 0, timeout: int = 30,
                   is_local: bool = False):
        """Pop `key`, draining records in front of it into the read-ahead dict."""
        if key in self._readahead:
            self.readahead_hits += 1
            return self._readahead.pop(key)
        ring = self._ring_in(key)
        deadline = (time.perf_counter() + timeout) if timeout and timeout > 0 else None
        spins = 0
        while True:
            if self._nonempty(ring):
                got_key, obj = self._pop_one(ring)
                if got_key == key:
                    return obj
                if self.peek == "off":
                    # the poll trusted FIFO and FIFO did not hold for this caller
                    self.key_mismatch += 1
                else:
                    self.readahead_pops += 1
                self._park(got_key, obj)
                continue
            # empty: fail fast if the producer has closed past this ticket, rather
            # than waiting out a timeout for a record that cannot come
            if self._read_closed(ring) and ring.next_ticket >= ring.closed - 1:
                self.closed_detected += 1
                raise TimeoutError(
                    "rma backend: %r will never arrive. Producer %d closed its "
                    "ring to rank %d after %d records and this rank has released "
                    "%d, so the ring is empty and will stay empty."
                    % (key, ring.producer, ring.consumer,
                       max(0, ring.closed - 1), ring.next_ticket))
            if deadline is not None and time.perf_counter() > deadline:
                self.read_timeouts += 1
                raise TimeoutError(
                    "rma backend: no %r from rank %d within %ss (ring %d -> %d, "
                    "head %d, next ticket %d, %d records parked). Either the "
                    "producer never wrote it, or it wrote fewer records than "
                    "this rank reads."
                    % (key, ring.producer, timeout, ring.producer, ring.consumer,
                       ring.head, ring.next_ticket, len(self._readahead)))
            if spins < self.spin_before_sleep:
                spins += 1
            else:
                time.sleep(min(1e-5 * 2 ** ((spins - self.spin_before_sleep) // 50),
                               self.sleep_max_s))
                spins += 1

    def clean_staged_data(self, key: str, client_id: int = 0,
                          is_local: bool = False):
        """Drop a parked record. A released slot is already gone.

        The ring itself needs no cleaning - `_release` freed the slot - but the
        read-ahead dict holds whole payloads, so a caller that parks records it
        never reads has somewhere to say so.
        """
        if self._readahead.pop(key, None) is not None:
            self.cleaned += 1

    # ---- teardown -----------------------------------------------------------
    def clean(self):
        """Wait for consumers to drain, publish CLOSED, end the epoch.

        Waiting first, bounded by `clean_drain_s`, for the reason `mpi.py`
        records: a producer that tears down while its consumer is still draining
        starves it, and jobs 1419252/1419253 died exactly so. Nothing is
        cancelled here - the records stay in the consumer's reach until the window
        goes away - but CLOSED is what lets a consumer stop waiting, so it is
        published only once the ring is empty or the deadline has passed.

        **This ends the epoch for the whole process** (`Unlock_all`, which is not
        collective). Any other store sharing these windows is unusable afterwards.
        """
        if self._closed_flag:
            return
        deadline = time.perf_counter() + self.clean_drain_s
        for ring in self._produce:
            while ring.committed > ring.tail:
                self._read_tail(ring)
                if ring.committed <= ring.tail or time.perf_counter() > deadline:
                    break
                time.sleep(1e-4)
            behind = max(0, ring.committed - ring.tail)
            if behind:
                self.clean_unreleased += behind
                msg = ("rma backend: rank %d closing its ring to rank %d with %d "
                       "of %d records unreleased after waiting %.0fs - a consumer "
                       "expected data it will now stop waiting for."
                       % (self.rank, ring.consumer, behind, ring.committed,
                          self.clean_drain_s))
                if self.logger:
                    self.logger.warning(msg)
                else:
                    print(msg, flush=True)
            self._atomic(ring, W_CLOSED, MPI.REPLACE, ring.committed + 1)

        for ring in self._consume:
            self.unread_records += max(0, self._read_head(ring) - ring.next_ticket)
        self.unread_records += len(self._readahead)

        self._closed_flag = True
        entry = self._wins
        entry["refs"] -= 1
        if entry.get("locked"):
            # Unlock_all completes this process's shared lock on every target; it
            # is not collective, so one rank closing early cannot hang another.
            # Done on the FIRST clean() rather than at refs == 0, because the
            # driver's closed loop leaves one store per sim rank uncleaned and a
            # refcount would never reach zero there.
            try:
                self.win_ct.Unlock_all()
                self.win_pl.Unlock_all()
            except Exception as e:                     # pragma: no cover
                if self.logger:
                    self.logger.warning("rma backend: Unlock_all: %r" % (e,))
            entry["locked"] = False
        if self.free_windows and entry["refs"] <= 0:
            # Win_free IS collective. Only safe where every rank of the ring
            # communicator constructed exactly one store and cleans it once.
            try:
                self.win_ct.Free()
                self.win_pl.Free()
                _WINDOWS.pop(self._sig, None)
            except Exception as e:                     # pragma: no cover
                if self.logger:
                    self.logger.warning("rma backend: Win.Free: %r" % (e,))
        if self.logger:
            self.logger.info("rma backend: clean() rank %d: %s"
                             % (self.rank, self.stats()))

    # ---- instrumentation ----------------------------------------------------
    def stats(self) -> dict:
        """Every counter, flat, stdlib types only - a collator reads this.

        `magic_mismatch`, `ticket_mismatch`, `size_mismatch`, `cas_mismatch` and
        `key_mismatch` are correctness tripwires: any of them non-zero invalidates
        the run rather than degrading it. `head_atomics` and `tail_atomics`
        against `pops` and `writes` are what show the counter caching held, i.e.
        that a message really cost one atomic per side and not four.
        """
        return {
            # geometry, configuration and registration
            "rings": len(self.rings), "rings_produced": len(self._produce),
            "rings_consumed": len(self._consume),
            "rings_owned": len(self._my_rings),
            "capacity": self.capacity, "drop_capacity": self.drop_capacity,
            "slot_bytes": dict((str(k), int(v))
                               for k, v in self.slot_bytes_by_class.items()),
            "payload_bytes": dict((str(k), int(v))
                                  for k, v in self.payload_bytes_by_class.items()),
            "payload_offset": self.payload_offset,
            "win_payload_bytes": self.my_pay_bytes,
            "win_ctl_bytes": self.my_ctl_words * 8,
            "win_payload_bytes_max": self.win_bytes_max,
            "win_alloc_s": self.win_alloc_s, "win_model": self.win_model,
            "osc_maps": self.osc_maps, "memlock_kb": list(self.memlock_kb),
            "ring_comm": self.ring_comm_kind,
            "ring_comm_size": self.rcomm.Get_size(),
            "ring_owner": self.ring_owner, "release_op": self.release_op,
            "peek": self.peek, "sync_mode": self.sync_mode,
            "windows": self.windows_mode, "free_windows": self.free_windows,
            # producer
            "writes": self.writes, "write_bytes": self.write_bytes,
            "commits": self.commits, "dropped": self.dropped,
            "blocked_writes": self.blocked_writes,
            "block_s_total": self.block_s_total, "block_s_max": self.block_s_max,
            "full_timeouts": self.full_timeouts, "stall_warns": self.stall_warns,
            "progress_pokes": self.progress_pokes,
            "tail_atomics": self.tail_atomics,
            "tail_cache_hits": self.tail_cache_hits,
            "local_store_s": self.local_store_s, "put_s": self.put_s,
            "sync_calls": self.sync_calls, "sync_s": self.sync_s,
            "pickled_writes": self.pickled_writes,
            "pickle_max_bytes": self.pickle_max_bytes,
            # consumer
            "polls": self.polls, "poll_hits": self.poll_hits,
            "poll_misses": self.poll_misses, "poll_cap_hits": self.poll_cap_hits,
            "head_atomics": self.head_atomics,
            "head_cache_hits": self.head_cache_hits,
            "peeks": self.peeks, "peek_bytes": self.peek_bytes,
            "peek_s": self.peek_s, "pop_sync_s": self.pop_sync_s,
            "pops": self.pops,
            "pop_bytes": self.pop_bytes, "get_s": self.get_s,
            "releases": self.releases, "release_s": self.release_s,
            "cas_retries": self.cas_retries, "cas_giveups": self.cas_giveups,
            "readahead_pops": self.readahead_pops,
            "readahead_hits": self.readahead_hits,
            "readahead_peak": self.readahead_peak,
            "readahead_left": len(self._readahead),
            "readahead_max": self.readahead_max,
            "readahead_max_bytes": self.readahead_max_bytes,
            "poll_max_pops": self.poll_max_pops,
            "cleaned": self.cleaned, "read_timeouts": self.read_timeouts,
            "closed_detected": self.closed_detected,
            "closed_checks": self.closed_checks,
            # first use, i.e. memory registration
            "first_atomic_s": self.first_atomic_s,
            "first_peek_s": self.first_peek_s,
            "first_get_s": self.first_get_s,
            "first_put_s": self.first_put_s,
            # tripwires and teardown
            "magic_mismatch": self.magic_mismatch,
            "ticket_mismatch": self.ticket_mismatch,
            "size_mismatch": self.size_mismatch,
            "cas_mismatch": self.cas_mismatch,
            "key_mismatch": self.key_mismatch,
            "clean_unreleased": self.clean_unreleased,
            "unread_records": self.unread_records,
        }

    def dump(self):
        s = self.stats()
        if self.logger:
            self.logger.info("rma backend rank %d: %s" % (self.rank, s))
        return s


class ServerManagerRMA(BaseServerManager):
    """No server exists; the windows are the transport.

    Unlike every store backend there is nothing to start, and unlike the direct
    path there is nothing to *address* either. The rings are created inside
    `DataStoreRMA.__init__`, because `MPI_Win_allocate` is collective and only the
    clients are guaranteed to exist on every rank.
    """

    def start_server(self):
        if self.logger:
            self.logger.info("rma backend: no server to start; the windows are "
                             "allocated collectively by the clients")

    def stop_server(self):
        if self.logger:
            self.logger.info("rma backend: no server to stop")

    def get_server_info(self) -> dict:
        return {"name": self.name, "type": self.config.type,
                "config": self.config.model_dump()}
