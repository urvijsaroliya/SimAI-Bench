"""
Direct MPI transport — no store, no mediation.

Every other backend is store-mediated: even node-local is POSIX I/O into
tmpfs, so the interconnect never appears as itself. This one sends the payload
rank-to-rank, so intra-node links / shared memory and inter-node direct transfers are
measured directly rather than through a filesystem.

Five things follow from MPI being point-to-point rather than a store, and they
are semantic differences, not implementation details:

1. **The producer must know its consumers.** A store lets a consumer poll a key
   the producer never addressed. Here `consumers` is config.

2. **Keys carry the producer rank.** `stage_read(key)` has no source argument,
   so the source is parsed from the key: field 1 of `<prefix>_<rank>_<...>`.
   Matches the convention the drivers already use.

3. **Bulk payloads are sent with the buffer protocol**, not pickled — that is
   the point. The receiver must therefore know shape and dtype in advance, so
   they are config. Keys outside `bulk_prefixes` fall back to pickled
   `isend`/`recv` for small control messages.

4. **Backpressure is real above the eager threshold, and only there.** A
   payload large enough to use rendezvous does not complete until the consumer
   posts a matching receive, so `max_outstanding` bounds how far ahead the
   producer may run. A payload small enough to be sent eagerly is copied into
   MPI's own buffers and completes immediately, so the window never fills and
   the knob does nothing.

   Measured intra-node with a window of 2 and 40 publishes into a consumer
   that never reads (`tests/test_mpi_datastore.py`, job 1620334):

       16 B  (eager)       0 dropped  - window never binds
       1 MB  (rendezvous) 38 dropped  - window binds as intended

   So `max_outstanding` is a buffer-depth knob for bulk payloads and inert for
   control messages. Untested inter-node.

5. **Lossless and keep-last-1 need opposite behaviour at a full window.** A
   lossless stream must block, or data is lost. A keep-last-1 stream must
   *drop*: the version being held up is about to be superseded, so stalling the
   producer to deliver it buys nothing. `drop_prefixes` selects the second and
   `dropped` counts the discards. A store makes the same distinction invisibly —
   overwriting a key silently discards whatever was there — so this is not a
   new policy, only one that point-to-point forces you to state.

   In a closed loop the difference is not merely wasteful but fatal: a producer
   blocked on a full window stops consuming the reverse direction, the consumer's
   own sends then back up, and it can no longer reach the read that would have
   drained the first queue.

**The caller owns a written buffer until the send completes.** `stage_write` is
non-blocking and `np.ascontiguousarray` returns the input unchanged when it is
already contiguous and of the right dtype, so no copy is made: MPI reads the
caller's array asynchronously, and for a rendezvous payload it may do so long
after the call returned. A producer that reuses one buffer across sends is
mutating memory with an Isend outstanding. Rotate through at least
`max_outstanding + 2` buffers (see `runs/coupled.py`).

Requires producer and consumer in one communicator. True when components are
ranks of a single launch (as in `runs/coupled.py`); not true of the upstream
Workflow launcher, which starts each component under its own mpirun and would
need an inter-communicator first.

Device buffers (CuPy arrays over CUDA-aware MPI, or NCCL) are the next step and
are what would expose NVLink specifically; with host buffers an intra-node
exchange goes through shared memory.
"""
import logging as logging_
import os
import socket
import time
import zlib
from typing import Any

import numpy as np

from .base import BaseDataStore, BaseServerManager

try:
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    MPI4PY_AVAILABLE = True
except ImportError:
    MPI4PY_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def _is_device(buf) -> bool:
    """CuPy and related libraries advertise device memory this way."""
    return hasattr(buf, "__cuda_array_interface__")


class DataStoreMPI(BaseDataStore):
    """Point-to-point MPI DataStore. See module docstring for the contract."""

    def __init__(self, name: str, server_info, logging: bool = False,
                 log_level: int = logging_.INFO, is_colocated: bool = False):
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is required for the mpi backend")
        if not MPI.Is_initialized():
            raise RuntimeError("MPI must be initialized before the mpi backend")

        super().__init__(name, server_info, logging, log_level, is_colocated)

        if isinstance(server_info, dict):
            cfg = server_info.get("config", server_info)
        else:
            cfg = BaseServerManager.deserialize(server_info)["config"]
        self.config = dict(cfg)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.consumers = [int(r) for r in self.config.get("consumers", [])]
        self.shape = tuple(self.config.get("shape", [319488]))
        self.dtype = np.dtype(self.config.get("dtype", "float32"))
        self.shapes = {str(k): tuple(v)
                       for k, v in (self.config.get("shapes", {}) or {}).items()}
        self.mpi_dtype = MPI._typedict[self.dtype.char]
        self.max_outstanding = int(self.config.get("max_outstanding", 8))
        self.bulk_prefixes = tuple(self.config.get("bulk_prefixes", ["input"]))
        # Which underscore-separated field of a key holds the producer rank.
        # Configurable because requiring the rank at a fixed position is an
        # unusual constraint to place on an interface that otherwise treats
        # keys as opaque, and a caller with its own naming should not have to
        # rewrite it to use this backend.
        self.key_rank_field = int(self.config.get("key_rank_field", 1))
        # Seconds a producer may stall on a full send window before saying so.
        self.stall_warn_s = float(self.config.get("stall_warn_s", 30.0))
        # Opt-in because it is collective; see _setup_client.
        self.verify_peers = bool(self.config.get("verify_peers", False))
        # Seconds clean() will wait for in-flight sends before cancelling them.
        self.clean_drain_s = float(self.config.get("clean_drain_s", 5.0))
        self.drop_prefixes = tuple(self.config.get("drop_prefixes", []))
        self.drop_max_outstanding = int(
            self.config.get("drop_max_outstanding", 0)) or self.max_outstanding
        self.dropped = 0

        # Device-resident payloads: the buffer stays in HBM and MPI is handed a
        # device pointer, so an intra-node exchange can use intra-node link and an
        # inter-node one directly transfer (e.g., RDMA), instead of staging through the host.
        # For cuda, requires a CUDA-aware MPI. 
        # TODO: Support for xpu and other device backends. For now, only CPU and CUDA are supported.
        self.device = str(self.config.get("device", "cpu"))
        if self.device not in ("cpu", "cuda"):
            raise NotImplementedError("mpi backend: device buffers only on cuda; use device='cpu' on xpu systems")
        if self.device == "cuda" and not CUPY_AVAILABLE:
            raise ImportError("device='cuda' needs cupy")

        ub = self.comm.Get_attr(MPI.TAG_UB)
        self._tag_mod = min(int(ub) if ub else 32767, 1 << 20)
        
        # Separate queues for the same reason the depths are separate: with one
        # shared queue, occupancy from one class charges the other's window.
        # A trainer publishing keep-last-1 weights alongside lossless control
        # messages must not have a weight update counted as "blocked" because
        # the lossless window is full - nor spuriously dropped because
        # lossless requests fill a shared list.
        # (request, buffer) pairs, not bare requests. Isend does not copy - see
        # the caller-ownership note above - so the buffer must stay reachable
        # until the send completes, or Python may collect it while MPI is still
        # reading. Holding it here also makes reuse detectable rather than
        # merely documented.
        self._pending = []          # outstanding lossless Isends
        self._pending_drop = []     # outstanding keep-last-1 Isends
        # tag -> key, to catch two keys hashing to one tag. Detection only:
        # resolving a collision by probing would need producer and consumer to
        # agree on the replacement, and they compute tags independently, so it
        # would take a collective exchange of the whole map. Raising is honest
        # about that; silently matching the wrong message is not.
        self._tags = {}

        self._setup_client()

    def _setup_client(self):
        size = self.comm.Get_size()

        # A rank number only means something inside one communicator. Components
        # launched separately each get their own COMM_WORLD, where the same
        # number refers to a different process entirely - so an out-of-range
        # consumer is the one symptom of that mistake this backend can see
        # locally, and it is worth refusing rather than addressing a stranger.
        bad = [r for r in self.consumers if not 0 <= r < size]
        if bad:
            raise ValueError(
                f"mpi backend: consumer rank(s) {bad} lie outside COMM_WORLD "
                f"(size {size}). Producer and consumer must be ranks of the "
                f"same launch. If each component is started under its own "
                f"mpirun they do not share a communicator, and this backend "
                f"cannot reach across that - use a store-backed backend, or "
                f"launch the components as one MPI job.")

        if self.verify_peers:
            self._verify_peers()

        if self.logger:
            self.logger.debug(f"mpi datastore rank {self.rank} "
                              f"consumers {self.consumers} shape {self.shape}")

    def _wire_format(self) -> tuple:
        """Everything a receiver must already agree on to decode a message."""
        # device belongs here: stage_read allocates a host or device buffer
        # from this config, so a cuda producer against a cpu consumer is a wire
        # mismatch like any other - and one MPI will not complain about.
        return (str(self.dtype), tuple(self.shape),
                tuple(sorted((k, tuple(v)) for k, v in self.shapes.items())),
                tuple(self.bulk_prefixes), tuple(self.drop_prefixes),
                self.key_rank_field, self.device)

    def _verify_peers(self):
        """Check declared consumers exist and agree on the wire format.

        A receiver allocates from its own shape and dtype before it has seen
        the message, so a producer/consumer mismatch is not an MPI error - it
        truncates or corrupts silently. One allgather at setup turns that into
        a startup failure.

        **This is collective over COMM_WORLD**, which is why it is opt-in: it
        requires every rank to construct the datastore, the same number of
        times. A workflow where one component builds two DataStore objects and
        another builds one will deadlock here rather than in the exchange.
        """
        ident = {"host": socket.gethostname(), "pid": os.getpid(),
                 "rank": self.rank, "wire": self._wire_format()}
        peers = self.comm.allgather(ident)
        mine = ident["wire"]
        bad = [(r, peers[r]["wire"]) for r in self.consumers
               if peers[r]["wire"] != mine]
        if bad:
            detail = "; ".join(f"rank {r} on {peers[r]['host']} has {w}"
                               for r, w in bad)
            raise ValueError(
                f"mpi backend: rank {self.rank} would send with wire format "
                f"{mine} but {detail}. The receiver sizes its buffer from its "
                f"own config, so this corrupts data rather than raising.")

    # ---- key contract ----------------------------------------------------
    def _tag(self, key: str) -> int:
        tag = zlib.crc32(key.encode()) % self._tag_mod
        seen = self._tags.setdefault(tag, key)
        if seen != key:
            raise ValueError(
                f"mpi backend: {key!r} and {seen!r} both hash to tag {tag}. "
                f"Recv matches on (source, tag), so one would silently return "
                f"the other's payload. Rename one key, or raise TAG_UB if the "
                f"MPI allows it (tag space here is {self._tag_mod}).")
        return tag

    def _src(self, key: str) -> int:
        """Producer rank, read from field `key_rank_field` of the key.

        Point-to-point has no directory to look a producer up in, so the
        source has to come from somewhere; `stage_read` has no argument for
        it, which leaves the key.
        """
        try:
            return int(key.split("_")[self.key_rank_field])
        except (IndexError, ValueError):
            raise ValueError(
                f"mpi backend reads the producer rank from field "
                f"{self.key_rank_field} of the key (0-based, split on '_'), "
                f"so keys must look like <prefix>_<rank>_... - got {key!r}. "
                f"Set key_rank_field if your keys are shaped differently.")

    def _is_bulk(self, key: str) -> bool:
        return key.startswith(self.bulk_prefixes)

    def _is_drop(self, key: str) -> bool:
        return bool(self.drop_prefixes) and key.startswith(self.drop_prefixes)

    def _shape_for(self, key: str) -> tuple:
        for pfx, shp in self.shapes.items():
            if key.startswith(pfx):
                return shp
        return self.shape

    def _reap(self, block: bool = False):
        """Retire completed sends; bound how far a producer may run ahead."""
        if block:
            for lst in (self._pending, self._pending_drop):
                if lst:
                    MPI.Request.Waitall([rb[0] for rb in lst])
            self._pending, self._pending_drop = [], []
            return
        self._pending = [rb for rb in self._pending if not rb[0].Test()]
        self._pending_drop = [rb for rb in self._pending_drop
                              if not rb[0].Test()]

    # ---- DataStore interface ---------------------------------------------
    def stage_write(self, key: str, data: Any, persistant: bool = True,
                    client_id: int = 0, is_local: bool = False):
        tag = self._tag(key)
        self._reap()
        drop = self._is_drop(key)
        pending = self._pending_drop if drop else self._pending
        depth = self.drop_max_outstanding if drop else self.max_outstanding
        cap = depth * max(len(self.consumers), 1)
        # Rendezvous means a bulk send completes only once matched, so an
        # unbounded queue here would hide exactly the backpressure we want.
        stall_since, warned, spins = None, False, 0
        while len(pending) >= cap:
            self._reap()
            pending = self._pending_drop if drop else self._pending
            if len(pending) < cap:
                break
            if drop:
                # keep-last-1 on a bounded transport means dropping a version
                # the consumer has no room for - not stalling to deliver one
                # that is about to be superseded anyway. A store does this
                # implicitly on every overwrite; point-to-point has to say it
                # out loud.
                #
                # Blocking here is lossless semantics applied to a lossy path,
                # and in a closed loop it deadlocks: the producer stops
                # consuming the reverse direction while it waits, so the
                # consumer's own sends back up until it can no longer reach the
                # read that would have drained this queue.
                # E.g., stale weights aren't worth waiting for, and in a loop, 
                # waiting for them can deadlock the whole workflow — so drop.
                self.dropped += 1
                return
            # Waitany would block here indefinitely and silently. A lossless
            # producer only stalls because its consumer is not receiving, and
            # the usual cause is send and receive counts that do not balance -
            # which otherwise presents as a job that hangs for its whole
            # walltime with no output to diagnose from. Poll instead, so the
            # stall can be named.
            if stall_since is None:
                stall_since = time.perf_counter()
            elif not warned and (time.perf_counter() - stall_since
                                 > self.stall_warn_s):
                warned = True
                msg = (f"mpi backend: rank {self.rank} has been waiting "
                       f"{self.stall_warn_s:.0f}s for a send slot on {key!r} "
                       f"({len(pending)} of {cap} outstanding). Every "
                       f"rendezvous send must be matched by a receive; if the "
                       f"consumer reads fewer messages than the producer "
                       f"writes, this never clears.")
                if self.logger:
                    self.logger.warning(msg)
                else:
                    print(msg, flush=True)
            # Back off, do not sleep a fixed interval. A rendezvous transfer
            # needs *this* rank to make progress, so any sleep here lands on
            # the consumer's Recv: a flat 200 us sleep cost 5x on read latency
            # (0.051 -> 0.259 ms per 1 MB read, bisected in job 1619893).
            # Spin first so the common short wait pays nothing, then grow the
            # interval, since a stall worth warning about lasts seconds.
            if spins < 200:
                spins += 1
            else:
                time.sleep(min(1e-5 * 2 ** ((spins - 200) // 50), 1e-3))
                spins += 1
            self._reap()
            pending = self._pending_drop if drop else self._pending

        for dest in self.consumers:
            if dest == self.rank:
                continue
            if self._is_bulk(key):
                if _is_device(data):
                    # never np.ascontiguousarray a device array: it would copy
                    # to host and silently undo the point of this path
                    buf = cp.ascontiguousarray(data)
                    cp.cuda.get_current_stream().synchronize()   # kernels done before MPI reads
                else:
                    buf = np.ascontiguousarray(data, dtype=self.dtype)
                # Reusing a buffer with a send outstanding corrupts the
                # payload silently; it cost real measurements before it was
                # found (see the caller-ownership note above), so say so.
                if any(b is buf for _, b in
                       (self._pending_drop if drop else self._pending)):
                    m = (f"mpi backend: rank {self.rank} wrote {key!r} from a "
                         f"buffer that still has a send in flight. MPI reads "
                         f"it asynchronously, so overwriting it corrupts the "
                         f"earlier message. Rotate through at least "
                         f"max_outstanding + 2 buffers.")
                    self.logger.warning(m) if self.logger else print(m, flush=True)
                (self._pending_drop if drop else self._pending).append(
                    (self.comm.Isend([buf, self.mpi_dtype], dest=dest, tag=tag),
                     buf))
            else:
                # small control message: eager, completes without a match
                (self._pending_drop if drop else self._pending).append(
                    (self.comm.isend(data, dest=dest, tag=tag), data))

    def poll_staged_data(self, key: str, client_id: int = 0,
                         is_local: bool = False) -> bool:
        self._reap()                       # also drives MPI progress
        return self.comm.Iprobe(source=self._src(key), tag=self._tag(key))

    def _await(self, req, timeout, what: str):
        """Complete `req`, or cancel it and raise once `timeout` has passed.

        Only reached when the message has not arrived yet - see stage_read for
        why the arrived case must not come through here. Spins before sleeping
        so a message landing shortly after the post is not charged a full sleep
        interval.
        """
        if not timeout or timeout <= 0:
            req.Wait()
            return
        deadline = time.perf_counter() + timeout
        spins = 0
        while not req.Test():
            spins += 1
            if spins > 2000:
                time.sleep(2e-4)
            if time.perf_counter() > deadline:
                req.Cancel()
                req.Wait()          # a cancelled request must still be completed
                raise TimeoutError(
                    f"mpi backend: no {what} within {timeout}s. Either the "
                    f"producer never sent it, or the send and receive counts "
                    f"do not balance - every rendezvous send must be matched.")

    def stage_read(self, key: str, client_id: int = 0, timeout: int = 30,
                   is_local: bool = False):
        src, tag = self._src(key), self._tag(key)
        if self._is_bulk(key):
            shape = self._shape_for(key)
            xp = cp if self.device == "cuda" else np
            buf = xp.empty(shape, dtype=self.dtype)
            # Probe before posting. If the envelope is already here the
            # receive will complete, so hand it to MPI once and let it block
            # rather than testing for completion from Python; callers poll
            # before reading, so this is the usual path. The deadline below
            # exists for the case that is genuinely still waiting.
            #
            # The extra Iprobe was not measurable either way (poll 0.0123 ->
            # 0.0125 ms). It is kept because blocking once is more direct than
            # a spin loop and does not burn a core while it waits, not because
            # it was shown to be faster.
            if self.comm.Iprobe(source=src, tag=tag):
                self.comm.Recv([buf, self.mpi_dtype], source=src, tag=tag)
            else:
                self._await(
                    self.comm.Irecv([buf, self.mpi_dtype], source=src, tag=tag),
                    timeout, f"{key!r} from rank {src}")
            if self.device == "cuda":
                cp.cuda.get_current_stream().synchronize()
            return buf
        # Pickled control message: recv sizes itself from the envelope, so probe
        # for the envelope rather than posting a receive of unknown size.
        if timeout and timeout > 0 and not self.comm.Iprobe(source=src, tag=tag):
            deadline = time.perf_counter() + timeout
            while not self.comm.Iprobe(source=src, tag=tag):
                self._reap()                # keep our own sends progressing
                if time.perf_counter() > deadline:
                    raise TimeoutError(
                        f"mpi backend: no {key!r} from rank {src} within "
                        f"{timeout}s.")
                time.sleep(2e-4)
        return self.comm.recv(source=src, tag=tag)

    def clean_staged_data(self, key: str, client_id: int = 0,
                          is_local: bool = False):
        """No store to clean: a received message is already gone."""
        return

    def clean(self):
        """Retire outstanding sends, waiting for them before giving up.

        Cancelling is the only way to reach Finalize with sends outstanding,
        but a cancelled send is data a consumer was waiting for. Doing that
        while it is still draining starves it - jobs 1419252 and 1419253 died
        exactly so, the producer cancelling 176 of 400 weight sends out from
        under a solver that then sat out its own drain timeout.

        So wait first, bounded by `clean_drain_s`. Ordering clean() after every
        consumer has finished is still the caller's job; this only stops a
        small skew in teardown order from destroying data.
        """
        self._reap()
        deadline = time.perf_counter() + self.clean_drain_s
        while self._pending or self._pending_drop:
            if time.perf_counter() > deadline:
                break
            time.sleep(1e-4)
            self._reap()

        # Whatever is left after waiting really is abandoned.
        n = len(self._pending) + len(self._pending_drop)
        if n:
            msg = (f"mpi backend: rank {self.rank} discarding {n} unmatched "
                   f"send(s) after waiting {self.clean_drain_s:.0f}s - a "
                   f"consumer expected data it will now never receive. Either "
                   f"a receive was never posted, or clean() ran before the "
                   f"consumer finished draining.")
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg, flush=True)
        for r, _ in self._pending + self._pending_drop:
            try:
                r.Cancel()
            except Exception:
                pass
        self._pending, self._pending_drop = [], []

class ServerManagerMPI(BaseServerManager):
    """No server exists; the communicator is the transport."""

    def start_server(self):
        if self.logger:
            self.logger.info("mpi backend: no server to start")

    def stop_server(self):
        if self.logger:
            self.logger.info("mpi backend: no server to stop")

    def get_server_info(self) -> dict:
        return {"name": self.name, "type": self.config.type,
                "config": self.config.model_dump()}
