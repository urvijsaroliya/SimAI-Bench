#!/usr/bin/env python3
"""Two-rank tests for the direct MPI DataStore.

Run under any MPI launcher with exactly 2 ranks:

    srun -n 2 python tests/test_mpi_datastore.py
    mpirun -n 2 python tests/test_mpi_datastore.py

Not a pytest module: the backend needs a real communicator, and pytest does
not launch one. Exit status is non-zero if any rank fails, so it drops into
CI or a job script unchanged.

Rank 0 produces, rank 1 consumes. Every test uses distinct keys and ends with
a barrier, because an unmatched send from a failing test would otherwise be
picked up by the next one and turn one failure into a cascade.
"""
import os
import sys
import time
import traceback

import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# MPI_DS_MODULE lets a run point at an alternate copy of the backend, so two
# implementations can be compared inside one allocation without swapping files
# in the working tree. Loaded under a dotted name so its relative imports still
# resolve against the real package.
_alt = os.environ.get("MPI_DS_MODULE")
if _alt:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "SimAIBench.datastore._mpi_alt", _alt)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)
    DataStoreMPI = _mod.DataStoreMPI
else:
    from SimAIBench.datastore.mpi import DataStoreMPI

PRODUCER, CONSUMER = 0, 1


def store(rank, **overrides):
    cfg = {"consumers": [CONSUMER] if rank == PRODUCER else [],
           "shape": [16], "dtype": "float32", "bulk_prefixes": ["input"]}
    cfg.update(overrides)
    return DataStoreMPI(f"t{rank}", {"config": cfg})


# --- tests ----------------------------------------------------------------
def test_bulk_roundtrip(comm, rank):
    """Contents must survive, not just arrive."""
    ds = store(rank)
    key = f"input_{PRODUCER}_0"
    if rank == PRODUCER:
        ds.stage_write(key, np.arange(16, dtype=np.float32))
    else:
        while not ds.poll_staged_data(key):
            time.sleep(1e-3)
        got = ds.stage_read(key)
        assert np.array_equal(got, np.arange(16, dtype=np.float32)), got
    comm.Barrier()


def test_control_message(comm, rank):
    """Keys outside bulk_prefixes travel pickled and may be any object."""
    ds = store(rank)
    key = f"done_{PRODUCER}"
    if rank == PRODUCER:
        ds.stage_write(key, {"steps": 7})
    else:
        while not ds.poll_staged_data(key):
            time.sleep(1e-3)
        assert ds.stage_read(key) == {"steps": 7}
    comm.Barrier()


def test_per_prefix_shape(comm, rank):
    """A rank exchanging two payload kinds must size each correctly."""
    ds = store(rank, bulk_prefixes=["input", "weights"],
               shapes={"weights": [4]})
    if rank == PRODUCER:
        ds.stage_write(f"input_{PRODUCER}_1", np.ones(16, dtype=np.float32))
        ds.stage_write(f"weights_{PRODUCER}", np.full(4, 3.0, dtype=np.float32))
    else:
        for key, n, val in ((f"input_{PRODUCER}_1", 16, 1.0),
                            (f"weights_{PRODUCER}", 4, 3.0)):
            while not ds.poll_staged_data(key):
                time.sleep(1e-3)
            got = ds.stage_read(key)
            assert got.shape == (n,), (key, got.shape)
            assert np.allclose(got, val), (key, got[:4])
    comm.Barrier()


def test_timeout_raises(comm, rank):
    """timeout is part of the DataStore contract; it must actually bound."""
    ds = store(rank)
    if rank == CONSUMER:
        t0 = time.perf_counter()
        try:
            ds.stage_read(f"input_{PRODUCER}_99", timeout=0.4)
        except TimeoutError:
            dt = time.perf_counter() - t0
            assert 0.3 < dt < 3.0, f"returned after {dt:.2f}s, expected ~0.4"
        else:
            raise AssertionError("no TimeoutError for a message never sent")
    comm.Barrier()


def test_rank_range_checked(comm, rank):
    """A consumer outside COMM_WORLD means separate launches; refuse it."""
    try:
        store(rank, consumers=[comm.Get_size() + 5])
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range consumer accepted")
    comm.Barrier()


def test_key_rank_field(comm, rank):
    """The producer rank need not sit at field 1."""
    ds = store(rank, key_rank_field=2, bulk_prefixes=["fld"])
    key = f"fld_x_{PRODUCER}_0"
    if rank == PRODUCER:
        ds.stage_write(key, np.full(16, 2.0, dtype=np.float32))
    else:
        while not ds.poll_staged_data(key):
            time.sleep(1e-3)
        assert np.allclose(ds.stage_read(key), 2.0)
    comm.Barrier()


def _drop_case(comm, rank, n_elem, label):
    """keep-last-1 discards rather than blocking, and says how many."""
    ds = store(rank, bulk_prefixes=["input", "weights"],
               shapes={"weights": [n_elem]},
               drop_prefixes=["weights"], drop_max_outstanding=2)
    if rank == PRODUCER:
        # Rotate buffers: a send in flight still owns the one it was given.
        bufs = [np.full(n_elem, i, dtype=np.float32) for i in range(12)]
        t0 = time.perf_counter()
        for i in range(40):
            ds.stage_write(f"weights_{PRODUCER}", bufs[i % 12])
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"publishing blocked for {elapsed:.1f}s"
        sent = 40 - ds.dropped
        # Reported, not asserted: whether the window binds at all is a platform
        # property, not a defect. 40 rapid publishes into a window of 2, with a
        # consumer that is not reading, should overflow it - unless MPI buffers
        # them internally, in which case max_outstanding is not a buffer-depth
        # knob on this transport. That is EXPERIMENTS.md open question 12, and
        # this line answers it in seconds instead of an allocation.
        print(f"    [drop] {label}: window=2, 40 publishes -> {ds.dropped} "
              f"dropped, {sent} sent"
              f"{'  <-- window did NOT bind (see OQ12)' if not ds.dropped else ''}",
              flush=True)
        comm.send(sent, dest=CONSUMER, tag=9001)
    else:
        sent = comm.recv(source=PRODUCER, tag=9001)
        got = 0
        deadline = time.perf_counter() + 20
        while got < sent and time.perf_counter() < deadline:
            if ds.poll_staged_data(f"weights_{PRODUCER}"):
                ds.stage_read(f"weights_{PRODUCER}")
                got += 1
            else:
                time.sleep(1e-4)
        assert got == sent, f"producer sent {sent}, consumer got {got}"
    comm.Barrier()


def test_drop_when_full(comm, rank):
    """Eager-sized payload: small enough that MPI buffers it internally."""
    _drop_case(comm, rank, 4, "eager 16 B")


def test_drop_when_full_rendezvous(comm, rank):
    """Rendezvous-sized payload.

    The question OQ12 actually cares about. A 16 B payload is buffered
    internally whatever max_outstanding says; 1 MB should exceed MPI's
    buffering and let the window bind - if it does not, the knob does nothing
    on this transport at any size, and the buffer-policy axis needs a
    different mechanism entirely.
    """
    _drop_case(comm, rank, 262144, "rendezvous 1 MB")


def test_clean_waits_for_drain(comm, rank):
    """clean() must not cancel sends a consumer is still collecting.

    The producer tears down immediately while the consumer is deliberately
    slow. Before clean() waited, this is the shape that destroyed jobs 1419252
    and 1419253: the producer cancelled in-flight sends and the consumer sat
    out its drain timeout waiting for data that no longer existed.
    """
    n_elem, N = 262144, 6
    ds = store(rank, shape=[n_elem])
    key = f"input_{PRODUCER}_drain"
    if rank == PRODUCER:
        bufs = [np.full(n_elem, i, dtype=np.float32) for i in range(N)]
        for b in bufs:
            ds.stage_write(key, b)
        ds.clean()                      # tear down while the consumer lags
    else:
        time.sleep(0.3)                 # lag on purpose
        got = 0
        deadline = time.perf_counter() + 20
        while got < N and time.perf_counter() < deadline:
            if ds.poll_staged_data(key):
                ds.stage_read(key)
                got += 1
            else:
                time.sleep(1e-4)
        assert got == N, f"producer sent {N}, consumer got {got} after clean()"
    comm.Barrier()


def test_device_roundtrip(comm, rank):
    """Device-resident payload must survive HBM to HBM with contents intact.

    F13 rests on this path and nothing exercised it. Contents are checked, not
    just arrival: a non-CUDA-aware MPI corrupts or segfaults rather than
    raising, so "it returned something" proves nothing.
    """
    try:
        import cupy as cp
    except ImportError:
        if rank == PRODUCER:
            print("    [skip] device roundtrip: no cupy", flush=True)
        comm.Barrier()
        return

    n = 4096
    ds = store(rank, device="cuda", shape=[n])
    key = f"input_{PRODUCER}_dev"
    expect = cp.arange(n, dtype=cp.float32)
    if rank == PRODUCER:
        ds.stage_write(key, expect.copy())
    else:
        while not ds.poll_staged_data(key):
            time.sleep(1e-3)
        got = ds.stage_read(key)
        assert type(got).__module__.split(".")[0] == "cupy", type(got)
        assert bool((got == expect).all()), "device payload corrupted"
    comm.Barrier()


def test_device_mismatch_caught(comm, rank):
    """A cuda producer against a cpu consumer is a wire mismatch.

    verify_peers must reject it: stage_read sizes and *places* its buffer from
    its own config, so the consumer would allocate host memory for a device
    send and MPI would not object.
    """
    try:
        import cupy  # noqa: F401
    except ImportError:
        comm.Barrier()
        return
    try:
        store(rank, device="cuda" if rank == PRODUCER else "cpu",
              verify_peers=True)
    except ValueError:
        pass                      # producer rejects; consumer declares nobody
    comm.Barrier()


def test_drain_completeness_report(comm, rank):
    """Does a drain-to-latest loop stop while newer versions are still queued?

    **UNRELIABLE - do not read the number as evidence.** Two attempts produced
    3 and then 1 drain boundary out of 120 messages, so the re-poll rate is
    computed over a handful of samples and rules nothing out. Adding a pause
    between passes made it worse, not better, which contradicts the model
    behind the test; the inner loop appears never to see an empty queue and it
    is not understood why. Fixing it needs per-drain diagnostics (sizes,
    timings, depth at entry), not another guess.

    Left in place because the hypothesis is still live for F19's staleness
    tail, and because a broken probe that says so is worth more than a deleted
    one. It asserts nothing, so it cannot fail the suite.

    Reported, not asserted. This is the surviving hypothesis for F19's
    staleness tail: the drain is Iprobe-driven, so it may report empty while a
    message is sent but not yet visible, leaving the consumer holding an older
    version than exists.

    Distinguishing "was there and missed" from "arrived just now" is the whole
    difficulty. The control is timing: the producer paces at PACE, and the
    re-poll takes microseconds, so if the drain were honest the re-poll should
    hit only about (repoll_time / PACE) of the time. A hit rate far above that
    is the drain stopping early.
    """
    # GAP stands in for the solver's kappa steps of compute between
    # acquisitions. Without it the inner loop never exits under rendezvous
    # backpressure - producer and consumer lock-step and 120 messages collapse
    # into 3 drain boundaries, which samples nothing.
    n_elem, N, PACE, GAP = 262144, 120, 2e-3, 8e-3
    ds = store(rank, shape=[n_elem])
    key, done = f"input_{PRODUCER}_dr", f"done_{PRODUCER}"
    if rank == PRODUCER:
        bufs = [np.zeros(n_elem, dtype=np.float32) for _ in range(12)]
        for i in range(1, N + 1):
            b = bufs[i % 12]
            b[0] = i
            ds.stage_write(key, b)
            time.sleep(PACE)
        ds.stage_write(done, N)
    else:
        drains = early = seen = 0
        repoll_t = 0.0
        deadline = time.perf_counter() + 60
        while time.perf_counter() < deadline:
            got_any = False
            while ds.poll_staged_data(key):
                ds.stage_read(key)
                seen += 1
                got_any = True
            if got_any:
                drains += 1
                t0 = time.perf_counter()
                hit = ds.poll_staged_data(key)
                repoll_t += time.perf_counter() - t0
                early += bool(hit)
            if seen >= N and ds.poll_staged_data(done):
                ds.stage_read(done)
                break
            time.sleep(GAP)
        rate = early / drains * 100 if drains else 0.0
        # Chance of a genuine arrival inside the re-poll window itself.
        chance = (repoll_t / drains) / PACE * 100 if drains else 0.0
        print(f"    [drain] {seen}/{N} received, {drains} drains, "
              f"re-poll hit {rate:.1f}% vs {chance:.2f}% expected by arrival "
              f"timing alone"
              f"  [UNRELIABLE: {drains} drain boundaries is too few to read]",
              flush=True)
    comm.Barrier()


def test_verify_peers_agrees(comm, rank):
    """Collective wire-format check passes when both sides match."""
    store(rank, verify_peers=True)
    comm.Barrier()


def test_tags_distinct(comm, rank):
    """Local check: a realistic key set must not collide on tags."""
    ds = store(rank)
    keys = ([f"input_{r}_{i}" for r in range(4) for i in range(500)]
            + [f"weights_{r}" for r in range(4)]
            + [f"done_{r}" for r in range(4)])
    for k in keys:
        ds._tag(k)              # raises on collision; registry lives in ds
    assert len(ds._tags) == len(set(keys)), "registry lost a key"

    # A collision must raise, not be silently matched. Two keys sharing a tag
    # are undetectable at the receiver: Recv matches on (source, tag) alone and
    # would hand back the other message's payload. Plant one and check.
    victim = f"input_{PRODUCER}_0"
    ds._tags[ds._tag(victim)] = "some_other_key"
    try:
        ds._tag(victim)
    except ValueError as e:
        assert "hash to tag" in str(e), e
    else:
        raise AssertionError("collision not detected")
    comm.Barrier()


def test_read_cost_report(comm, rank):
    """Report per-read cost for a coupled-sized payload.

    Reported, never asserted: this is a bisect handle, not a threshold. The
    closed-loop runs measure ~0.13-0.15 ms per 1 MB return-path read, and a
    change of that size currently costs a 20-minute coupled job to detect.
    Here it costs seconds, which is the difference between bisecting a
    regression and theorising about one.

    Buffers rotate because a non-blocking send borrows the caller's array
    until it completes - reusing one here would measure corruption.
    """
    n_elem, N, depth = 262144, 100, 12          # 1 MB, matching the weights payload
    ds = store(rank, shape=[n_elem])
    key = f"input_{PRODUCER}_b"
    if rank == PRODUCER:
        bufs = [np.full(n_elem, i, dtype=np.float32) for i in range(depth)]
        for i in range(N):
            ds.stage_write(key, bufs[i % depth])
    else:
        ts = []
        for _ in range(N):
            while not ds.poll_staged_data(key):
                time.sleep(1e-5)
            t0 = time.perf_counter()
            ds.stage_read(key)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        print(f"    [bench] 1 MB read  p50 {ts[len(ts) // 2] * 1e3:.4f} ms  "
              f"p95 {ts[int(len(ts) * 0.95)] * 1e3:.4f} ms  (n={N})", flush=True)
    comm.Barrier()


TESTS = [test_bulk_roundtrip, test_control_message, test_per_prefix_shape,
         test_timeout_raises, test_rank_range_checked, test_key_rank_field,
         test_drop_when_full, test_drop_when_full_rendezvous,
         test_clean_waits_for_drain, test_device_roundtrip,
         test_device_mismatch_caught, test_drain_completeness_report,
         test_verify_peers_agrees, test_tags_distinct,
         test_read_cost_report]


def main():
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != 2:
        if rank == 0:
            print(f"needs exactly 2 ranks, got {size}")
        MPI.Finalize()
        return 2

    # MPI_DS_ONLY selects a subset by name. Needed when pointing MPI_DS_MODULE
    # at an older backend: a test for a feature that revision lacks does not
    # fail there, it hangs (test_timeout_raises would wait forever on code that
    # ignores the timeout), and a hang is exactly what these tests exist to
    # stop costing us.
    only = os.environ.get("MPI_DS_ONLY")
    selected = [t for t in TESTS
                if not only or t.__name__ in only.split(",")]

    failures = []
    for t in selected:
        try:
            t(comm, rank)
            ok = True
        except Exception:
            ok = False
            failures.append((t.__name__, traceback.format_exc()))
            comm.Barrier()          # keep ranks in step after a failure
        results = comm.allgather(ok)
        if rank == 0:
            print(f"  {'PASS' if all(results) else 'FAIL'}  {t.__name__}")

    allfail = comm.allgather(failures)
    rc = 0
    if rank == 0:
        flat = [f for per_rank in allfail for f in per_rank]
        print(f"\n{len(selected) - len({n for n, _ in flat})}/{len(selected)} passed")
        for name, tb in flat:
            print(f"\n--- {name} ---\n{tb}")
        rc = 1 if flat else 0
    rc = comm.bcast(rc, root=0)
    MPI.Finalize()
    return rc


if __name__ == "__main__":
    sys.exit(main())
