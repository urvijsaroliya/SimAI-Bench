from pydantic import BaseModel, Field
import multiprocessing as mp
from typing import Literal, Dict, Any, Type, List
from difflib import get_close_matches


class SystemConfig(BaseModel):
    """Input configuration of the system"""
    name: str
    ncpus: int = mp.cpu_count()
    ngpus: int = 0
    cpus: List = Field(default_factory=list)
    gpus: List = Field(default_factory=list)


class OchestratorConfig(BaseModel):
    """
    Configuration for the Orchestrator.
    Attributes:
        name (str): The name of the executor. Defaults to "process-pool".
        submit_loop_sleep_time (int): Time in seconds for the orchestrator to sleep between submission loops. Defaults to 10.
        listening_time (int): Time in seconds for the orchestrator to listen for incoming tasks or events. Defaults to 300.
    """
    name: Literal["ray","process-pool","parsl-local","parsl-htex","taps-htex","dask","thread-pool","dragon","taskvine","dragon"] = "process-pool"
    submit_loop_sleep_time: int = 10
    listening_time: int = 300
    profile: bool = False

class ServerConfig(BaseModel):
    type: str
    server_address: str


class ServerConfigRegistry:
    """Registry for server configuration classes"""
    
    def __init__(self):
        self._configs: Dict[str, Type[ServerConfig]] = {}
    
    def register(self, server_type: str):
        """Decorator to register server config classes"""
        def decorator(cls: Type[ServerConfig]):
            self._configs[server_type] = cls
            return cls
        return decorator
    
    def get_config_class(self, server_type: str) -> Type[ServerConfig]:
        """Get the appropriate server config class for a given type"""
        if server_type not in self._configs:
            # Find the closest match using edit distance
            available_types = list(self._configs.keys())
            suggestions = get_close_matches(server_type, available_types, n=1, cutoff=0.6)
            
            error_msg = f"Unknown server type: {server_type}. "
            if suggestions:
                error_msg += f"Did you mean: {', '.join(suggestions)}?"
            else:
                error_msg += f"Available types: {', '.join(available_types)}"
            
            raise ValueError(error_msg)
        return self._configs[server_type]
    
    def create_config(self, type: str, **kwargs) -> ServerConfig:
        """Create a server config instance for a given type"""
        config_class = self.get_config_class(type)
        return config_class(type=type, **kwargs)
    
    def list_types(self) -> list[str]:
        """List all available server types"""
        return list(self._configs.keys())
    
    def is_registered(self, server_type: str) -> bool:
        """Check if a server type is registered"""
        return server_type in self._configs


# Create global registry instance
server_registry = ServerConfigRegistry()

# Register existing server configs
@server_registry.register("filesystem")
@server_registry.register("node-local")
class FilesystemServerConfig(ServerConfig):
    type: Literal["filesystem", "node-local"] = "filesystem"
    server_address: str = "./.tmp"
    nshards: int = 64

@server_registry.register("redis")
class RedisServerConfig(ServerConfig):
    type: Literal["redis"] = "redis"
    server_address: str = "localhost:6379"
    redis_server_exe: str = "redis-server"
    is_clustered: bool = False

@server_registry.register("dragon")
class DragonServerConfig(ServerConfig):
    type: Literal["dragon"] = "dragon"
    server_address: str = "localhost:8888"
    server_options: Dict[str, Any] = Field(default_factory=dict)

@server_registry.register("daos")
class DaosServerConfig(ServerConfig):
    type: Literal["daos"] = "daos"
    server_address: str = "/path/to/dfuse/mount"
    mode: Literal["posix", "kv"] = "posix"
    nshards: int = 64

@server_registry.register("mpi")
class MPIServerConfig(ServerConfig):
    """Direct point-to-point MPI. No server; the communicator is the transport."""
    type: Literal["mpi"] = "mpi"
    server_address: str = "comm-world" # unused; base class requires it
    
    # With a store, senders don't need to know who reads; with direct sends, they must.
    consumers: List[int] = Field(default_factory=list)
    
    # A receiver must have a buffer ready before data arrives, 
    # so sizes can't be discovered on the fly like store-mediated scenarios. 
    # shape is the default size; shapes lets specific message types override it.
    shape: List[int] = Field(default_factory=lambda: [319488]) 
    shapes: Dict[str, List[int]] = Field(default_factory=dict)
    dtype: str = "float32"

    # Keys with these prefixes are sent as raw array bytes (no serialization, GPU-direct if enabled)
    # into the pre-posted buffers above; everything else is pickled per message.
    # Raw needs known sizes – fits large fixed-size simulation data, not small variable control messages.
    bulk_prefixes: List[str] = Field(default_factory=lambda: ["input"])
    
    # When the send window fills, default is to block (backpressure).
    # max_outstanding: how far a producer may run ahead of consumers before blocking.
    # drop_prefixes keys instead drop the oldest unsent – newest-wins,
    # for payloads where only the latest matters (e.g., weights).
    # drop_max_outstanding: own window depth for these; 0 = share.
    max_outstanding: int = 8 
    drop_prefixes: List[str] = Field(default_factory=list)
    drop_max_outstanding: int = 0
    # 0-based field of the key holding the producer rank (keys are
    # <prefix>_<rank>_...); point-to-point has no directory to look it up in
    key_rank_field: int = 1
    # seconds a producer may stall on a full send window before warning
    stall_warn_s: float = 30.0
    # collective wire-format check at setup; requires every rank to build
    # the datastore the same number of times, so it is off by default
    verify_peers: bool = False
    # seconds clean() waits for in-flight sends before cancelling them
    clean_drain_s: float = 5.0
    
    device: Literal["cpu", "cuda"] = "cpu" # cuda keeps payloads in HBM (needs CUDA-aware MPI)

@server_registry.register("rma")
class RMAServerConfig(ServerConfig):
    """Ring queue in MPI one-sided windows. No server; the windows are the transport.

    One ring per (producer, consumer, class), single-producer single-consumer,
    with HEAD/TAIL as uint64 atomics in the ring owner's control window. See
    `datastore/rma.py` for the protocol and for which of these knobs the 2c
    feasibility gate (job 4242310) fixed the default of.
    """
    type: Literal["rma"] = "rma"
    server_address: str = "comm-world"  # unused; base class requires it

    # A ticket queue addresses nobody, but the rings are still built from the
    # declared topology: one per pair, so this is what gets allocated.
    consumers: List[int] = Field(default_factory=list)

    # A consumer sizes its Get from its own config before it has seen the record,
    # so shape/shapes/dtype are a wire contract and are checked at setup.
    shape: List[int] = Field(default_factory=lambda: [319488])
    shapes: Dict[str, List[int]] = Field(default_factory=dict)
    dtype: str = "float32"
    bulk_prefixes: List[str] = Field(default_factory=lambda: ["input"])

    # M_q, v3 section 15.5's explicit ring capacity, in slots. Unlike the direct
    # path's max_outstanding this bounds the queue at EVERY payload size, not
    # only above the rendezvous switch (7194 B within a node, 4111 B across it;
    # F20 revised). drop_capacity: own depth for drop_prefixes; 0 = same.
    capacity: int = 8
    drop_capacity: int = 0
    # A full ring blocks for lossless keys and drops the new record for these,
    # exactly as the direct path does - blocking a keep-last-1 stream deadlocks a
    # closed loop. v3 section 15.5: the ring has no native keep-last-1, so a
    # consumer that wants only the newest version still drains to it.
    drop_prefixes: List[str] = Field(default_factory=list)
    # What a full drop-class ring does. "newest" discards the arriving record and
    # HOLES the index sequence, so it cannot serve a keep-last FORWARD path.
    # "oldest" laps: the producer advances TAIL past the unread record, the
    # sequence stays contiguous, and the consumer detects that it was lapped
    # (`lap_skipped`, `lap_torn`). Lapping breaks the invariant that TAIL moves
    # only after a payload has landed - that is what it is for - so it is opt-in.
    drop_mode: Literal["newest", "oldest"] = "newest"
    # refused up front rather than failing inside MPI: window creation is IB
    # memory registration on this build (M33 observation 3)
    max_window_bytes: int = 1 << 30

    # Which communicator the windows live on. "world" is the default because that
    # is where a coupled job's windows actually live - and M33 is that a
    # world-communicator window gets osc/ucx, which on this build has no
    # shared-memory lane: an intra-node atomic costs 0.0120-0.0128 ms against
    # osc/sm's 0.0038-0.0047. "node" (Split_type(COMM_TYPE_SHARED)) recovers most
    # of that and can only carry pairs that share a node, so it is a control arm.
    # The tiered arrangement is the author's open option and is not implemented.
    ring_comm: Literal["world", "node"] = "world"
    # Where the ring memory lives. "producer": the push is a local store (no RMA
    # at all) and the pop is a remote Get the consumer can overlap - which is the
    # only direction that overlaps within a node (M33). "consumer": the push is a
    # Put and the pop is a local load, i.e. v3 section 15.8 (i)'s zero-copy pop,
    # which the gate never timed.
    ring_owner: Literal["producer", "consumer"] = "producer"
    # compare-and-swap on TAIL asserts the ring really had one consumer; "faa" is
    # the same operation without the assertion, for a cell that prices it
    release_op: Literal["cas", "faa"] = "cas"
    # "on": poll_staged_data checks the head record's key exactly (one small Get)
    # so a poll for a key queued behind others answers honestly - every driver
    # path depends on that. "off": one atomic, trust FIFO, key_mismatch counts
    # when that trust was wrong.
    peek: Literal["on", "off"] = "on"
    # moves the payload Get from read_tot into poll_tot; for checking the
    # attribution, not for a measured cell
    fetch_on_poll: bool = False
    # MPI_Win_sync after a local store (producer-owned rings) is the unified-model
    # memory barrier that publishes the payload before the commit; "none" removes
    # it and is only for pricing it
    sync_mode: Literal["sync", "none"] = "sync"
    # MPI_Win_allocate is collective, so a second store in one process must join
    # the first one's windows. "singleton" refuses a second geometry rather than
    # deadlocking (coupled.py builds a second store on the sim ranks only);
    # "per-config" allocates per geometry and is for a harness where every rank
    # builds the same sequence of stores.
    windows: Literal["singleton", "per-config"] = "singleton"
    # MPI_Win_free is collective too, and the driver leaves one store per sim rank
    # uncleaned, so clean() ends the epoch and leaves the windows to Finalize
    free_windows: bool = False
    win_info: Dict[str, str] = Field(default_factory=dict)

    # the key travels verbatim in the slot (no hash, so no collision class);
    # control_max_bytes is the floor on a slot's payload for pickled messages
    key_rank_field: int = 1
    key_max_bytes: int = 64
    control_max_bytes: int = 4096

    # bounds: no wait here is unbounded
    full_timeout_s: float = 300.0   # producer blocked on a full ring
    stall_warn_s: float = 30.0
    poll_max_pops: int = 64         # records one poll may drain into the read-ahead
    # 0 = derive from this rank's consumed capacity (2x, min 16). A producer
    # cannot run more than capacity records ahead, so that is the most that can
    # sit in front of a key; a parked record is a whole payload, so it is a
    # memory bound as well.
    readahead_max: int = 0
    cas_max_retries: int = 64
    clean_drain_s: float = 5.0
    spin_before_sleep: int = 200
    sleep_max_s: float = 1e-3
    closed_check_every: int = 64
    # a sleeping producer may be a consumer's stall if an osc/ucx atomic needs
    # progress on the target - the gate's untested question (v3 15.8 ii)
    progress_poke: bool = True

    device: Literal["cpu"] = "cpu"  # v1; OQ3 (CUDA-aware Put) is unanswered, not answered
