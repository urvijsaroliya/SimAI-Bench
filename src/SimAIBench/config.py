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
