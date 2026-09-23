"""Detect the local GPU and system memory, and estimate whether a model fits.

GPU detection is NVIDIA-only, through ``nvidia-smi`` (installed with the driver
on Windows and Linux). System RAM comes from ``GlobalMemoryStatusEx`` on
Windows, ``/proc/meminfo`` on Linux and ``sysctl hw.memsize`` on macOS. Anything
that cannot be detected is reported as unknown rather than guessed; every
planner entry point also accepts explicit ``--vram`` / ``--ram`` values.

The fit estimate is deliberately simple and transparent:

    weights   = the model's download size (GGUF weights are loaded as-is)
    kv cache  = 2 (K and V) x layers x kv_heads x head_dim x bytes/element x num_ctx
    overhead  = a fixed allowance for the CUDA context and compute buffers (1 GiB)

If weights + kv cache + overhead fit in VRAM the whole model runs on the GPU.
Otherwise Ollama offloads whole layers (weights and their share of the KV cache)
to system RAM, and generation slows down roughly in proportion to the share of
layers left on the CPU. These are planning estimates, not measurements; check
the real split with ``ollama ps`` or ``python -m src.model_manager doctor``.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

GiB = 1024 ** 3
MiB = 1024 ** 2

# Bytes per cached K/V element for Ollama's OLLAMA_KV_CACHE_TYPE settings
# (q8_0 and q4_0 store 32 values plus a 2-byte scale per block).
KV_BYTES_PER_ELEMENT = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
DEFAULT_OVERHEAD_GIB = 1.0
DEFAULT_NUM_CTX = 4096          # Ollama's default context window
RAM_USABLE_FRACTION = 0.8       # leave room for the OS and other programs
CTX_STEP = 1024                 # max_ctx results are rounded down to a multiple of this

_NVIDIA_SMI_QUERY = [
    "--query-gpu=index,name,memory.total,memory.used",
    "--format=csv,noheader,nounits",
]


# -- detection --------------------------------------------------------------------
@dataclass
class GPU:
    index: int
    name: str
    total_mib: int
    used_mib: int | None = None

    @property
    def total_gib(self) -> float:
        return self.total_mib * MiB / GiB

    @property
    def used_gib(self) -> float | None:
        return None if self.used_mib is None else self.used_mib * MiB / GiB


@dataclass
class Hardware:
    gpus: list[GPU] = field(default_factory=list)
    ram_total_gib: float | None = None
    ram_available_gib: float | None = None
    gpu_note: str = ""   # why no GPU was found, when none was

    @property
    def vram_gib(self) -> float:
        """Total VRAM across NVIDIA GPUs (Ollama can split one model's layers across cards)."""
        return sum(g.total_gib for g in self.gpus)

    def describe(self) -> str:
        if self.gpus:
            parts = []
            for g in self.gpus:
                used = f", {g.used_gib:.1f} GiB in use" if g.used_gib is not None else ""
                parts.append(f"{g.name} ({g.total_gib:.1f} GiB VRAM{used})")
            gpu = "GPU: " + "; ".join(parts)
        else:
            gpu = f"GPU: none detected ({self.gpu_note})" if self.gpu_note else "GPU: none detected"
        if self.ram_total_gib is None:
            ram = "RAM: unknown"
        else:
            free = f", {self.ram_available_gib:.1f} GiB free" if self.ram_available_gib is not None else ""
            ram = f"RAM: {self.ram_total_gib:.1f} GiB{free}"
        return f"{gpu}  |  {ram}"

    def as_dict(self) -> dict:
        return {
            "gpus": [{"index": g.index, "name": g.name, "vram_gib": round(g.total_gib, 2),
                      "used_gib": None if g.used_gib is None else round(g.used_gib, 2)} for g in self.gpus],
            "vram_gib": round(self.vram_gib, 2),
            "ram_total_gib": None if self.ram_total_gib is None else round(self.ram_total_gib, 2),
            "ram_available_gib": None if self.ram_available_gib is None else round(self.ram_available_gib, 2),
            "gpu_note": self.gpu_note,
        }


def _to_int(text: str) -> int | None:
    text = text.strip()
    try:
        return int(float(text))
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


def parse_nvidia_smi(text: str) -> list[GPU]:
    """Parse ``index, name, memory.total, memory.used`` CSV lines (MiB, no header/units)."""
    gpus = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        head, _, rest = line.partition(",")
        parts = rest.rsplit(",", 2)
        if len(parts) != 3:
            continue
        index, total = _to_int(head), _to_int(parts[1])
        if index is None or not total:
            continue
        gpus.append(GPU(index=index, name=parts[0].strip(), total_mib=total, used_mib=_to_int(parts[2])))
    return gpus


def find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if sys.platform == "win32":
        for candidate in (
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                         "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
        ):
            if os.path.isfile(candidate):
                return candidate
    return None


def detect_gpus(runner=subprocess.run, exe: str | None = None) -> tuple[list[GPU], str]:
    """Return ``(gpus, note)``; ``note`` explains an empty list."""
    exe = exe or find_nvidia_smi()
    if not exe:
        return [], "nvidia-smi not found: no NVIDIA driver, or a non-NVIDIA GPU"
    try:
        proc = runner([exe, *_NVIDIA_SMI_QUERY], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"nvidia-smi could not run: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return [], f"nvidia-smi failed: {detail[0] if detail else f'exit code {proc.returncode}'}"
    gpus = parse_nvidia_smi(proc.stdout)
    return gpus, "" if gpus else "nvidia-smi reported no GPUs"


def system_memory() -> tuple[float | None, float | None]:
    """``(total_gib, available_gib)`` of system RAM; ``None`` where unknown."""
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / GiB, status.ullAvailPhys / GiB
            return None, None
        if sys.platform.startswith("linux") and os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    key, _, value = line.partition(":")
                    info[key] = int(value.split()[0]) * 1024
            return info["MemTotal"] / GiB, info.get("MemAvailable", 0) / GiB or None
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / GiB, None
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / GiB, None
    except (OSError, ValueError, KeyError, AttributeError, subprocess.SubprocessError):
        return None, None


def detect(runner=subprocess.run) -> Hardware:
    gpus, note = detect_gpus(runner)
    total, available = system_memory()
    return Hardware(gpus=gpus, ram_total_gib=total, ram_available_gib=available, gpu_note=note)


def platform_summary() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


# -- fit planning -------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    size_gb: float                     # download size in GB (10^9 bytes), as `ollama list` shows it
    n_layers: int | None = None
    n_kv_heads: int | None = None
    head_dim: int | None = None
    context_length: int | None = None

    @property
    def weights_gib(self) -> float:
        return self.size_gb * 1e9 / GiB

    @property
    def has_architecture(self) -> bool:
        return bool(self.n_layers and self.n_kv_heads and self.head_dim)

    def kv_bytes_per_token(self, kv_type: str = "f16") -> float:
        if not self.has_architecture:
            return 0.0
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * KV_BYTES_PER_ELEMENT[kv_type]

    @classmethod
    def from_catalog(cls, entry: dict) -> "ModelSpec":
        return cls(
            name=entry["name"],
            size_gb=float(entry.get("size_gb", 0)),
            n_layers=entry.get("n_layers"),
            n_kv_heads=entry.get("n_kv_heads"),
            head_dim=entry.get("head_dim"),
            context_length=entry.get("context_length"),
        )

    @classmethod
    def from_ollama(cls, name: str, size_bytes: int, model_info: dict) -> "ModelSpec":
        """Build a spec from ``/api/tags`` size and ``/api/show`` ``model_info`` (GGUF metadata)."""
        arch = model_info.get("general.architecture", "")

        def get(key):
            value = model_info.get(f"{arch}.{key}")
            return int(value) if isinstance(value, (int, float)) and value else None

        heads, kv_heads = get("attention.head_count"), get("attention.head_count_kv")
        head_dim = get("attention.key_length")
        if not head_dim and get("embedding_length") and heads:
            head_dim = get("embedding_length") // heads
        return cls(
            name=name,
            size_gb=size_bytes / 1e9,
            n_layers=get("block_count"),
            n_kv_heads=kv_heads or heads,
            head_dim=head_dim,
            context_length=get("context_length"),
        )


@dataclass
class FitResult:
    model: str
    status: str                  # "gpu", "partial", "cpu", "too_large"
    num_ctx: int
    weights_gib: float
    kv_gib: float
    overhead_gib: float
    gpu_layers: int | None       # None when the architecture is unknown
    n_layers: int | None
    gpu_percent: float           # share of the model (weights + cache) on the GPU
    max_ctx_on_gpu: int | None   # largest num_ctx that keeps everything on the GPU
    note: str = ""

    @property
    def total_gib(self) -> float:
        return self.weights_gib + self.kv_gib + self.overhead_gib

    @property
    def label(self) -> str:
        return {
            "gpu": "fits in VRAM",
            "partial": f"partial offload (~{self.gpu_percent:.0f}% on GPU)",
            "cpu": "CPU only",
            "too_large": "too large for RAM",
        }[self.status]


def max_ctx_on_gpu(spec: ModelSpec, vram_gib: float, kv_type: str = "f16",
                   overhead_gib: float = DEFAULT_OVERHEAD_GIB) -> int | None:
    """Largest ``num_ctx`` (a multiple of 1024, capped at the model's context length)
    whose KV cache still fits in VRAM next to the weights; ``None`` if not even 1024 fits."""
    per_token = spec.kv_bytes_per_token(kv_type)
    free = (vram_gib - spec.weights_gib - overhead_gib) * GiB
    if per_token <= 0 or free <= 0:
        return None
    ctx = int(free // per_token)
    if spec.context_length:
        ctx = min(ctx, spec.context_length)
    ctx -= ctx % CTX_STEP
    return ctx if ctx >= CTX_STEP else None


def plan_fit(
    spec: ModelSpec,
    vram_gib: float,
    ram_gib: float | None,
    num_ctx: int = DEFAULT_NUM_CTX,
    kv_type: str = "f16",
    overhead_gib: float = DEFAULT_OVERHEAD_GIB,
) -> FitResult:
    """Estimate where a model runs with ``num_ctx`` tokens of context.

    ``ram_gib`` is total system RAM (80% of it is treated as usable); ``None``
    skips the RAM check.
    """
    if kv_type not in KV_BYTES_PER_ELEMENT:
        raise ValueError(f"kv_type must be one of {sorted(KV_BYTES_PER_ELEMENT)}")
    weights = spec.weights_gib
    kv = spec.kv_bytes_per_token(kv_type) * num_ctx / GiB
    note = ""
    if spec.context_length and num_ctx > spec.context_length:
        note = f"num_ctx is above the model's trained context ({spec.context_length})"
    if not spec.has_architecture:
        note = note or "architecture unknown: KV cache not estimated"

    ram_budget = None if ram_gib is None else ram_gib * RAM_USABLE_FRACTION
    common = dict(model=spec.name, num_ctx=num_ctx, weights_gib=weights, kv_gib=kv, overhead_gib=overhead_gib,
                  n_layers=spec.n_layers,
                  max_ctx_on_gpu=max_ctx_on_gpu(spec, vram_gib, kv_type, overhead_gib) if vram_gib > 0 else None)

    if vram_gib > 0 and weights + kv + overhead_gib <= vram_gib:
        return FitResult(status="gpu", gpu_layers=spec.n_layers, gpu_percent=100.0, note=note, **common)

    if spec.has_architecture:
        per_layer = (weights + kv) / spec.n_layers
        room = max(vram_gib - overhead_gib, 0.0) if vram_gib > 0 else 0.0
        gpu_layers = min(spec.n_layers, int(room // per_layer))
        on_cpu = (spec.n_layers - gpu_layers) * per_layer
        percent = 100.0 * gpu_layers / spec.n_layers
    else:
        gpu_layers, on_cpu, percent = None, weights + kv, 0.0

    if ram_budget is not None and on_cpu > ram_budget:
        return FitResult(status="too_large", gpu_layers=gpu_layers, gpu_percent=percent,
                         note=note or f"needs ~{on_cpu:.1f} GiB of RAM, ~{ram_budget:.1f} GiB usable", **common)
    status = "partial" if gpu_layers else "cpu"
    return FitResult(status=status, gpu_layers=gpu_layers, gpu_percent=percent if gpu_layers else 0.0,
                     note=note, **common)
