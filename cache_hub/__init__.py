from .flash_attn_cache import flash_attn_cache


try:
    from .retroinfer_cache import retroinfer_cache
except Exception as exc:
    _retroinfer_cache_exc = exc

    class retroinfer_cache:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RetroInfer CPU-offload cache is unavailable. "
                "Build `library/retroinfer` and install the RetroInfer runtime dependencies first."
            ) from _retroinfer_cache_exc


try:
    from .retroinfer_cache_gpu import retroinfer_cache_gpu
except Exception as exc:
    _retroinfer_cache_gpu_exc = exc

    class retroinfer_cache_gpu:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RetroInfer GPU-only cache is unavailable. "
                "Build `library/retroinfer` and install the RetroInfer runtime dependencies first."
            ) from _retroinfer_cache_gpu_exc
