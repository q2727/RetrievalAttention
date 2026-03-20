"""
Project-owned interface for the sparse RetroInfer weighted decode operator.

This module intentionally defines only the API boundary used by RetroInfer.
The backend implementation is the next dedicated work item:

1. lock down a reference implementation that matches the original semantics
2. add a Triton backend
3. add an optional CUDA backend only if Triton is insufficient
"""


def weighted_flash_decoding(
    q,
    k,
    v,
    cluster_size=None,
    previous_out=None,
    previous_lse=None,
    cache_seqlens=None,
    return_softmax_lse=False,
):
    raise NotImplementedError(
        "RetroInfer sparse decode is intentionally left unimplemented here. "
        "Implement the project-local `weighted_flash_decoding` backend in "
        "`weighted_flash_decoding.py` before enabling sparse RetroInfer. "
        "Required semantics: cluster_size-aware decode, cache_seqlens masking, "
        "and previous_out/previous_lse merge."
    )
