try:
    from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
except Exception:
    _flash_attn_with_kvcache = None

try:
    from flash_attn.cute import flash_attn_func as _flash_attn_func
except Exception:
    _flash_attn_func = None

import torch
import torch.nn.functional as F


def has_legacy_kvcache():
    return _flash_attn_with_kvcache is not None


def has_fa4():
    return _flash_attn_func is not None


def _can_use_fa4(query_states):
    if _flash_attn_func is None or not query_states.is_cuda:
        return False
    props = torch.cuda.get_device_properties(query_states.device)
    return props.major in (9, 10, 11)


def _expand_kv_for_gqa(query_states, key_states, value_states):
    q_heads = query_states.shape[2]
    kv_heads = key_states.shape[2]
    if q_heads == kv_heads:
        return key_states, value_states
    if q_heads % kv_heads != 0:
        raise ValueError(f"Query heads ({q_heads}) must be divisible by KV heads ({kv_heads}).")
    group_size = q_heads // kv_heads
    key_states = key_states.repeat_interleave(group_size, dim=2)
    value_states = value_states.repeat_interleave(group_size, dim=2)
    return key_states, value_states


def _sdpa_attention(query_states, key_states, value_states, causal):
    key_states, value_states = _expand_kv_for_gqa(query_states, key_states, value_states)
    q = query_states.transpose(1, 2)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
    return out.transpose(1, 2).contiguous()


def full_attention_prefill(query_states, key_states, value_states, causal=True):
    if _flash_attn_with_kvcache is not None:
        return _flash_attn_with_kvcache(
            q=query_states,
            k_cache=key_states,
            v_cache=value_states,
            causal=causal,
        )

    if _can_use_fa4(query_states):
        key_states, value_states = _expand_kv_for_gqa(query_states, key_states, value_states)
        return _flash_attn_func(query_states, key_states, value_states, causal=causal)

    return _sdpa_attention(query_states, key_states, value_states, causal=causal)


def full_attention_decode(query_states, key_cache, value_cache, cache_lens):
    if _flash_attn_with_kvcache is not None:
        return _flash_attn_with_kvcache(
            q=query_states,
            k_cache=key_cache,
            v_cache=value_cache,
            cache_seqlens=cache_lens,
        )

    outputs = []
    batch_size = query_states.shape[0]
    for bdx in range(batch_size):
        valid_len = int(cache_lens[bdx].item())
        q = query_states[bdx:bdx + 1]
        k = key_cache[bdx:bdx + 1, :valid_len]
        v = value_cache[bdx:bdx + 1, :valid_len]
        if _can_use_fa4(q):
            k, v = _expand_kv_for_gqa(q, k, v)
            outputs.append(_flash_attn_func(q, k, v, causal=False))
        else:
            outputs.append(_sdpa_attention(q, k, v, causal=False))
    return torch.cat(outputs, dim=0)
