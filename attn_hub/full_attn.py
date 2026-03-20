from flash_attn_compat import full_attention_decode, full_attention_prefill



def full_prefill_attn(query_states, key_states, value_states, causal):
    return full_attention_prefill(query_states, key_states, value_states, causal=causal)



def full_decode_attn(query_states, key_states, value_states, layer_idx, full_attn_cache):

    valid_len = full_attn_cache.valid_length if layer_idx == full_attn_cache.layer_num-1 else full_attn_cache.valid_length+1

    return full_attention_decode(query_states, key_states, value_states, valid_len)
