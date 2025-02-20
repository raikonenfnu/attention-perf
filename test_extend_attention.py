import torch
from wave_ops.extend_attention import extend_attention_wave
from triton_ops.extend_attention import extend_attention_fwd, redundant_attention

def _test_extend_attention_once(B, N_CTX, H_Q, H_KV, D):
    dtype = torch.float16
    extend_seq_len = 1024

    b_seq_len_prefix = torch.full(
        (B,), N_CTX // B, dtype=torch.int32, device="cuda"
    )
    b_seq_len_extend = torch.full(
        (B,), extend_seq_len, dtype=torch.int32, device="cuda"
    )
    b_seq_len = b_seq_len_prefix + b_seq_len_extend
    max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

    b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
    req_to_tokens = torch.empty(
        (B, max_len_in_batch), dtype=torch.int32, device="cuda"
    )
    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    for i in range(B):
        req_to_tokens[i, : b_seq_len[i]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len[i]
        )

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device="cuda"
    ).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device="cuda"
    ).normal_(mean=0.1, std=0.2)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
    v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = torch.empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)

    o_redundant = torch.empty(
        (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
    )

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

    redundant_attention(
        q_extend,
        o_redundant,
        k_buffer,
        v_buffer,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_seq_len_prefix,
        max_len_in_batch,
    )

    o_extend = torch.empty(
        (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
    )
    extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        b_seq_len_extend,
        b_start_loc_extend,
        max_len_extend,
    )

    o_wave = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")

    extend_attention_wave(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        b_seq_len_extend,
        b_start_loc_extend,
        max_len_extend,
        o_wave,
        is_causal=True,
    )

    assert(torch.allclose(o_extend, o_redundant, rtol=1e-2))
    assert(torch.allclose(o_wave, o_redundant, rtol=1e-2))
    print("PASS!")

def test_extend_attention():

    # Define the varying parameter values
    attention_values = [128]

    # Loop through the values and call the method
    for value in attention_values:
        _test_extend_attention_once(32, 16384, 6, 1, value)

def main():
    test_extend_attention()

if __name__ == "__main__":
    main()