# Rank: 1, Graph 47

class GraphModule(torch.nn.Module):
    def forward(self, l_q_: "bf16[1, 1024]", l_forward_batch_token_to_kv_pool_k_buffer_12_: "bf16[20842304, 2, 64]", l_forward_batch_token_to_kv_pool_v_buffer_12_: "bf16[20842304, 2, 64]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_q_, l_forward_batch_token_to_kv_pool_k_buffer_12_, l_forward_batch_token_to_kv_pool_v_buffer_12_);  l_q_ = l_forward_batch_token_to_kv_pool_k_buffer_12_ = l_forward_batch_token_to_kv_pool_v_buffer_12_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1]
        getitem_2 = thunder_0[2];  thunder_0 = None
        return (getitem, getitem_1, getitem_2)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_q_: "bf16[1, 1024]", l_forward_batch_token_to_kv_pool_k_buffer_12_: "bf16[20842304, 2, 64]", l_forward_batch_token_to_kv_pool_v_buffer_12_: "bf16[20842304, 2, 64]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:534 in forward_decode, code: q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            contiguous: "bf16[1, 1024]" = l_q_.contiguous();  l_q_ = None
            q: "bf16[1, 16, 64]" = contiguous.view(-1, 16, 64);  contiguous = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:538 in forward_decode, code: k_cache = k_cache.view(
            view_1: "bf16[325661, 64, 2, 64]" = l_forward_batch_token_to_kv_pool_k_buffer_12_.view(-1, 64, 2, 64);  l_forward_batch_token_to_kv_pool_k_buffer_12_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:540 in forward_decode, code: ).permute(0, 2, 1, 3)
            k_cache: "bf16[325661, 2, 64, 64]" = view_1.permute(0, 2, 1, 3);  view_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:541 in forward_decode, code: v_cache = v_cache.view(
            view_2: "bf16[325661, 64, 2, 64]" = l_forward_batch_token_to_kv_pool_v_buffer_12_.view(-1, 64, 2, 64);  l_forward_batch_token_to_kv_pool_v_buffer_12_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:543 in forward_decode, code: ).permute(0, 2, 1, 3)
            v_cache: "bf16[325661, 2, 64, 64]" = view_2.permute(0, 2, 1, 3);  view_2 = None
            return (q, k_cache, v_cache)
            
        class _model(torch.nn.Module):
            def forward(self, l_q_: "bf16[1, 1024]", l_forward_batch_token_to_kv_pool_k_buffer_12_: "bf16[20842304, 2, 64]", l_forward_batch_token_to_kv_pool_v_buffer_12_: "bf16[20842304, 2, 64]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:534 in forward_decode, code: q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                contiguous: "bf16[1, 1024]" = l_q_.contiguous();  l_q_ = None
                q: "bf16[1, 16, 64]" = contiguous.view(-1, 16, 64);  contiguous = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:538 in forward_decode, code: k_cache = k_cache.view(
                view_1: "bf16[325661, 64, 2, 64]" = l_forward_batch_token_to_kv_pool_k_buffer_12_.view(-1, 64, 2, 64);  l_forward_batch_token_to_kv_pool_k_buffer_12_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:540 in forward_decode, code: ).permute(0, 2, 1, 3)
                k_cache: "bf16[325661, 2, 64, 64]" = view_1.permute(0, 2, 1, 3);  view_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:541 in forward_decode, code: v_cache = v_cache.view(
                view_2: "bf16[325661, 64, 2, 64]" = l_forward_batch_token_to_kv_pool_v_buffer_12_.view(-1, 64, 2, 64);  l_forward_batch_token_to_kv_pool_v_buffer_12_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:543 in forward_decode, code: ).permute(0, 2, 1, 3)
                v_cache: "bf16[325661, 2, 64, 64]" = view_2.permute(0, 2, 1, 3);  view_2 = None
                return (q, k_cache, v_cache)
                