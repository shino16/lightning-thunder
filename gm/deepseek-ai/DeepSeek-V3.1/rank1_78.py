# Rank: 1, Graph 78

class GraphModule(torch.nn.Module):
    def forward(self, l_k_: "bf16[1, 1, 512]", l_k_rope_: "bf16[1, 1, 64]", l_forward_batch_token_to_kv_pool_kv_buffer_41_: "bf16[1259824, 1, 576]", l_forward_batch_out_cache_loc: "i64[1]", l_q_rope_: "bf16[1, 32, 64]", l_q_: "bf16[1, 32, 512]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_forward_batch_token_to_kv_pool_kv_buffer_41_, l_k_, l_k_rope_, l_forward_batch_out_cache_loc);  l_k_ = l_k_rope_ = l_forward_batch_out_cache_loc = inductor_0 = None
        thunder_1 = self.thunder_1(l_q_, l_q_rope_, l_forward_batch_token_to_kv_pool_kv_buffer_41_);  l_q_ = l_q_rope_ = l_forward_batch_token_to_kv_pool_kv_buffer_41_ = None
        getitem = thunder_1[0]
        getitem_1 = thunder_1[1]
        getitem_2 = thunder_1[2];  thunder_1 = None
        inductor_2 = self.inductor_2(getitem)
        thunder_3 = self.thunder_3(getitem_1);  getitem_1 = None
        getitem_3 = thunder_3[0]
        getitem_4 = thunder_3[1];  thunder_3 = None
        return (getitem, getitem_2, getitem_3, getitem_4, inductor_2)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_forward_batch_token_to_kv_pool_kv_buffer_41_: "bf16[1259824, 1, 576]", l_k_: "bf16[1, 1, 512]", l_k_rope_: "bf16[1, 1, 64]", l_forward_batch_out_cache_loc: "i64[1]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:1006 in set_mla_kv_buffer_triton, code: set_mla_kv_buffer_kernel[grid](
            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 15, constant_args_idx = 132, grid = [(1, 5, 1)], tma_descriptor_metadata = {}, kwargs = {'kv_buffer_ptr': l_forward_batch_token_to_kv_pool_kv_buffer_41_, 'cache_k_nope_ptr': l_k_, 'cache_k_rope_ptr': l_k_rope_, 'loc_ptr': l_forward_batch_out_cache_loc});  l_forward_batch_token_to_kv_pool_kv_buffer_41_ = l_k_ = l_k_rope_ = l_forward_batch_out_cache_loc = triton_kernel_wrapper_mutation = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_forward_batch_token_to_kv_pool_kv_buffer_41_: "bf16[1259824, 1, 576]", l_k_: "bf16[1, 1, 512]", l_k_rope_: "bf16[1, 1, 64]", l_forward_batch_out_cache_loc: "i64[1]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:1006 in set_mla_kv_buffer_triton, code: set_mla_kv_buffer_kernel[grid](
                triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 15, constant_args_idx = 132, grid = [(1, 5, 1)], tma_descriptor_metadata = {}, kwargs = {'kv_buffer_ptr': l_forward_batch_token_to_kv_pool_kv_buffer_41_, 'cache_k_nope_ptr': l_k_, 'cache_k_rope_ptr': l_k_rope_, 'loc_ptr': l_forward_batch_out_cache_loc});  l_forward_batch_token_to_kv_pool_kv_buffer_41_ = l_k_ = l_k_rope_ = l_forward_batch_out_cache_loc = triton_kernel_wrapper_mutation = None
                return ()
                
    class thunder_1(torch.nn.Module):
        def forward(self, l_q_: "bf16[1, 32, 512]", l_q_rope_: "bf16[1, 32, 64]", l_forward_batch_token_to_kv_pool_kv_buffer_41_: "bf16[1259824, 1, 576]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:615 in forward_decode, code: q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_nope: "bf16[1, 32, 512]" = l_q_.view(-1, 32, 512);  l_q_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:616 in forward_decode, code: q_rope = q_rope.view(
            q_rope: "bf16[1, 32, 64]" = l_q_rope_.view(-1, 32, 64);  l_q_rope_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:624 in forward_decode, code: k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
            k_buffer: "bf16[1259824, 1, 576]" = l_forward_batch_token_to_kv_pool_kv_buffer_41_.to(torch.bfloat16);  l_forward_batch_token_to_kv_pool_kv_buffer_41_ = None
            return (q_nope, k_buffer, q_rope)
            
        class _model(torch.nn.Module):
            def forward(self, l_q_: "bf16[1, 32, 512]", l_q_rope_: "bf16[1, 32, 64]", l_forward_batch_token_to_kv_pool_kv_buffer_41_: "bf16[1259824, 1, 576]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:615 in forward_decode, code: q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_nope: "bf16[1, 32, 512]" = l_q_.view(-1, 32, 512);  l_q_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:616 in forward_decode, code: q_rope = q_rope.view(
                q_rope: "bf16[1, 32, 64]" = l_q_rope_.view(-1, 32, 64);  l_q_rope_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:624 in forward_decode, code: k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                k_buffer: "bf16[1259824, 1, 576]" = l_forward_batch_token_to_kv_pool_kv_buffer_41_.to(torch.bfloat16);  l_forward_batch_token_to_kv_pool_kv_buffer_41_ = None
                return (q_nope, k_buffer, q_rope)
                
    class inductor_2(torch.nn.Module):
        def forward(self, q_nope: "bf16[1, 32, 512]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:628 in forward_decode, code: o = q_nope.new_empty(q_nope.shape)
            o: "bf16[1, 32, 512]" = q_nope.new_empty((1, 32, 512));  q_nope = None
            return o
            
        class _orig_mod(torch.nn.Module):
            def forward(self, q_nope: "bf16[1, 32, 512]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:628 in forward_decode, code: o = q_nope.new_empty(q_nope.shape)
                o: "bf16[1, 32, 512]" = q_nope.new_empty((1, 32, 512));  q_nope = None
                return o
                
    class thunder_3(torch.nn.Module):
        def forward(self, k_buffer: "bf16[1259824, 1, 576]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:633 in forward_decode, code: k_buffer[:, :, : layer.v_head_dim],
            getitem: "bf16[1259824, 1, 512]" = k_buffer[(slice(None, None, None), slice(None, None, None), slice(None, 512, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:634 in forward_decode, code: k_buffer[:, :, layer.v_head_dim :],
            getitem_1: "bf16[1259824, 1, 64]" = k_buffer[(slice(None, None, None), slice(None, None, None), slice(512, None, None))];  k_buffer = None
            return (getitem, getitem_1)
            
        class _model(torch.nn.Module):
            def forward(self, k_buffer: "bf16[1259824, 1, 576]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:633 in forward_decode, code: k_buffer[:, :, : layer.v_head_dim],
                getitem: "bf16[1259824, 1, 512]" = k_buffer[(slice(None, None, None), slice(None, None, None), slice(None, 512, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:634 in forward_decode, code: k_buffer[:, :, layer.v_head_dim :],
                getitem_1: "bf16[1259824, 1, 64]" = k_buffer[(slice(None, None, None), slice(None, None, None), slice(512, None, None))];  k_buffer = None
                return (getitem, getitem_1)
                