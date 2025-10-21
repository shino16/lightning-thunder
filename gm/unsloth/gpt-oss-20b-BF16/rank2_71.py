# Rank: 2, Graph 71

class GraphModule(torch.nn.Module):
    def forward(self, l_forward_batch_token_to_kv_pool_k_buffer_21_: "bf16[20312896, 2, 64]", l_forward_batch_token_to_kv_pool_v_buffer_21_: "bf16[20312896, 2, 64]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_forward_batch_token_to_kv_pool_k_buffer_21_, l_forward_batch_token_to_kv_pool_v_buffer_21_);  l_forward_batch_token_to_kv_pool_k_buffer_21_ = l_forward_batch_token_to_kv_pool_v_buffer_21_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_forward_batch_token_to_kv_pool_k_buffer_21_: "bf16[20312896, 2, 64]", l_forward_batch_token_to_kv_pool_v_buffer_21_: "bf16[20312896, 2, 64]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/gpt_oss.py:215 in _create_fused_set_kv_buffer_arg, code: k_buffer=k_buffer.view(k_buffer.shape[0], -1),
            view: "bf16[20312896, 128]" = l_forward_batch_token_to_kv_pool_k_buffer_21_.view(20312896, -1);  l_forward_batch_token_to_kv_pool_k_buffer_21_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/gpt_oss.py:216 in _create_fused_set_kv_buffer_arg, code: v_buffer=v_buffer.view(v_buffer.shape[0], -1),
            view_1: "bf16[20312896, 128]" = l_forward_batch_token_to_kv_pool_v_buffer_21_.view(20312896, -1);  l_forward_batch_token_to_kv_pool_v_buffer_21_ = None
            return (view_1, view)
            
        class _model(torch.nn.Module):
            def forward(self, l_forward_batch_token_to_kv_pool_k_buffer_21_: "bf16[20312896, 2, 64]", l_forward_batch_token_to_kv_pool_v_buffer_21_: "bf16[20312896, 2, 64]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/gpt_oss.py:215 in _create_fused_set_kv_buffer_arg, code: k_buffer=k_buffer.view(k_buffer.shape[0], -1),
                view: "bf16[20312896, 128]" = l_forward_batch_token_to_kv_pool_k_buffer_21_.view(20312896, -1);  l_forward_batch_token_to_kv_pool_k_buffer_21_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/gpt_oss.py:216 in _create_fused_set_kv_buffer_arg, code: v_buffer=v_buffer.view(v_buffer.shape[0], -1),
                view_1: "bf16[20312896, 128]" = l_forward_batch_token_to_kv_pool_v_buffer_21_.view(20312896, -1);  l_forward_batch_token_to_kv_pool_v_buffer_21_ = None
                return (view_1, view)
                