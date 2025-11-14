# Rank: 0, Graph 6

class GraphModule(torch.nn.Module):
    def forward(self, l_q_: "bf16[1, 1280]", l_k_: "bf16[1, 2, 128]", l_forward_batch_token_to_kv_pool_k_buffer_0_: "bf16[3195971, 2, 128]", l_forward_batch_out_cache_loc: "i64[1]", l_forward_batch_token_to_kv_pool_v_buffer_0_: "bf16[3195971, 2, 128]", l_v_: "bf16[1, 2, 128]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_q_);  l_q_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        inductor_1 = self.inductor_1()
        getitem_2 = inductor_1[0]
        getitem_3 = inductor_1[1];  inductor_1 = None
        thunder_2 = self.thunder_2(l_forward_batch_token_to_kv_pool_k_buffer_0_, l_forward_batch_out_cache_loc, l_k_);  l_forward_batch_token_to_kv_pool_k_buffer_0_ = l_k_ = thunder_2 = None
        inductor_3 = self.inductor_3(getitem_2)
        thunder_4 = self.thunder_4(l_forward_batch_token_to_kv_pool_v_buffer_0_, l_forward_batch_out_cache_loc, l_v_);  l_forward_batch_token_to_kv_pool_v_buffer_0_ = l_forward_batch_out_cache_loc = l_v_ = thunder_4 = None
        inductor_5 = self.inductor_5(inductor_3, getitem_3, getitem_2);  inductor_3 = getitem_3 = getitem_2 = inductor_5 = None
        thunder_6 = self.thunder_6(getitem, getitem_1);  getitem = None
        getitem_4 = thunder_6[0]
        getitem_5 = thunder_6[1];  thunder_6 = None
        return (getitem_4, getitem_5, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_q_: "bf16[1, 1280]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/triton_backend.py:800 in forward_decode, code: q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
            q: "bf16[1, 1280]" = l_q_.reshape(-1, 1280);  l_q_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/triton_backend.py:806 in forward_decode, code: o = torch.empty_like(q)
            o: "bf16[1, 1280]" = torch.empty_like(q)
            return (q, o)
            
        class _model(torch.nn.Module):
            def forward(self, l_q_: "bf16[1, 1280]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/triton_backend.py:800 in forward_decode, code: q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
                q: "bf16[1, 1280]" = l_q_.reshape(-1, 1280);  l_q_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/triton_backend.py:806 in forward_decode, code: o = torch.empty_like(q)
                o: "bf16[1, 1280]" = torch.empty_like(q)
                return (q, o)
                
    class inductor_1(torch.nn.Module):
        def forward(self):
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:630 in set_kv_buffer, code: if get_is_capture_mode() and self.alt_stream is not None:
            stream = torch.cuda.streams.Stream(stream_id = 35, device_index = 0, device_type = 1)
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:632 in set_kv_buffer, code: current_stream = self.device_module.current_stream()
            current_stream = torch.cuda.current_stream()
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:633 in set_kv_buffer, code: self.alt_stream.wait_stream(current_stream)
            wait_stream = stream.wait_stream(current_stream);  wait_stream = None
            return (stream, current_stream)
            
        class graph_module(torch.nn.Module):
            def forward(self):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:630 in set_kv_buffer, code: if get_is_capture_mode() and self.alt_stream is not None:
                stream = torch.cuda.streams.Stream(stream_id = 35, device_index = 0, device_type = 1)
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:632 in set_kv_buffer, code: current_stream = self.device_module.current_stream()
                current_stream = torch.cuda.current_stream()
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:633 in set_kv_buffer, code: self.alt_stream.wait_stream(current_stream)
                wait_stream = stream.wait_stream(current_stream);  wait_stream = None
                return (stream, current_stream)
                
    class thunder_2(torch.nn.Module):
        def forward(self, l_forward_batch_token_to_kv_pool_k_buffer_0_: "bf16[3195971, 2, 128]", l_forward_batch_out_cache_loc: "i64[1]", l_k_: "bf16[1, 2, 128]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:634 in set_kv_buffer, code: self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            l_forward_batch_token_to_kv_pool_k_buffer_0_[l_forward_batch_out_cache_loc] = l_k_;  setitem = l_forward_batch_token_to_kv_pool_k_buffer_0_;  l_forward_batch_token_to_kv_pool_k_buffer_0_ = l_forward_batch_out_cache_loc = l_k_ = setitem = None
            return ()
            
        class _model(torch.nn.Module):
            def forward(self, l_forward_batch_token_to_kv_pool_k_buffer_0_: "bf16[3195971, 2, 128]", l_forward_batch_out_cache_loc: "i64[1]", l_k_: "bf16[1, 2, 128]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:634 in set_kv_buffer, code: self.k_buffer[layer_id - self.start_layer][loc] = cache_k
                l_forward_batch_token_to_kv_pool_k_buffer_0_[l_forward_batch_out_cache_loc] = l_k_;  setitem = l_forward_batch_token_to_kv_pool_k_buffer_0_;  l_forward_batch_token_to_kv_pool_k_buffer_0_ = l_forward_batch_out_cache_loc = l_k_ = setitem = None
                return ()
                
    class inductor_3(torch.nn.Module):
        def forward(self, stream):
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:635 in set_kv_buffer, code: with self.device_module.stream(self.alt_stream):
            current_stream_1 = torch.cuda.current_stream(None)
            set_stream = torch.cuda.set_stream(stream);  stream = set_stream = None
            return current_stream_1
            
        class graph_module(torch.nn.Module):
            def forward(self, stream):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:635 in set_kv_buffer, code: with self.device_module.stream(self.alt_stream):
                current_stream_1 = torch.cuda.current_stream(None)
                set_stream = torch.cuda.set_stream(stream);  stream = set_stream = None
                return current_stream_1
                
    class thunder_4(torch.nn.Module):
        def forward(self, l_forward_batch_token_to_kv_pool_v_buffer_0_: "bf16[3195971, 2, 128]", l_forward_batch_out_cache_loc: "i64[1]", l_v_: "bf16[1, 2, 128]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:636 in set_kv_buffer, code: self.v_buffer[layer_id - self.start_layer][loc] = cache_v
            l_forward_batch_token_to_kv_pool_v_buffer_0_[l_forward_batch_out_cache_loc] = l_v_;  setitem_1 = l_forward_batch_token_to_kv_pool_v_buffer_0_;  l_forward_batch_token_to_kv_pool_v_buffer_0_ = l_forward_batch_out_cache_loc = l_v_ = setitem_1 = None
            return ()
            
        class _model(torch.nn.Module):
            def forward(self, l_forward_batch_token_to_kv_pool_v_buffer_0_: "bf16[3195971, 2, 128]", l_forward_batch_out_cache_loc: "i64[1]", l_v_: "bf16[1, 2, 128]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:636 in set_kv_buffer, code: self.v_buffer[layer_id - self.start_layer][loc] = cache_v
                l_forward_batch_token_to_kv_pool_v_buffer_0_[l_forward_batch_out_cache_loc] = l_v_;  setitem_1 = l_forward_batch_token_to_kv_pool_v_buffer_0_;  l_forward_batch_token_to_kv_pool_v_buffer_0_ = l_forward_batch_out_cache_loc = l_v_ = setitem_1 = None
                return ()
                
    class inductor_5(torch.nn.Module):
        def forward(self, current_stream_1, current_stream, stream):
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:635 in set_kv_buffer, code: with self.device_module.stream(self.alt_stream):
            set_stream_1 = torch.cuda.set_stream(current_stream_1);  current_stream_1 = set_stream_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:637 in set_kv_buffer, code: current_stream.wait_stream(self.alt_stream)
            wait_stream_1 = current_stream.wait_stream(stream);  current_stream = stream = wait_stream_1 = None
            return ()
            
        class graph_module(torch.nn.Module):
            def forward(self, current_stream_1, current_stream, stream):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:635 in set_kv_buffer, code: with self.device_module.stream(self.alt_stream):
                set_stream_1 = torch.cuda.set_stream(current_stream_1);  current_stream_1 = set_stream_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:637 in set_kv_buffer, code: current_stream.wait_stream(self.alt_stream)
                wait_stream_1 = current_stream.wait_stream(stream);  current_stream = stream = wait_stream_1 = None
                return ()
                
    class thunder_6(torch.nn.Module):
        def forward(self, q: "bf16[1, 1280]", o: "bf16[1, 1280]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/triton_backend.py:823 in forward_decode, code: q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            view: "bf16[1, 10, 128]" = q.view(-1, 10, 128);  q = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/triton_backend.py:826 in forward_decode, code: o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            view_1: "bf16[1, 10, 128]" = o.view(-1, 10, 128);  o = None
            return (view, view_1)
            
        class _model(torch.nn.Module):
            def forward(self, q: "bf16[1, 1280]", o: "bf16[1, 1280]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/triton_backend.py:823 in forward_decode, code: q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                view: "bf16[1, 10, 128]" = q.view(-1, 10, 128);  q = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/triton_backend.py:826 in forward_decode, code: o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                view_1: "bf16[1, 10, 128]" = o.view(-1, 10, 128);  o = None
                return (view, view_1)
                