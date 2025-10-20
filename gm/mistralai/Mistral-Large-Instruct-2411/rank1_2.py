# Rank: 1, Graph 2

class GraphModule(torch.nn.Module):
    def forward(self, l_self_modules_qkv_proj_parameters_weight_: "f16[3584, 12288]", l_hidden_states_: "f16[1, 12288]", l_positions_: "i64[1]", l_self_modules_rotary_emb_buffers_cos_sin_cache_: "f32[131072, 128]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_hidden_states_, l_self_modules_qkv_proj_parameters_weight_, l_positions_, l_self_modules_rotary_emb_buffers_cos_sin_cache_);  l_hidden_states_ = l_self_modules_qkv_proj_parameters_weight_ = l_positions_ = l_self_modules_rotary_emb_buffers_cos_sin_cache_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1]
        getitem_2 = thunder_0[2];  thunder_0 = None
        return (getitem, getitem_1, getitem_2)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_hidden_states_: "f16[1, 12288]", l_self_modules_qkv_proj_parameters_weight_: "f16[3584, 12288]", l_positions_: "i64[1]", l_self_modules_rotary_emb_buffers_cos_sin_cache_: "f32[131072, 128]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel: "f16[1, 3584]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_qkv_proj_parameters_weight_, None);  l_hidden_states_ = l_self_modules_qkv_proj_parameters_weight_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/llama.py:195 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            split = output_parallel.split([3072, 256, 256], dim = -1);  output_parallel = None
            q: "f16[1, 3072]" = split[0]
            k: "f16[1, 256]" = split[1]
            v: "f16[1, 256]" = split[2];  split = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:149 in forward_native, code: positions = positions.flatten()
            positions: "i64[1]" = l_positions_.flatten();  l_positions_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:151 in forward_native, code: cos_sin = self.cos_sin_cache.index_select(0, positions)
            cos_sin: "f32[1, 128]" = l_self_modules_rotary_emb_buffers_cos_sin_cache_.index_select(0, positions);  l_self_modules_rotary_emb_buffers_cos_sin_cache_ = positions = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:152 in forward_native, code: cos, sin = cos_sin.chunk(2, dim=-1)
            chunk = cos_sin.chunk(2, dim = -1);  cos_sin = None
            cos: "f32[1, 64]" = chunk[0]
            sin: "f32[1, 64]" = chunk[1];  chunk = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:155 in forward_native, code: query = query.view(num_tokens, -1, self.head_size)
            query: "f16[1, 24, 128]" = q.view(1, -1, 128);  q = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:156 in forward_native, code: query_rot = query[..., : self.rotary_dim]
            query_rot: "f16[1, 24, 128]" = query[(Ellipsis, slice(None, 128, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:157 in forward_native, code: query_pass = query[..., self.rotary_dim :]
            query_pass: "f16[1, 24, 0]" = query[(Ellipsis, slice(128, None, None))];  query = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:64 in _apply_rotary_emb, code: cos = cos.unsqueeze(-2).to(x.dtype)
            unsqueeze: "f32[1, 1, 64]" = cos.unsqueeze(-2)
            cos_1: "f16[1, 1, 64]" = unsqueeze.to(torch.float16);  unsqueeze = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:65 in _apply_rotary_emb, code: sin = sin.unsqueeze(-2).to(x.dtype)
            unsqueeze_1: "f32[1, 1, 64]" = sin.unsqueeze(-2)
            sin_1: "f16[1, 1, 64]" = unsqueeze_1.to(torch.float16);  unsqueeze_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:67 in _apply_rotary_emb, code: x1, x2 = torch.chunk(x, 2, dim=-1)
            chunk_1 = torch.chunk(query_rot, 2, dim = -1);  query_rot = None
            x1: "f16[1, 24, 64]" = chunk_1[0]
            x2: "f16[1, 24, 64]" = chunk_1[1];  chunk_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:71 in _apply_rotary_emb, code: o1 = x1 * cos - x2 * sin
            mul: "f16[1, 24, 64]" = x1 * cos_1
            mul_1: "f16[1, 24, 64]" = x2 * sin_1
            o1: "f16[1, 24, 64]" = mul - mul_1;  mul = mul_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:72 in _apply_rotary_emb, code: o2 = x2 * cos + x1 * sin
            mul_2: "f16[1, 24, 64]" = x2 * cos_1;  x2 = cos_1 = None
            mul_3: "f16[1, 24, 64]" = x1 * sin_1;  x1 = sin_1 = None
            o2: "f16[1, 24, 64]" = mul_2 + mul_3;  mul_2 = mul_3 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:74 in _apply_rotary_emb, code: return torch.cat((o1, o2), dim=-1)
            query_rot_1: "f16[1, 24, 128]" = torch.cat((o1, o2), dim = -1);  o1 = o2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:159 in forward_native, code: query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
            cat_1: "f16[1, 24, 128]" = torch.cat((query_rot_1, query_pass), dim = -1);  query_rot_1 = query_pass = None
            query_1: "f16[1, 3072]" = cat_1.reshape((1, 3072));  cat_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:162 in forward_native, code: key = key.view(num_tokens, -1, self.head_size)
            key: "f16[1, 2, 128]" = k.view(1, -1, 128);  k = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:163 in forward_native, code: key_rot = key[..., : self.rotary_dim]
            key_rot: "f16[1, 2, 128]" = key[(Ellipsis, slice(None, 128, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:164 in forward_native, code: key_pass = key[..., self.rotary_dim :]
            key_pass: "f16[1, 2, 0]" = key[(Ellipsis, slice(128, None, None))];  key = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:64 in _apply_rotary_emb, code: cos = cos.unsqueeze(-2).to(x.dtype)
            unsqueeze_2: "f32[1, 1, 64]" = cos.unsqueeze(-2);  cos = None
            cos_2: "f16[1, 1, 64]" = unsqueeze_2.to(torch.float16);  unsqueeze_2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:65 in _apply_rotary_emb, code: sin = sin.unsqueeze(-2).to(x.dtype)
            unsqueeze_3: "f32[1, 1, 64]" = sin.unsqueeze(-2);  sin = None
            sin_2: "f16[1, 1, 64]" = unsqueeze_3.to(torch.float16);  unsqueeze_3 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:67 in _apply_rotary_emb, code: x1, x2 = torch.chunk(x, 2, dim=-1)
            chunk_2 = torch.chunk(key_rot, 2, dim = -1);  key_rot = None
            x1_1: "f16[1, 2, 64]" = chunk_2[0]
            x2_1: "f16[1, 2, 64]" = chunk_2[1];  chunk_2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:71 in _apply_rotary_emb, code: o1 = x1 * cos - x2 * sin
            mul_4: "f16[1, 2, 64]" = x1_1 * cos_2
            mul_5: "f16[1, 2, 64]" = x2_1 * sin_2
            o1_1: "f16[1, 2, 64]" = mul_4 - mul_5;  mul_4 = mul_5 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:72 in _apply_rotary_emb, code: o2 = x2 * cos + x1 * sin
            mul_6: "f16[1, 2, 64]" = x2_1 * cos_2;  x2_1 = cos_2 = None
            mul_7: "f16[1, 2, 64]" = x1_1 * sin_2;  x1_1 = sin_2 = None
            o2_1: "f16[1, 2, 64]" = mul_6 + mul_7;  mul_6 = mul_7 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:74 in _apply_rotary_emb, code: return torch.cat((o1, o2), dim=-1)
            key_rot_1: "f16[1, 2, 128]" = torch.cat((o1_1, o2_1), dim = -1);  o1_1 = o2_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:166 in forward_native, code: key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
            cat_3: "f16[1, 2, 128]" = torch.cat((key_rot_1, key_pass), dim = -1);  key_rot_1 = key_pass = None
            key_1: "f16[1, 256]" = cat_3.reshape((1, 256));  cat_3 = None
            return (query_1, key_1, v)
            
        class _model(torch.nn.Module):
            def forward(self, l_hidden_states_: "f16[1, 12288]", l_self_modules_qkv_proj_parameters_weight_: "f16[3584, 12288]", l_positions_: "i64[1]", l_self_modules_rotary_emb_buffers_cos_sin_cache_: "f32[131072, 128]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel: "f16[1, 3584]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_qkv_proj_parameters_weight_, None);  l_hidden_states_ = l_self_modules_qkv_proj_parameters_weight_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/llama.py:195 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                split = output_parallel.split([3072, 256, 256], dim = -1);  output_parallel = None
                q: "f16[1, 3072]" = split[0]
                k: "f16[1, 256]" = split[1]
                v: "f16[1, 256]" = split[2];  split = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:149 in forward_native, code: positions = positions.flatten()
                positions: "i64[1]" = l_positions_.flatten();  l_positions_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:151 in forward_native, code: cos_sin = self.cos_sin_cache.index_select(0, positions)
                cos_sin: "f32[1, 128]" = l_self_modules_rotary_emb_buffers_cos_sin_cache_.index_select(0, positions);  l_self_modules_rotary_emb_buffers_cos_sin_cache_ = positions = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:152 in forward_native, code: cos, sin = cos_sin.chunk(2, dim=-1)
                chunk = cos_sin.chunk(2, dim = -1);  cos_sin = None
                cos: "f32[1, 64]" = chunk[0]
                sin: "f32[1, 64]" = chunk[1];  chunk = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:155 in forward_native, code: query = query.view(num_tokens, -1, self.head_size)
                query: "f16[1, 24, 128]" = q.view(1, -1, 128);  q = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:156 in forward_native, code: query_rot = query[..., : self.rotary_dim]
                query_rot: "f16[1, 24, 128]" = query[(Ellipsis, slice(None, 128, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:157 in forward_native, code: query_pass = query[..., self.rotary_dim :]
                query_pass: "f16[1, 24, 0]" = query[(Ellipsis, slice(128, None, None))];  query = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:64 in _apply_rotary_emb, code: cos = cos.unsqueeze(-2).to(x.dtype)
                unsqueeze: "f32[1, 1, 64]" = cos.unsqueeze(-2)
                cos_1: "f16[1, 1, 64]" = unsqueeze.to(torch.float16);  unsqueeze = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:65 in _apply_rotary_emb, code: sin = sin.unsqueeze(-2).to(x.dtype)
                unsqueeze_1: "f32[1, 1, 64]" = sin.unsqueeze(-2)
                sin_1: "f16[1, 1, 64]" = unsqueeze_1.to(torch.float16);  unsqueeze_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:67 in _apply_rotary_emb, code: x1, x2 = torch.chunk(x, 2, dim=-1)
                chunk_1 = torch.chunk(query_rot, 2, dim = -1);  query_rot = None
                x1: "f16[1, 24, 64]" = chunk_1[0]
                x2: "f16[1, 24, 64]" = chunk_1[1];  chunk_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:71 in _apply_rotary_emb, code: o1 = x1 * cos - x2 * sin
                mul: "f16[1, 24, 64]" = x1 * cos_1
                mul_1: "f16[1, 24, 64]" = x2 * sin_1
                o1: "f16[1, 24, 64]" = mul - mul_1;  mul = mul_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:72 in _apply_rotary_emb, code: o2 = x2 * cos + x1 * sin
                mul_2: "f16[1, 24, 64]" = x2 * cos_1;  x2 = cos_1 = None
                mul_3: "f16[1, 24, 64]" = x1 * sin_1;  x1 = sin_1 = None
                o2: "f16[1, 24, 64]" = mul_2 + mul_3;  mul_2 = mul_3 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:74 in _apply_rotary_emb, code: return torch.cat((o1, o2), dim=-1)
                query_rot_1: "f16[1, 24, 128]" = torch.cat((o1, o2), dim = -1);  o1 = o2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:159 in forward_native, code: query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
                cat_1: "f16[1, 24, 128]" = torch.cat((query_rot_1, query_pass), dim = -1);  query_rot_1 = query_pass = None
                query_1: "f16[1, 3072]" = cat_1.reshape((1, 3072));  cat_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:162 in forward_native, code: key = key.view(num_tokens, -1, self.head_size)
                key: "f16[1, 2, 128]" = k.view(1, -1, 128);  k = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:163 in forward_native, code: key_rot = key[..., : self.rotary_dim]
                key_rot: "f16[1, 2, 128]" = key[(Ellipsis, slice(None, 128, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:164 in forward_native, code: key_pass = key[..., self.rotary_dim :]
                key_pass: "f16[1, 2, 0]" = key[(Ellipsis, slice(128, None, None))];  key = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:64 in _apply_rotary_emb, code: cos = cos.unsqueeze(-2).to(x.dtype)
                unsqueeze_2: "f32[1, 1, 64]" = cos.unsqueeze(-2);  cos = None
                cos_2: "f16[1, 1, 64]" = unsqueeze_2.to(torch.float16);  unsqueeze_2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:65 in _apply_rotary_emb, code: sin = sin.unsqueeze(-2).to(x.dtype)
                unsqueeze_3: "f32[1, 1, 64]" = sin.unsqueeze(-2);  sin = None
                sin_2: "f16[1, 1, 64]" = unsqueeze_3.to(torch.float16);  unsqueeze_3 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:67 in _apply_rotary_emb, code: x1, x2 = torch.chunk(x, 2, dim=-1)
                chunk_2 = torch.chunk(key_rot, 2, dim = -1);  key_rot = None
                x1_1: "f16[1, 2, 64]" = chunk_2[0]
                x2_1: "f16[1, 2, 64]" = chunk_2[1];  chunk_2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:71 in _apply_rotary_emb, code: o1 = x1 * cos - x2 * sin
                mul_4: "f16[1, 2, 64]" = x1_1 * cos_2
                mul_5: "f16[1, 2, 64]" = x2_1 * sin_2
                o1_1: "f16[1, 2, 64]" = mul_4 - mul_5;  mul_4 = mul_5 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:72 in _apply_rotary_emb, code: o2 = x2 * cos + x1 * sin
                mul_6: "f16[1, 2, 64]" = x2_1 * cos_2;  x2_1 = cos_2 = None
                mul_7: "f16[1, 2, 64]" = x1_1 * sin_2;  x1_1 = sin_2 = None
                o2_1: "f16[1, 2, 64]" = mul_6 + mul_7;  mul_6 = mul_7 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:74 in _apply_rotary_emb, code: return torch.cat((o1, o2), dim=-1)
                key_rot_1: "f16[1, 2, 128]" = torch.cat((o1_1, o2_1), dim = -1);  o1_1 = o2_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:166 in forward_native, code: key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
                cat_3: "f16[1, 2, 128]" = torch.cat((key_rot_1, key_pass), dim = -1);  key_rot_1 = key_pass = None
                key_1: "f16[1, 256]" = cat_3.reshape((1, 256));  cat_3 = None
                return (query_1, key_1, v)
                