# Rank: 2, Graph 5

class GraphModule(torch.nn.Module):
    def forward(self, l_args_0_: "i64[1]", l_self_buffers_cos_sin_cache_: "f32[40960, 128]", l_args_1_: "bf16[1, 1024]", l_args_2_: "bf16[1, 128]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_args_0_, l_self_buffers_cos_sin_cache_, l_args_1_, l_args_2_);  l_args_0_ = l_self_buffers_cos_sin_cache_ = l_args_1_ = l_args_2_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_args_0_: "i64[1]", l_self_buffers_cos_sin_cache_: "f32[40960, 128]", l_args_1_: "bf16[1, 1024]", l_args_2_: "bf16[1, 128]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:149 in forward_native, code: positions = positions.flatten()
            positions: "i64[1]" = l_args_0_.flatten();  l_args_0_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:151 in forward_native, code: cos_sin = self.cos_sin_cache.index_select(0, positions)
            cos_sin: "f32[1, 128]" = l_self_buffers_cos_sin_cache_.index_select(0, positions);  l_self_buffers_cos_sin_cache_ = positions = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:152 in forward_native, code: cos, sin = cos_sin.chunk(2, dim=-1)
            chunk = cos_sin.chunk(2, dim = -1);  cos_sin = None
            cos: "f32[1, 64]" = chunk[0]
            sin: "f32[1, 64]" = chunk[1];  chunk = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:155 in forward_native, code: query = query.view(num_tokens, -1, self.head_size)
            query: "bf16[1, 8, 128]" = l_args_1_.view(1, -1, 128);  l_args_1_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:156 in forward_native, code: query_rot = query[..., : self.rotary_dim]
            query_rot: "bf16[1, 8, 128]" = query[(Ellipsis, slice(None, 128, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:157 in forward_native, code: query_pass = query[..., self.rotary_dim :]
            query_pass: "bf16[1, 8, 0]" = query[(Ellipsis, slice(128, None, None))];  query = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:64 in _apply_rotary_emb, code: cos = cos.unsqueeze(-2).to(x.dtype)
            unsqueeze: "f32[1, 1, 64]" = cos.unsqueeze(-2)
            cos_1: "bf16[1, 1, 64]" = unsqueeze.to(torch.bfloat16);  unsqueeze = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:65 in _apply_rotary_emb, code: sin = sin.unsqueeze(-2).to(x.dtype)
            unsqueeze_1: "f32[1, 1, 64]" = sin.unsqueeze(-2)
            sin_1: "bf16[1, 1, 64]" = unsqueeze_1.to(torch.bfloat16);  unsqueeze_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:67 in _apply_rotary_emb, code: x1, x2 = torch.chunk(x, 2, dim=-1)
            chunk_1 = torch.chunk(query_rot, 2, dim = -1);  query_rot = None
            x1: "bf16[1, 8, 64]" = chunk_1[0]
            x2: "bf16[1, 8, 64]" = chunk_1[1];  chunk_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:71 in _apply_rotary_emb, code: o1 = x1 * cos - x2 * sin
            mul: "bf16[1, 8, 64]" = x1 * cos_1
            mul_1: "bf16[1, 8, 64]" = x2 * sin_1
            o1: "bf16[1, 8, 64]" = mul - mul_1;  mul = mul_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:72 in _apply_rotary_emb, code: o2 = x2 * cos + x1 * sin
            mul_2: "bf16[1, 8, 64]" = x2 * cos_1;  x2 = cos_1 = None
            mul_3: "bf16[1, 8, 64]" = x1 * sin_1;  x1 = sin_1 = None
            o2: "bf16[1, 8, 64]" = mul_2 + mul_3;  mul_2 = mul_3 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:74 in _apply_rotary_emb, code: return torch.cat((o1, o2), dim=-1)
            query_rot_1: "bf16[1, 8, 128]" = torch.cat((o1, o2), dim = -1);  o1 = o2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:159 in forward_native, code: query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
            cat_1: "bf16[1, 8, 128]" = torch.cat((query_rot_1, query_pass), dim = -1);  query_rot_1 = query_pass = None
            query_1: "bf16[1, 1024]" = cat_1.reshape((1, 1024));  cat_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:162 in forward_native, code: key = key.view(num_tokens, -1, self.head_size)
            key: "bf16[1, 1, 128]" = l_args_2_.view(1, -1, 128);  l_args_2_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:163 in forward_native, code: key_rot = key[..., : self.rotary_dim]
            key_rot: "bf16[1, 1, 128]" = key[(Ellipsis, slice(None, 128, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:164 in forward_native, code: key_pass = key[..., self.rotary_dim :]
            key_pass: "bf16[1, 1, 0]" = key[(Ellipsis, slice(128, None, None))];  key = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:64 in _apply_rotary_emb, code: cos = cos.unsqueeze(-2).to(x.dtype)
            unsqueeze_2: "f32[1, 1, 64]" = cos.unsqueeze(-2);  cos = None
            cos_2: "bf16[1, 1, 64]" = unsqueeze_2.to(torch.bfloat16);  unsqueeze_2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:65 in _apply_rotary_emb, code: sin = sin.unsqueeze(-2).to(x.dtype)
            unsqueeze_3: "f32[1, 1, 64]" = sin.unsqueeze(-2);  sin = None
            sin_2: "bf16[1, 1, 64]" = unsqueeze_3.to(torch.bfloat16);  unsqueeze_3 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:67 in _apply_rotary_emb, code: x1, x2 = torch.chunk(x, 2, dim=-1)
            chunk_2 = torch.chunk(key_rot, 2, dim = -1);  key_rot = None
            x1_1: "bf16[1, 1, 64]" = chunk_2[0]
            x2_1: "bf16[1, 1, 64]" = chunk_2[1];  chunk_2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:71 in _apply_rotary_emb, code: o1 = x1 * cos - x2 * sin
            mul_4: "bf16[1, 1, 64]" = x1_1 * cos_2
            mul_5: "bf16[1, 1, 64]" = x2_1 * sin_2
            o1_1: "bf16[1, 1, 64]" = mul_4 - mul_5;  mul_4 = mul_5 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:72 in _apply_rotary_emb, code: o2 = x2 * cos + x1 * sin
            mul_6: "bf16[1, 1, 64]" = x2_1 * cos_2;  x2_1 = cos_2 = None
            mul_7: "bf16[1, 1, 64]" = x1_1 * sin_2;  x1_1 = sin_2 = None
            o2_1: "bf16[1, 1, 64]" = mul_6 + mul_7;  mul_6 = mul_7 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:74 in _apply_rotary_emb, code: return torch.cat((o1, o2), dim=-1)
            key_rot_1: "bf16[1, 1, 128]" = torch.cat((o1_1, o2_1), dim = -1);  o1_1 = o2_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:166 in forward_native, code: key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
            cat_3: "bf16[1, 1, 128]" = torch.cat((key_rot_1, key_pass), dim = -1);  key_rot_1 = key_pass = None
            key_1: "bf16[1, 128]" = cat_3.reshape((1, 128));  cat_3 = None
            return (query_1, key_1)
            
        class _model(torch.nn.Module):
            def forward(self, l_args_0_: "i64[1]", l_self_buffers_cos_sin_cache_: "f32[40960, 128]", l_args_1_: "bf16[1, 1024]", l_args_2_: "bf16[1, 128]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:149 in forward_native, code: positions = positions.flatten()
                positions: "i64[1]" = l_args_0_.flatten();  l_args_0_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:151 in forward_native, code: cos_sin = self.cos_sin_cache.index_select(0, positions)
                cos_sin: "f32[1, 128]" = l_self_buffers_cos_sin_cache_.index_select(0, positions);  l_self_buffers_cos_sin_cache_ = positions = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:152 in forward_native, code: cos, sin = cos_sin.chunk(2, dim=-1)
                chunk = cos_sin.chunk(2, dim = -1);  cos_sin = None
                cos: "f32[1, 64]" = chunk[0]
                sin: "f32[1, 64]" = chunk[1];  chunk = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:155 in forward_native, code: query = query.view(num_tokens, -1, self.head_size)
                query: "bf16[1, 8, 128]" = l_args_1_.view(1, -1, 128);  l_args_1_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:156 in forward_native, code: query_rot = query[..., : self.rotary_dim]
                query_rot: "bf16[1, 8, 128]" = query[(Ellipsis, slice(None, 128, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:157 in forward_native, code: query_pass = query[..., self.rotary_dim :]
                query_pass: "bf16[1, 8, 0]" = query[(Ellipsis, slice(128, None, None))];  query = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:64 in _apply_rotary_emb, code: cos = cos.unsqueeze(-2).to(x.dtype)
                unsqueeze: "f32[1, 1, 64]" = cos.unsqueeze(-2)
                cos_1: "bf16[1, 1, 64]" = unsqueeze.to(torch.bfloat16);  unsqueeze = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:65 in _apply_rotary_emb, code: sin = sin.unsqueeze(-2).to(x.dtype)
                unsqueeze_1: "f32[1, 1, 64]" = sin.unsqueeze(-2)
                sin_1: "bf16[1, 1, 64]" = unsqueeze_1.to(torch.bfloat16);  unsqueeze_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:67 in _apply_rotary_emb, code: x1, x2 = torch.chunk(x, 2, dim=-1)
                chunk_1 = torch.chunk(query_rot, 2, dim = -1);  query_rot = None
                x1: "bf16[1, 8, 64]" = chunk_1[0]
                x2: "bf16[1, 8, 64]" = chunk_1[1];  chunk_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:71 in _apply_rotary_emb, code: o1 = x1 * cos - x2 * sin
                mul: "bf16[1, 8, 64]" = x1 * cos_1
                mul_1: "bf16[1, 8, 64]" = x2 * sin_1
                o1: "bf16[1, 8, 64]" = mul - mul_1;  mul = mul_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:72 in _apply_rotary_emb, code: o2 = x2 * cos + x1 * sin
                mul_2: "bf16[1, 8, 64]" = x2 * cos_1;  x2 = cos_1 = None
                mul_3: "bf16[1, 8, 64]" = x1 * sin_1;  x1 = sin_1 = None
                o2: "bf16[1, 8, 64]" = mul_2 + mul_3;  mul_2 = mul_3 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:74 in _apply_rotary_emb, code: return torch.cat((o1, o2), dim=-1)
                query_rot_1: "bf16[1, 8, 128]" = torch.cat((o1, o2), dim = -1);  o1 = o2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:159 in forward_native, code: query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
                cat_1: "bf16[1, 8, 128]" = torch.cat((query_rot_1, query_pass), dim = -1);  query_rot_1 = query_pass = None
                query_1: "bf16[1, 1024]" = cat_1.reshape((1, 1024));  cat_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:162 in forward_native, code: key = key.view(num_tokens, -1, self.head_size)
                key: "bf16[1, 1, 128]" = l_args_2_.view(1, -1, 128);  l_args_2_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:163 in forward_native, code: key_rot = key[..., : self.rotary_dim]
                key_rot: "bf16[1, 1, 128]" = key[(Ellipsis, slice(None, 128, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:164 in forward_native, code: key_pass = key[..., self.rotary_dim :]
                key_pass: "bf16[1, 1, 0]" = key[(Ellipsis, slice(128, None, None))];  key = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:64 in _apply_rotary_emb, code: cos = cos.unsqueeze(-2).to(x.dtype)
                unsqueeze_2: "f32[1, 1, 64]" = cos.unsqueeze(-2);  cos = None
                cos_2: "bf16[1, 1, 64]" = unsqueeze_2.to(torch.bfloat16);  unsqueeze_2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:65 in _apply_rotary_emb, code: sin = sin.unsqueeze(-2).to(x.dtype)
                unsqueeze_3: "f32[1, 1, 64]" = sin.unsqueeze(-2);  sin = None
                sin_2: "bf16[1, 1, 64]" = unsqueeze_3.to(torch.bfloat16);  unsqueeze_3 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:67 in _apply_rotary_emb, code: x1, x2 = torch.chunk(x, 2, dim=-1)
                chunk_2 = torch.chunk(key_rot, 2, dim = -1);  key_rot = None
                x1_1: "bf16[1, 1, 64]" = chunk_2[0]
                x2_1: "bf16[1, 1, 64]" = chunk_2[1];  chunk_2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:71 in _apply_rotary_emb, code: o1 = x1 * cos - x2 * sin
                mul_4: "bf16[1, 1, 64]" = x1_1 * cos_2
                mul_5: "bf16[1, 1, 64]" = x2_1 * sin_2
                o1_1: "bf16[1, 1, 64]" = mul_4 - mul_5;  mul_4 = mul_5 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:72 in _apply_rotary_emb, code: o2 = x2 * cos + x1 * sin
                mul_6: "bf16[1, 1, 64]" = x2_1 * cos_2;  x2_1 = cos_2 = None
                mul_7: "bf16[1, 1, 64]" = x1_1 * sin_2;  x1_1 = sin_2 = None
                o2_1: "bf16[1, 1, 64]" = mul_6 + mul_7;  mul_6 = mul_7 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:74 in _apply_rotary_emb, code: return torch.cat((o1, o2), dim=-1)
                key_rot_1: "bf16[1, 1, 128]" = torch.cat((o1_1, o2_1), dim = -1);  o1_1 = o2_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:166 in forward_native, code: key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
                cat_3: "bf16[1, 1, 128]" = torch.cat((key_rot_1, key_pass), dim = -1);  key_rot_1 = key_pass = None
                key_1: "bf16[1, 128]" = cat_3.reshape((1, 128));  cat_3 = None
                return (query_1, key_1)
                