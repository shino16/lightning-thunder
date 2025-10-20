# Rank: 3, Graph 12

class GraphModule(torch.nn.Module):
    def forward(self, l_args_1_: "bf16[1, 32, 64]", l_args_2_: "bf16[1, 1, 64]", l_self_buffers_cos_sin_cache_: "f32[163840, 64]", l_args_0_: "i64[1]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_args_1_, l_args_2_, l_self_buffers_cos_sin_cache_, l_args_0_);  l_args_1_ = l_args_2_ = l_self_buffers_cos_sin_cache_ = l_args_0_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1]
        getitem_2 = thunder_0[2];  thunder_0 = None
        return (getitem, getitem_1, getitem_2)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_args_1_: "bf16[1, 32, 64]", l_args_2_: "bf16[1, 1, 64]", l_self_buffers_cos_sin_cache_: "f32[163840, 64]", l_args_0_: "i64[1]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:746 in forward_native, code: query_rot = query[..., : self.rotary_dim]
            query_rot: "bf16[1, 32, 64]" = l_args_1_[(Ellipsis, slice(None, 64, None))];  l_args_1_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:747 in forward_native, code: key_rot = key[..., : self.rotary_dim]
            key_rot: "bf16[1, 1, 64]" = l_args_2_[(Ellipsis, slice(None, 64, None))];  l_args_2_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:752 in forward_native, code: self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
            to: "f32[163840, 64]" = l_self_buffers_cos_sin_cache_.to(device(type='cuda', index=3));  l_self_buffers_cos_sin_cache_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:753 in forward_native, code: cos_sin = self.cos_sin_cache[
            cos_sin: "f32[1, 64]" = to[l_args_0_];  l_args_0_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:756 in forward_native, code: cos, sin = cos_sin.chunk(2, dim=-1)
            chunk = cos_sin.chunk(2, dim = -1);  cos_sin = None
            cos: "f32[1, 32]" = chunk[0]
            sin: "f32[1, 32]" = chunk[1];  chunk = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:763 in forward_native, code: cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            repeat_interleave: "f32[1, 64]" = cos.repeat_interleave(2, dim = -1);  cos = None
            cos_1: "f32[1, 1, 64]" = repeat_interleave.unsqueeze(-2);  repeat_interleave = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:764 in forward_native, code: sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
            repeat_interleave_1: "f32[1, 64]" = sin.repeat_interleave(2, dim = -1);  sin = None
            sin_1: "f32[1, 1, 64]" = repeat_interleave_1.unsqueeze(-2);  repeat_interleave_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:767 in forward_native, code: query_rot = query_rot * cos + rotate_fn(query_rot) * sin
            mul: "f32[1, 32, 64]" = query_rot * cos_1
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:44 in _rotate_gptj, code: x1 = x[..., ::2]
            x1: "bf16[1, 32, 32]" = query_rot[(Ellipsis, slice(None, None, 2))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:45 in _rotate_gptj, code: x2 = x[..., 1::2]
            x2: "bf16[1, 32, 32]" = query_rot[(Ellipsis, slice(1, None, 2))];  query_rot = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:46 in _rotate_gptj, code: x = torch.stack((-x2, x1), dim=-1)
            neg: "bf16[1, 32, 32]" = -x2;  x2 = None
            x: "bf16[1, 32, 32, 2]" = torch.stack((neg, x1), dim = -1);  neg = x1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:47 in _rotate_gptj, code: return x.flatten(-2)
            flatten: "bf16[1, 32, 64]" = x.flatten(-2);  x = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:767 in forward_native, code: query_rot = query_rot * cos + rotate_fn(query_rot) * sin
            mul_1: "f32[1, 32, 64]" = flatten * sin_1;  flatten = None
            query_rot_1: "f32[1, 32, 64]" = mul + mul_1;  mul = mul_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:768 in forward_native, code: key_rot = key_rot * cos + rotate_fn(key_rot) * sin
            mul_2: "f32[1, 1, 64]" = key_rot * cos_1;  cos_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:44 in _rotate_gptj, code: x1 = x[..., ::2]
            x1_1: "bf16[1, 1, 32]" = key_rot[(Ellipsis, slice(None, None, 2))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:45 in _rotate_gptj, code: x2 = x[..., 1::2]
            x2_1: "bf16[1, 1, 32]" = key_rot[(Ellipsis, slice(1, None, 2))];  key_rot = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:46 in _rotate_gptj, code: x = torch.stack((-x2, x1), dim=-1)
            neg_1: "bf16[1, 1, 32]" = -x2_1;  x2_1 = None
            x_1: "bf16[1, 1, 32, 2]" = torch.stack((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:47 in _rotate_gptj, code: return x.flatten(-2)
            flatten_1: "bf16[1, 1, 64]" = x_1.flatten(-2);  x_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:768 in forward_native, code: key_rot = key_rot * cos + rotate_fn(key_rot) * sin
            mul_3: "f32[1, 1, 64]" = flatten_1 * sin_1;  flatten_1 = sin_1 = None
            key_rot_1: "f32[1, 1, 64]" = mul_2 + mul_3;  mul_2 = mul_3 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:776 in forward_native, code: return query.to(dtype), key.to(dtype)
            to_1: "bf16[1, 32, 64]" = query_rot_1.to(torch.bfloat16);  query_rot_1 = None
            to_2: "bf16[1, 1, 64]" = key_rot_1.to(torch.bfloat16);  key_rot_1 = None
            return (to_1, to_2, to)
            
        class _model(torch.nn.Module):
            def forward(self, l_args_1_: "bf16[1, 32, 64]", l_args_2_: "bf16[1, 1, 64]", l_self_buffers_cos_sin_cache_: "f32[163840, 64]", l_args_0_: "i64[1]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:746 in forward_native, code: query_rot = query[..., : self.rotary_dim]
                query_rot: "bf16[1, 32, 64]" = l_args_1_[(Ellipsis, slice(None, 64, None))];  l_args_1_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:747 in forward_native, code: key_rot = key[..., : self.rotary_dim]
                key_rot: "bf16[1, 1, 64]" = l_args_2_[(Ellipsis, slice(None, 64, None))];  l_args_2_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:752 in forward_native, code: self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
                to: "f32[163840, 64]" = l_self_buffers_cos_sin_cache_.to(device(type='cuda', index=3));  l_self_buffers_cos_sin_cache_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:753 in forward_native, code: cos_sin = self.cos_sin_cache[
                cos_sin: "f32[1, 64]" = to[l_args_0_];  l_args_0_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:756 in forward_native, code: cos, sin = cos_sin.chunk(2, dim=-1)
                chunk = cos_sin.chunk(2, dim = -1);  cos_sin = None
                cos: "f32[1, 32]" = chunk[0]
                sin: "f32[1, 32]" = chunk[1];  chunk = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:763 in forward_native, code: cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
                repeat_interleave: "f32[1, 64]" = cos.repeat_interleave(2, dim = -1);  cos = None
                cos_1: "f32[1, 1, 64]" = repeat_interleave.unsqueeze(-2);  repeat_interleave = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:764 in forward_native, code: sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
                repeat_interleave_1: "f32[1, 64]" = sin.repeat_interleave(2, dim = -1);  sin = None
                sin_1: "f32[1, 1, 64]" = repeat_interleave_1.unsqueeze(-2);  repeat_interleave_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:767 in forward_native, code: query_rot = query_rot * cos + rotate_fn(query_rot) * sin
                mul: "f32[1, 32, 64]" = query_rot * cos_1
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:44 in _rotate_gptj, code: x1 = x[..., ::2]
                x1: "bf16[1, 32, 32]" = query_rot[(Ellipsis, slice(None, None, 2))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:45 in _rotate_gptj, code: x2 = x[..., 1::2]
                x2: "bf16[1, 32, 32]" = query_rot[(Ellipsis, slice(1, None, 2))];  query_rot = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:46 in _rotate_gptj, code: x = torch.stack((-x2, x1), dim=-1)
                neg: "bf16[1, 32, 32]" = -x2;  x2 = None
                x: "bf16[1, 32, 32, 2]" = torch.stack((neg, x1), dim = -1);  neg = x1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:47 in _rotate_gptj, code: return x.flatten(-2)
                flatten: "bf16[1, 32, 64]" = x.flatten(-2);  x = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:767 in forward_native, code: query_rot = query_rot * cos + rotate_fn(query_rot) * sin
                mul_1: "f32[1, 32, 64]" = flatten * sin_1;  flatten = None
                query_rot_1: "f32[1, 32, 64]" = mul + mul_1;  mul = mul_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:768 in forward_native, code: key_rot = key_rot * cos + rotate_fn(key_rot) * sin
                mul_2: "f32[1, 1, 64]" = key_rot * cos_1;  cos_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:44 in _rotate_gptj, code: x1 = x[..., ::2]
                x1_1: "bf16[1, 1, 32]" = key_rot[(Ellipsis, slice(None, None, 2))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:45 in _rotate_gptj, code: x2 = x[..., 1::2]
                x2_1: "bf16[1, 1, 32]" = key_rot[(Ellipsis, slice(1, None, 2))];  key_rot = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:46 in _rotate_gptj, code: x = torch.stack((-x2, x1), dim=-1)
                neg_1: "bf16[1, 1, 32]" = -x2_1;  x2_1 = None
                x_1: "bf16[1, 1, 32, 2]" = torch.stack((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:47 in _rotate_gptj, code: return x.flatten(-2)
                flatten_1: "bf16[1, 1, 64]" = x_1.flatten(-2);  x_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:768 in forward_native, code: key_rot = key_rot * cos + rotate_fn(key_rot) * sin
                mul_3: "f32[1, 1, 64]" = flatten_1 * sin_1;  flatten_1 = sin_1 = None
                key_rot_1: "f32[1, 1, 64]" = mul_2 + mul_3;  mul_2 = mul_3 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:776 in forward_native, code: return query.to(dtype), key.to(dtype)
                to_1: "bf16[1, 32, 64]" = query_rot_1.to(torch.bfloat16);  query_rot_1 = None
                to_2: "bf16[1, 1, 64]" = key_rot_1.to(torch.bfloat16);  key_rot_1 = None
                return (to_1, to_2, to)
                