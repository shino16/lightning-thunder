# Rank: 1, Graph 2

class GraphModule(torch.nn.Module):
    def forward(self, l_self_modules_qkv_proj_parameters_weight_: "bf16[1280, 3072]", l_hidden_states_: "bf16[1, 3072]", l_positions_: "i64[1]", l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_: "bf16[135168, 96]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_hidden_states_, l_self_modules_qkv_proj_parameters_weight_, l_positions_, l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_);  l_hidden_states_ = l_self_modules_qkv_proj_parameters_weight_ = l_positions_ = l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1]
        getitem_2 = thunder_0[2]
        getitem_3 = thunder_0[3];  thunder_0 = None
        return (getitem, getitem_1, getitem_2, getitem_3)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_hidden_states_: "bf16[1, 3072]", l_self_modules_qkv_proj_parameters_weight_: "bf16[1280, 3072]", l_positions_: "i64[1]", l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_: "bf16[135168, 96]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel: "bf16[1, 1280]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_qkv_proj_parameters_weight_, None);  l_hidden_states_ = l_self_modules_qkv_proj_parameters_weight_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/llama.py:195 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            split = output_parallel.split([768, 256, 256], dim = -1);  output_parallel = None
            q: "bf16[1, 768]" = split[0]
            k: "bf16[1, 256]" = split[1]
            v: "bf16[1, 256]" = split[2];  split = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:611 in forward, code: query = query.view(*query.shape[:-1], -1, self.head_size)
            query: "bf16[1, 6, 128]" = q.view(1, -1, 128);  q = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:612 in forward, code: key = key.view(*key.shape[:-1], -1, self.head_size)
            key: "bf16[1, 2, 128]" = k.view(1, -1, 128);  k = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:616 in forward, code: torch.any(positions > k).float() * torch.full_like(positions, k)
            gt: "b8[1]" = l_positions_ > 4096
            any_1: "b8[]" = torch.any(gt);  gt = None
            float_1: "f32[]" = any_1.float();  any_1 = None
            full_like: "i64[1]" = torch.full_like(l_positions_, 4096)
            mul: "f32[1]" = float_1 * full_like;  float_1 = full_like = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:617 in forward, code: ).long()
            long_prompt_offset: "i64[1]" = mul.long();  mul = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:619 in forward, code: torch.add(positions, long_prompt_offset)
            idx: "i64[1]" = torch.add(l_positions_, long_prompt_offset);  l_positions_ = long_prompt_offset = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:623 in forward, code: self.long_short_cos_sin_cache: torch.Tensor = self.long_short_cos_sin_cache.to(
            to: "bf16[135168, 96]" = l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_.to(device(type='cuda', index=1));  l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:627 in forward, code: cos_sin = torch.index_select(self.long_short_cos_sin_cache, 0, idx)
            cos_sin: "bf16[1, 96]" = torch.index_select(to, 0, idx);  idx = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:629 in forward, code: cos, sin = cos_sin.chunk(2, dim=-1)
            chunk = cos_sin.chunk(2, dim = -1);  cos_sin = None
            cos: "bf16[1, 48]" = chunk[0]
            sin: "bf16[1, 48]" = chunk[1];  chunk = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:630 in forward, code: cos = cos.repeat(1, 2).unsqueeze(-2)
            repeat: "bf16[1, 96]" = cos.repeat(1, 2);  cos = None
            cos_1: "bf16[1, 1, 96]" = repeat.unsqueeze(-2);  repeat = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:631 in forward, code: sin = sin.repeat(1, 2).unsqueeze(-2)
            repeat_1: "bf16[1, 96]" = sin.repeat(1, 2);  sin = None
            sin_1: "bf16[1, 1, 96]" = repeat_1.unsqueeze(-2);  repeat_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:633 in forward, code: query_rot = query[..., : self.rotary_dim]
            query_rot: "bf16[1, 6, 96]" = query[(Ellipsis, slice(None, 96, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:634 in forward, code: query_pass = query[..., self.rotary_dim :]
            query_pass: "bf16[1, 6, 32]" = query[(Ellipsis, slice(96, None, None))];  query = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:635 in forward, code: query_rot = query_rot * cos + _rotate_neox(query_rot) * sin
            mul_1: "bf16[1, 6, 96]" = query_rot * cos_1
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:38 in _rotate_neox, code: x1 = x[..., : x.shape[-1] // 2]
            x1: "bf16[1, 6, 48]" = query_rot[(Ellipsis, slice(None, 48, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:39 in _rotate_neox, code: x2 = x[..., x.shape[-1] // 2 :]
            x2: "bf16[1, 6, 48]" = query_rot[(Ellipsis, slice(48, None, None))];  query_rot = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:40 in _rotate_neox, code: return torch.cat((-x2, x1), dim=-1)
            neg: "bf16[1, 6, 48]" = -x2;  x2 = None
            cat: "bf16[1, 6, 96]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:635 in forward, code: query_rot = query_rot * cos + _rotate_neox(query_rot) * sin
            mul_2: "bf16[1, 6, 96]" = cat * sin_1;  cat = None
            query_rot_1: "bf16[1, 6, 96]" = mul_1 + mul_2;  mul_1 = mul_2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:636 in forward, code: query = torch.cat((query_rot, query_pass), dim=-1)
            query_1: "bf16[1, 6, 128]" = torch.cat((query_rot_1, query_pass), dim = -1);  query_rot_1 = query_pass = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:638 in forward, code: key_rot = key[..., : self.rotary_dim]
            key_rot: "bf16[1, 2, 96]" = key[(Ellipsis, slice(None, 96, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:639 in forward, code: key_pass = key[..., self.rotary_dim :]
            key_pass: "bf16[1, 2, 32]" = key[(Ellipsis, slice(96, None, None))];  key = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:640 in forward, code: key_rot = key_rot * cos + _rotate_neox(key_rot) * sin
            mul_3: "bf16[1, 2, 96]" = key_rot * cos_1;  cos_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:38 in _rotate_neox, code: x1 = x[..., : x.shape[-1] // 2]
            x1_1: "bf16[1, 2, 48]" = key_rot[(Ellipsis, slice(None, 48, None))]
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:39 in _rotate_neox, code: x2 = x[..., x.shape[-1] // 2 :]
            x2_1: "bf16[1, 2, 48]" = key_rot[(Ellipsis, slice(48, None, None))];  key_rot = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:40 in _rotate_neox, code: return torch.cat((-x2, x1), dim=-1)
            neg_1: "bf16[1, 2, 48]" = -x2_1;  x2_1 = None
            cat_2: "bf16[1, 2, 96]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:640 in forward, code: key_rot = key_rot * cos + _rotate_neox(key_rot) * sin
            mul_4: "bf16[1, 2, 96]" = cat_2 * sin_1;  cat_2 = sin_1 = None
            key_rot_1: "bf16[1, 2, 96]" = mul_3 + mul_4;  mul_3 = mul_4 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:641 in forward, code: key = torch.cat((key_rot, key_pass), dim=-1)
            key_1: "bf16[1, 2, 128]" = torch.cat((key_rot_1, key_pass), dim = -1);  key_rot_1 = key_pass = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:643 in forward, code: return query.flatten(-2), key.flatten(-2)
            q_1: "bf16[1, 768]" = query_1.flatten(-2);  query_1 = None
            k_1: "bf16[1, 256]" = key_1.flatten(-2);  key_1 = None
            return (q_1, k_1, v, to)
            
        class _model(torch.nn.Module):
            def forward(self, l_hidden_states_: "bf16[1, 3072]", l_self_modules_qkv_proj_parameters_weight_: "bf16[1280, 3072]", l_positions_: "i64[1]", l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_: "bf16[135168, 96]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel: "bf16[1, 1280]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_qkv_proj_parameters_weight_, None);  l_hidden_states_ = l_self_modules_qkv_proj_parameters_weight_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/llama.py:195 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
                split = output_parallel.split([768, 256, 256], dim = -1);  output_parallel = None
                q: "bf16[1, 768]" = split[0]
                k: "bf16[1, 256]" = split[1]
                v: "bf16[1, 256]" = split[2];  split = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:611 in forward, code: query = query.view(*query.shape[:-1], -1, self.head_size)
                query: "bf16[1, 6, 128]" = q.view(1, -1, 128);  q = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:612 in forward, code: key = key.view(*key.shape[:-1], -1, self.head_size)
                key: "bf16[1, 2, 128]" = k.view(1, -1, 128);  k = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:616 in forward, code: torch.any(positions > k).float() * torch.full_like(positions, k)
                gt: "b8[1]" = l_positions_ > 4096
                any_1: "b8[]" = torch.any(gt);  gt = None
                float_1: "f32[]" = any_1.float();  any_1 = None
                full_like: "i64[1]" = torch.full_like(l_positions_, 4096)
                mul: "f32[1]" = float_1 * full_like;  float_1 = full_like = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:617 in forward, code: ).long()
                long_prompt_offset: "i64[1]" = mul.long();  mul = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:619 in forward, code: torch.add(positions, long_prompt_offset)
                idx: "i64[1]" = torch.add(l_positions_, long_prompt_offset);  l_positions_ = long_prompt_offset = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:623 in forward, code: self.long_short_cos_sin_cache: torch.Tensor = self.long_short_cos_sin_cache.to(
                to: "bf16[135168, 96]" = l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_.to(device(type='cuda', index=1));  l_self_modules_rotary_emb_buffers_long_short_cos_sin_cache_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:627 in forward, code: cos_sin = torch.index_select(self.long_short_cos_sin_cache, 0, idx)
                cos_sin: "bf16[1, 96]" = torch.index_select(to, 0, idx);  idx = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:629 in forward, code: cos, sin = cos_sin.chunk(2, dim=-1)
                chunk = cos_sin.chunk(2, dim = -1);  cos_sin = None
                cos: "bf16[1, 48]" = chunk[0]
                sin: "bf16[1, 48]" = chunk[1];  chunk = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:630 in forward, code: cos = cos.repeat(1, 2).unsqueeze(-2)
                repeat: "bf16[1, 96]" = cos.repeat(1, 2);  cos = None
                cos_1: "bf16[1, 1, 96]" = repeat.unsqueeze(-2);  repeat = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:631 in forward, code: sin = sin.repeat(1, 2).unsqueeze(-2)
                repeat_1: "bf16[1, 96]" = sin.repeat(1, 2);  sin = None
                sin_1: "bf16[1, 1, 96]" = repeat_1.unsqueeze(-2);  repeat_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:633 in forward, code: query_rot = query[..., : self.rotary_dim]
                query_rot: "bf16[1, 6, 96]" = query[(Ellipsis, slice(None, 96, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:634 in forward, code: query_pass = query[..., self.rotary_dim :]
                query_pass: "bf16[1, 6, 32]" = query[(Ellipsis, slice(96, None, None))];  query = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:635 in forward, code: query_rot = query_rot * cos + _rotate_neox(query_rot) * sin
                mul_1: "bf16[1, 6, 96]" = query_rot * cos_1
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:38 in _rotate_neox, code: x1 = x[..., : x.shape[-1] // 2]
                x1: "bf16[1, 6, 48]" = query_rot[(Ellipsis, slice(None, 48, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:39 in _rotate_neox, code: x2 = x[..., x.shape[-1] // 2 :]
                x2: "bf16[1, 6, 48]" = query_rot[(Ellipsis, slice(48, None, None))];  query_rot = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:40 in _rotate_neox, code: return torch.cat((-x2, x1), dim=-1)
                neg: "bf16[1, 6, 48]" = -x2;  x2 = None
                cat: "bf16[1, 6, 96]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:635 in forward, code: query_rot = query_rot * cos + _rotate_neox(query_rot) * sin
                mul_2: "bf16[1, 6, 96]" = cat * sin_1;  cat = None
                query_rot_1: "bf16[1, 6, 96]" = mul_1 + mul_2;  mul_1 = mul_2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:636 in forward, code: query = torch.cat((query_rot, query_pass), dim=-1)
                query_1: "bf16[1, 6, 128]" = torch.cat((query_rot_1, query_pass), dim = -1);  query_rot_1 = query_pass = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:638 in forward, code: key_rot = key[..., : self.rotary_dim]
                key_rot: "bf16[1, 2, 96]" = key[(Ellipsis, slice(None, 96, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:639 in forward, code: key_pass = key[..., self.rotary_dim :]
                key_pass: "bf16[1, 2, 32]" = key[(Ellipsis, slice(96, None, None))];  key = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:640 in forward, code: key_rot = key_rot * cos + _rotate_neox(key_rot) * sin
                mul_3: "bf16[1, 2, 96]" = key_rot * cos_1;  cos_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:38 in _rotate_neox, code: x1 = x[..., : x.shape[-1] // 2]
                x1_1: "bf16[1, 2, 48]" = key_rot[(Ellipsis, slice(None, 48, None))]
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:39 in _rotate_neox, code: x2 = x[..., x.shape[-1] // 2 :]
                x2_1: "bf16[1, 2, 48]" = key_rot[(Ellipsis, slice(48, None, None))];  key_rot = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:40 in _rotate_neox, code: return torch.cat((-x2, x1), dim=-1)
                neg_1: "bf16[1, 2, 48]" = -x2_1;  x2_1 = None
                cat_2: "bf16[1, 2, 96]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:640 in forward, code: key_rot = key_rot * cos + _rotate_neox(key_rot) * sin
                mul_4: "bf16[1, 2, 96]" = cat_2 * sin_1;  cat_2 = sin_1 = None
                key_rot_1: "bf16[1, 2, 96]" = mul_3 + mul_4;  mul_3 = mul_4 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:641 in forward, code: key = torch.cat((key_rot, key_pass), dim=-1)
                key_1: "bf16[1, 2, 128]" = torch.cat((key_rot_1, key_pass), dim = -1);  key_rot_1 = key_pass = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/rotary_embedding.py:643 in forward, code: return query.flatten(-2), key.flatten(-2)
                q_1: "bf16[1, 768]" = query_1.flatten(-2);  query_1 = None
                k_1: "bf16[1, 256]" = key_1.flatten(-2);  key_1 = None
                return (q_1, k_1, v, to)
                