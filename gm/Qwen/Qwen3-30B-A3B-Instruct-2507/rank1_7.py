# Rank: 1, Graph 7

class GraphModule(torch.nn.Module):
    def forward(self, l_q_: "bf16[1, 8, 128]", l_paged_kv_cache_0_: "bf16[5914578, 1, 128]", l_paged_kv_cache_1_: "bf16[5914578, 1, 128]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0();  inductor_0 = None
        thunder_1 = self.thunder_1(l_paged_kv_cache_0_, l_paged_kv_cache_1_, l_q_);  l_paged_kv_cache_0_ = l_paged_kv_cache_1_ = l_q_ = None
        getitem = thunder_1[0]
        getitem_1 = thunder_1[1]
        getitem_2 = thunder_1[2]
        getitem_3 = thunder_1[3];  thunder_1 = None
        return (getitem, getitem_1, getitem_2, getitem_3)
        
    class inductor_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:236 in get_compute_capability, code: return torch.cuda.get_device_capability(device.index)
            get_device_capability = torch.cuda.get_device_capability(1);  get_device_capability = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:236 in get_compute_capability, code: return torch.cuda.get_device_capability(device.index)
                get_device_capability = torch.cuda.get_device_capability(1);  get_device_capability = None
                return ()
                
    class thunder_1(torch.nn.Module):
        def forward(self, l_paged_kv_cache_0_: "bf16[5914578, 1, 128]", l_paged_kv_cache_1_: "bf16[5914578, 1, 128]", l_q_: "bf16[1, 8, 128]"):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:89 in _expand_4d, code: return x.unsqueeze(-3)
            k_cache: "bf16[5914578, 1, 1, 128]" = l_paged_kv_cache_0_.unsqueeze(-3);  l_paged_kv_cache_0_ = None
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:89 in _expand_4d, code: return x.unsqueeze(-3)
            v_cache: "bf16[5914578, 1, 1, 128]" = l_paged_kv_cache_1_.unsqueeze(-3);  l_paged_kv_cache_1_ = None
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/decode.py:1273 in run, code: out = torch.empty_like(q)
            out: "bf16[1, 8, 128]" = torch.empty_like(l_q_);  l_q_ = None
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:175 in get_alibi_slopes, code: m = torch.pow(m_0, torch.arange(1, 1 + n))
            arange: "i64[8]" = torch.arange(1, 9)
            m: "f32[8]" = torch.pow(0.5, arange);  arange = None
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:180 in get_alibi_slopes, code: return m.float()
            float_1: "f32[8]" = m.float();  m = None
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:216 in _get_cache_alibi_slopes_buf, code: buf = get_alibi_slopes(num_qo_heads).to(device)
            buf: "f32[8]" = float_1.to(device(type='cuda', index=1));  float_1 = None
            return (k_cache, v_cache, out, buf)
            
        class _model(torch.nn.Module):
            def forward(self, l_paged_kv_cache_0_: "bf16[5914578, 1, 128]", l_paged_kv_cache_1_: "bf16[5914578, 1, 128]", l_q_: "bf16[1, 8, 128]"):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:89 in _expand_4d, code: return x.unsqueeze(-3)
                k_cache: "bf16[5914578, 1, 1, 128]" = l_paged_kv_cache_0_.unsqueeze(-3);  l_paged_kv_cache_0_ = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:89 in _expand_4d, code: return x.unsqueeze(-3)
                v_cache: "bf16[5914578, 1, 1, 128]" = l_paged_kv_cache_1_.unsqueeze(-3);  l_paged_kv_cache_1_ = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/decode.py:1273 in run, code: out = torch.empty_like(q)
                out: "bf16[1, 8, 128]" = torch.empty_like(l_q_);  l_q_ = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:175 in get_alibi_slopes, code: m = torch.pow(m_0, torch.arange(1, 1 + n))
                arange: "i64[8]" = torch.arange(1, 9)
                m: "f32[8]" = torch.pow(0.5, arange);  arange = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:180 in get_alibi_slopes, code: return m.float()
                float_1: "f32[8]" = m.float();  m = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:216 in _get_cache_alibi_slopes_buf, code: buf = get_alibi_slopes(num_qo_heads).to(device)
                buf: "f32[8]" = float_1.to(device(type='cuda', index=1));  float_1 = None
                return (k_cache, v_cache, out, buf)
                