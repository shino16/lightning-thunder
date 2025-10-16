# Rank: 0, Graph 8

class GraphModule(torch.nn.Module):
    def forward(self, l_q_: "bf16[1, 8, 128]", l_paged_kv_cache_0_: "bf16[4253378, 2, 128]", l_paged_kv_cache_1_: "bf16[4253378, 2, 128]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0();  inductor_0 = None
        thunder_1 = self.thunder_1(l_paged_kv_cache_0_, l_paged_kv_cache_1_, l_q_);  l_paged_kv_cache_0_ = l_paged_kv_cache_1_ = l_q_ = None
        getitem = thunder_1[0]
        getitem_1 = thunder_1[1]
        getitem_2 = thunder_1[2];  thunder_1 = None
        return (getitem, getitem_1, getitem_2)
        
    class inductor_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:236 in get_compute_capability, code: return torch.cuda.get_device_capability(device.index)
            get_device_capability = torch.cuda.get_device_capability(0);  get_device_capability = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:236 in get_compute_capability, code: return torch.cuda.get_device_capability(device.index)
                get_device_capability = torch.cuda.get_device_capability(0);  get_device_capability = None
                return ()
                
    class thunder_1(torch.nn.Module):
        def forward(self, l_paged_kv_cache_0_: "bf16[4253378, 2, 128]", l_paged_kv_cache_1_: "bf16[4253378, 2, 128]", l_q_: "bf16[1, 8, 128]"):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:89 in _expand_4d, code: return x.unsqueeze(-3)
            k_cache: "bf16[4253378, 1, 2, 128]" = l_paged_kv_cache_0_.unsqueeze(-3);  l_paged_kv_cache_0_ = None
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:89 in _expand_4d, code: return x.unsqueeze(-3)
            v_cache: "bf16[4253378, 1, 2, 128]" = l_paged_kv_cache_1_.unsqueeze(-3);  l_paged_kv_cache_1_ = None
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/decode.py:1273 in run, code: out = torch.empty_like(q)
            out: "bf16[1, 8, 128]" = torch.empty_like(l_q_);  l_q_ = None
            return (k_cache, v_cache, out)
            
        class _model(torch.nn.Module):
            def forward(self, l_paged_kv_cache_0_: "bf16[4253378, 2, 128]", l_paged_kv_cache_1_: "bf16[4253378, 2, 128]", l_q_: "bf16[1, 8, 128]"):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:89 in _expand_4d, code: return x.unsqueeze(-3)
                k_cache: "bf16[4253378, 1, 2, 128]" = l_paged_kv_cache_0_.unsqueeze(-3);  l_paged_kv_cache_0_ = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:89 in _expand_4d, code: return x.unsqueeze(-3)
                v_cache: "bf16[4253378, 1, 2, 128]" = l_paged_kv_cache_1_.unsqueeze(-3);  l_paged_kv_cache_1_ = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/decode.py:1273 in run, code: out = torch.empty_like(q)
                out: "bf16[1, 8, 128]" = torch.empty_like(l_q_);  l_q_ = None
                return (k_cache, v_cache, out)
                