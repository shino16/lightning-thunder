# Rank: 3, Graph 6

class GraphModule(torch.nn.Module):
    def forward(self, l_k_: "bf16[1, 128]", l_v_: "bf16[1, 128]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_k_, l_v_);  l_k_ = l_v_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_k_: "bf16[1, 128]", l_v_: "bf16[1, 128]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/radix_attention.py:103 in forward, code: k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            k: "bf16[1, 1, 128]" = l_k_.view(-1, 1, 128);  l_k_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/radix_attention.py:104 in forward, code: v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            v: "bf16[1, 1, 128]" = l_v_.view(-1, 1, 128);  l_v_ = None
            return (k, v)
            
        class _model(torch.nn.Module):
            def forward(self, l_k_: "bf16[1, 128]", l_v_: "bf16[1, 128]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/radix_attention.py:103 in forward, code: k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                k: "bf16[1, 1, 128]" = l_k_.view(-1, 1, 128);  l_k_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/radix_attention.py:104 in forward, code: v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
                v: "bf16[1, 1, 128]" = l_v_.view(-1, 1, 128);  l_v_ = None
                return (k, v)
                