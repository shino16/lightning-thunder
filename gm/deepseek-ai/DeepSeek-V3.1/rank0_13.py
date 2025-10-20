# Rank: 0, Graph 13

class GraphModule(torch.nn.Module):
    def forward(self, l_k_: "bf16[1, 1, 512]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_k_);  l_k_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_k_: "bf16[1, 1, 512]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/radix_attention.py:106 in forward, code: k = k.view(-1, self.tp_k_head_num, self.v_head_dim)
            k: "bf16[1, 1, 512]" = l_k_.view(-1, 1, 512);  l_k_ = None
            return k
            
        class _model(torch.nn.Module):
            def forward(self, l_k_: "bf16[1, 1, 512]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/radix_attention.py:106 in forward, code: k = k.view(-1, self.tp_k_head_num, self.v_head_dim)
                k: "bf16[1, 1, 512]" = l_k_.view(-1, 1, 512);  l_k_ = None
                return k
                