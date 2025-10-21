# Rank: 0, Graph 18

class GraphModule(torch.nn.Module):
    def forward(self, l_w1_: "bf16[32, 2880, 1440]", l_w2_: "bf16[32, 720, 2880]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_w1_, l_w2_);  l_w1_ = l_w2_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_w1_: "bf16[32, 2880, 1440]", l_w2_: "bf16[32, 720, 2880]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:27 in quantize, code: return w.to(torch.bfloat16), InFlexData()
            w1: "bf16[32, 2880, 1440]" = l_w1_.to(torch.bfloat16);  l_w1_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:27 in quantize, code: return w.to(torch.bfloat16), InFlexData()
            w2: "bf16[32, 720, 2880]" = l_w2_.to(torch.bfloat16);  l_w2_ = None
            return (w1, w2)
            
        class _model(torch.nn.Module):
            def forward(self, l_w1_: "bf16[32, 2880, 1440]", l_w2_: "bf16[32, 720, 2880]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:27 in quantize, code: return w.to(torch.bfloat16), InFlexData()
                w1: "bf16[32, 2880, 1440]" = l_w1_.to(torch.bfloat16);  l_w1_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:27 in quantize, code: return w.to(torch.bfloat16), InFlexData()
                w2: "bf16[32, 720, 2880]" = l_w2_.to(torch.bfloat16);  l_w2_ = None
                return (w1, w2)
                