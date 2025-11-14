# Rank: 1, Graph 3

class GraphModule(torch.nn.Module):
    def forward(self, l_x_: "bf16[1, 5120]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_x_);  l_x_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_x_: "bf16[1, 5120]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:552 in sglang_per_token_quant_fp8, code: x_q = torch.empty_like(x, device=x.device, dtype=dtype)
            x_q: "f8e4m3fn[1, 5120]" = torch.empty_like(l_x_, device = device(type='cuda', index=1), dtype = torch.float8_e4m3fn);  l_x_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:553 in sglang_per_token_quant_fp8, code: x_s = torch.empty(
            x_s: "f32[1, 1]" = torch.empty(1, 1, device = device(type='cuda', index=1), dtype = torch.float32)
            return (x_q, x_s)
            
        class _model(torch.nn.Module):
            def forward(self, l_x_: "bf16[1, 5120]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:552 in sglang_per_token_quant_fp8, code: x_q = torch.empty_like(x, device=x.device, dtype=dtype)
                x_q: "f8e4m3fn[1, 5120]" = torch.empty_like(l_x_, device = device(type='cuda', index=1), dtype = torch.float8_e4m3fn);  l_x_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:553 in sglang_per_token_quant_fp8, code: x_s = torch.empty(
                x_s: "f32[1, 1]" = torch.empty(1, 1, device = device(type='cuda', index=1), dtype = torch.float32)
                return (x_q, x_s)
                