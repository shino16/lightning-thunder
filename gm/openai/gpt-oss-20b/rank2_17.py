# Rank: 2, Graph 17

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0()
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fp8_quantization.py:70 in mxfp8_quantize_sm100, code: out_val = torch.empty(
            out_val: "f8e4m3fn[1, 3072]" = torch.empty((1, 3072), dtype = torch.float8_e4m3fn, device = device(type='cuda', index=2))
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fp8_quantization.py:79 in mxfp8_quantize_sm100, code: out_sf = torch.empty((out_sf_size,), dtype=torch.uint8, device=input.device)
            out_sf: "u8[96]" = torch.empty((96,), dtype = torch.uint8, device = device(type='cuda', index=2))
            return (out_val, out_sf)
            
        class _model(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fp8_quantization.py:70 in mxfp8_quantize_sm100, code: out_val = torch.empty(
                out_val: "f8e4m3fn[1, 3072]" = torch.empty((1, 3072), dtype = torch.float8_e4m3fn, device = device(type='cuda', index=2))
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fp8_quantization.py:79 in mxfp8_quantize_sm100, code: out_sf = torch.empty((out_sf_size,), dtype=torch.uint8, device=input.device)
                out_sf: "u8[96]" = torch.empty((96,), dtype = torch.uint8, device = device(type='cuda', index=2))
                return (out_val, out_sf)
                