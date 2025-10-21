# Rank: 0, Graph 24

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0()
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:340 in apply_allocation, code: output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
            output: "bf16[1, 1, 2880]" = torch.empty((1, 1, 2880), device = device(type='cuda', index=0), dtype = torch.bfloat16)
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:343 in apply_allocation, code: ret["output"] = output[None, :, :]
            getitem: "bf16[1, 1, 1, 2880]" = output[(None, slice(None, None, None), slice(None, None, None))];  output = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:345 in apply_allocation, code: k: torch.empty(v[0], device=allocation.device, dtype=v[1])
            empty_1: "f32[3, 1, 4, 2880]" = torch.empty((3, 1, 4, 2880), device = device(type='cuda', index=0), dtype = torch.float32)
            return (getitem, empty_1)
            
        class _model(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:340 in apply_allocation, code: output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
                output: "bf16[1, 1, 2880]" = torch.empty((1, 1, 2880), device = device(type='cuda', index=0), dtype = torch.bfloat16)
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:343 in apply_allocation, code: ret["output"] = output[None, :, :]
                getitem: "bf16[1, 1, 1, 2880]" = output[(None, slice(None, None, None), slice(None, None, None))];  output = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:345 in apply_allocation, code: k: torch.empty(v[0], device=allocation.device, dtype=v[1])
                empty_1: "f32[3, 1, 4, 2880]" = torch.empty((3, 1, 4, 2880), device = device(type='cuda', index=0), dtype = torch.float32)
                return (getitem, empty_1)
                