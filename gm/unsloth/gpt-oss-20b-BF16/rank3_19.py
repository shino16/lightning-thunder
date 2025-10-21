# Rank: 3, Graph 19

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0();  inductor_0 = None
        return ()
        
    class inductor_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/target_info.py:46 in cuda_capability_geq, code: cached_capabilities["cuda"] = torch.cuda.get_device_capability()
            get_device_capability = torch.cuda.get_device_capability();  get_device_capability = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:442 in torch_dynamo_resume_in_matmul_ogs_at_410, code: can_use_tma = can_use_tma and (torch.cuda.get_device_capability()[0] > 9 or bitwidth(w.dtype) != 4)
            get_device_capability_1 = torch.cuda.get_device_capability();  get_device_capability_1 = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/target_info.py:46 in cuda_capability_geq, code: cached_capabilities["cuda"] = torch.cuda.get_device_capability()
                get_device_capability = torch.cuda.get_device_capability();  get_device_capability = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:442 in torch_dynamo_resume_in_matmul_ogs_at_410, code: can_use_tma = can_use_tma and (torch.cuda.get_device_capability()[0] > 9 or bitwidth(w.dtype) != 4)
                get_device_capability_1 = torch.cuda.get_device_capability();  get_device_capability_1 = None
                return ()
                