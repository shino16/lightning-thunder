# Rank: 0, Graph 20

class GraphModule(torch.nn.Module):
    def forward(self, l_x_storage_data: "bf16[1, 2880]", l_routing_data_expt_data_token_offs_pad_64_: "i32[33]", l_w_storage_data: "bf16[32, 2880, 1440]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_routing_data_expt_data_token_offs_pad_64_, l_x_storage_data, l_w_storage_data);  l_routing_data_expt_data_token_offs_pad_64_ = l_x_storage_data = l_w_storage_data = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1]
        getitem_2 = thunder_0[2]
        getitem_3 = thunder_0[3]
        getitem_4 = thunder_0[4];  thunder_0 = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_routing_data_expt_data_token_offs_pad_64_: "i32[33]", l_x_storage_data: "bf16[1, 2880]", l_w_storage_data: "bf16[32, 2880, 1440]"):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:340 in apply_allocation, code: output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
            output: "bf16[1, 4, 720]" = torch.empty((1, 4, 720), device = device(type='cuda', index=0), dtype = torch.bfloat16)
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:343 in apply_allocation, code: ret["output"] = output[None, :, :]
            out0: "bf16[1, 1, 4, 720]" = output[(None, slice(None, None, None), slice(None, None, None))];  output = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:345 in apply_allocation, code: k: torch.empty(v[0], device=allocation.device, dtype=v[1])
            out0_1: "f32[11, 1, 4, 1440]" = torch.empty((11, 1, 4, 1440), device = device(type='cuda', index=0), dtype = torch.float32)
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:496 in torch_dynamo_resume_in_matmul_ogs_at_444, code: expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[block_m][-1]
            expt_hist_sum: "i32[]" = l_routing_data_expt_data_token_offs_pad_64_[-1];  l_routing_data_expt_data_token_offs_pad_64_ = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:359 in _canonicalize_storage, code: new_storage_data = storage.data.view(new_storage_shape)
            new_storage_data: "bf16[1, 2880]" = l_x_storage_data.view([1, 2880]);  l_x_storage_data = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:359 in _canonicalize_storage, code: new_storage_data = storage.data.view(new_storage_shape)
            new_storage_data_1: "bf16[32, 2880, 1440]" = l_w_storage_data.view([32, 2880, 1440]);  l_w_storage_data = None
            return (new_storage_data, new_storage_data_1, out0, out0_1, expt_hist_sum)
            
        class _model(torch.nn.Module):
            def forward(self, l_routing_data_expt_data_token_offs_pad_64_: "i32[33]", l_x_storage_data: "bf16[1, 2880]", l_w_storage_data: "bf16[32, 2880, 1440]"):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:340 in apply_allocation, code: output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
                output: "bf16[1, 4, 720]" = torch.empty((1, 4, 720), device = device(type='cuda', index=0), dtype = torch.bfloat16)
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:343 in apply_allocation, code: ret["output"] = output[None, :, :]
                out0: "bf16[1, 1, 4, 720]" = output[(None, slice(None, None, None), slice(None, None, None))];  output = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:345 in apply_allocation, code: k: torch.empty(v[0], device=allocation.device, dtype=v[1])
                out0_1: "f32[11, 1, 4, 1440]" = torch.empty((11, 1, 4, 1440), device = device(type='cuda', index=0), dtype = torch.float32)
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:496 in torch_dynamo_resume_in_matmul_ogs_at_444, code: expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[block_m][-1]
                expt_hist_sum: "i32[]" = l_routing_data_expt_data_token_offs_pad_64_[-1];  l_routing_data_expt_data_token_offs_pad_64_ = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:359 in _canonicalize_storage, code: new_storage_data = storage.data.view(new_storage_shape)
                new_storage_data: "bf16[1, 2880]" = l_x_storage_data.view([1, 2880]);  l_x_storage_data = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:359 in _canonicalize_storage, code: new_storage_data = storage.data.view(new_storage_shape)
                new_storage_data_1: "bf16[32, 2880, 1440]" = l_w_storage_data.view([32, 2880, 1440]);  l_w_storage_data = None
                return (new_storage_data, new_storage_data_1, out0, out0_1, expt_hist_sum)
                