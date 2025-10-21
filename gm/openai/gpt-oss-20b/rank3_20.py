# Rank: 3, Graph 20

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0()
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        inductor_1 = self.inductor_1();  inductor_1 = None
        thunder_2 = self.thunder_2()
        return (getitem, getitem_1, thunder_2)
        
    class thunder_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py:1293 in trtllm_fp4_block_scale_moe_op, code: topk_ids = torch.empty(
            topk_ids: "i32[1, 4]" = torch.empty(1, 4, dtype = torch.int32, device = device(type='cuda', index=3))
            
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py:1297 in trtllm_fp4_block_scale_moe_op, code: expert_weights = torch.empty(
            expert_weights: "bf16[1, 4]" = torch.empty(1, 4, dtype = torch.bfloat16, device = device(type='cuda', index=3))
            return (topk_ids, expert_weights)
            
        class _model(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py:1293 in trtllm_fp4_block_scale_moe_op, code: topk_ids = torch.empty(
                topk_ids: "i32[1, 4]" = torch.empty(1, 4, dtype = torch.int32, device = device(type='cuda', index=3))
                
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py:1297 in trtllm_fp4_block_scale_moe_op, code: expert_weights = torch.empty(
                expert_weights: "bf16[1, 4]" = torch.empty(1, 4, dtype = torch.bfloat16, device = device(type='cuda', index=3))
                return (topk_ids, expert_weights)
                
    class inductor_1(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:236 in get_compute_capability, code: return torch.cuda.get_device_capability(device.index)
            get_device_capability = torch.cuda.get_device_capability(3);  get_device_capability = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:236 in get_compute_capability, code: return torch.cuda.get_device_capability(device.index)
                get_device_capability = torch.cuda.get_device_capability(3);  get_device_capability = None
                return ()
                
    class thunder_2(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py:1303 in trtllm_fp4_block_scale_moe_op, code: output = torch.empty(
            output: "bf16[1, 3072]" = torch.empty(1, 3072, dtype = torch.bfloat16, device = device(type='cuda', index=3))
            return output
            
        class _model(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py:1303 in trtllm_fp4_block_scale_moe_op, code: output = torch.empty(
                output: "bf16[1, 3072]" = torch.empty(1, 3072, dtype = torch.bfloat16, device = device(type='cuda', index=3))
                return output
                