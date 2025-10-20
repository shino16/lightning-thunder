# Rank: 2, Graph 8

class GraphModule(torch.nn.Module):
    def forward(self, l_input_: "bf16[1, 1536]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_input_);  l_input_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1]
        getitem_2 = thunder_0[2];  thunder_0 = None
        inductor_1 = self.inductor_1(getitem, getitem_1, getitem_2);  inductor_1 = None
        return (getitem_1, getitem_2, getitem)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_input_: "bf16[1, 1536]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_utils.py:296 in triton_w8a8_block_fp8_linear, code: input_2d = input.view(-1, input.shape[-1])
            input_2d: "bf16[1, 1536]" = l_input_.view(-1, 1536);  l_input_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:251 in _per_token_group_quant_8bit_raw, code: x_q = torch.empty_like(x, device=x.device, dtype=dtype)
            x_q: "f8e4m3fn[1, 1536]" = torch.empty_like(input_2d, device = device(type='cuda', index=2), dtype = torch.float8_e4m3fn)
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:464 in create_per_token_group_quant_fp8_output_scale, code: return torch.empty(
            x_s: "f32[1, 12]" = torch.empty((1, 12), device = device(type='cuda', index=2), dtype = torch.float32)
            return (input_2d, x_q, x_s)
            
        class _model(torch.nn.Module):
            def forward(self, l_input_: "bf16[1, 1536]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_utils.py:296 in triton_w8a8_block_fp8_linear, code: input_2d = input.view(-1, input.shape[-1])
                input_2d: "bf16[1, 1536]" = l_input_.view(-1, 1536);  l_input_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:251 in _per_token_group_quant_8bit_raw, code: x_q = torch.empty_like(x, device=x.device, dtype=dtype)
                x_q: "f8e4m3fn[1, 1536]" = torch.empty_like(input_2d, device = device(type='cuda', index=2), dtype = torch.float8_e4m3fn)
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:464 in create_per_token_group_quant_fp8_output_scale, code: return torch.empty(
                x_s: "f32[1, 12]" = torch.empty((1, 12), device = device(type='cuda', index=2), dtype = torch.float32)
                return (input_2d, x_q, x_s)
                
    class inductor_1(torch.nn.Module):
        def forward(self, input_2d: "bf16[1, 1536]", x_q: "f8e4m3fn[1, 1536]", x_s: "f32[1, 12]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:286 in _per_token_group_quant_8bit_raw, code: _per_token_group_quant_8bit[(M,)](
            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 13, constant_args_idx = 13, grid = [(12, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'y_ptr': input_2d, 'y_q_ptr': x_q, 'y_s_ptr': x_s});  input_2d = x_q = x_s = triton_kernel_wrapper_mutation = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, input_2d: "bf16[1, 1536]", x_q: "f8e4m3fn[1, 1536]", x_s: "f32[1, 12]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:286 in _per_token_group_quant_8bit_raw, code: _per_token_group_quant_8bit[(M,)](
                triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 13, constant_args_idx = 13, grid = [(12, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'y_ptr': input_2d, 'y_q_ptr': x_q, 'y_s_ptr': x_s});  input_2d = x_q = x_s = triton_kernel_wrapper_mutation = None
                return ()
                