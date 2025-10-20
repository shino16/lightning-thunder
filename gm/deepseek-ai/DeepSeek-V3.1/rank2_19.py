# Rank: 2, Graph 19

class GraphModule(torch.nn.Module):
    def forward(self, l_a_: "f8e4m3fn[1, 4096]", l_b_: "f8e4m3fn[7168, 4096]", l_c_: "bf16[1, 7168]", l_as_: "f32[1, 32]", l_bs_: "f32[56, 32]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_a_, l_b_, l_c_, l_as_, l_bs_);  l_a_ = l_b_ = l_as_ = l_bs_ = inductor_0 = None
        return (l_c_,)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_a_: "f8e4m3fn[1, 4096]", l_b_: "f8e4m3fn[7168, 4096]", l_c_: "bf16[1, 7168]", l_as_: "f32[1, 32]", l_bs_: "f32[56, 32]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:1094 in torch_dynamo_resume_in_w8a8_block_fp8_matmul_triton_at_1070, code: kernel[grid](
            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 21, constant_args_idx = 31, grid = [(56, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'A': l_a_, 'B': l_b_, 'C': l_c_, 'As': l_as_, 'Bs': l_bs_});  l_a_ = l_b_ = l_c_ = l_as_ = l_bs_ = triton_kernel_wrapper_mutation = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_a_: "f8e4m3fn[1, 4096]", l_b_: "f8e4m3fn[7168, 4096]", l_c_: "bf16[1, 7168]", l_as_: "f32[1, 32]", l_bs_: "f32[56, 32]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:1094 in torch_dynamo_resume_in_w8a8_block_fp8_matmul_triton_at_1070, code: kernel[grid](
                triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 21, constant_args_idx = 31, grid = [(56, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'A': l_a_, 'B': l_b_, 'C': l_c_, 'As': l_as_, 'Bs': l_bs_});  l_a_ = l_b_ = l_c_ = l_as_ = l_bs_ = triton_kernel_wrapper_mutation = None
                return ()
                