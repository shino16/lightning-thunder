# Rank: 1, Graph 28

class GraphModule(torch.nn.Module):
    def forward(self, l_a_: "f8e4m3fn[1, 4608]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_a_);  l_a_ = None
        return (inductor_0,)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_a_: "f8e4m3fn[1, 4608]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:1015 in prepare_block_fp8_matmul_inputs, code: C = A.new_empty(C_shape, dtype=output_dtype)
            C: "bf16[1, 7168]" = l_a_.new_empty((1, 7168), dtype = torch.bfloat16);  l_a_ = None
            return C
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_a_: "f8e4m3fn[1, 4608]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_kernel.py:1015 in prepare_block_fp8_matmul_inputs, code: C = A.new_empty(C_shape, dtype=output_dtype)
                C: "bf16[1, 7168]" = l_a_.new_empty((1, 7168), dtype = torch.bfloat16);  l_a_ = None
                return C
                