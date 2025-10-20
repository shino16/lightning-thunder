# Rank: 1, Graph 5

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 2112]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_);  l_stack0_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 2112]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_utils.py:307 in torch_dynamo_resume_in_triton_w8a8_block_fp8_linear_at_302, code: return output.to(dtype=input_2d.dtype).view(*output_shape)
            to: "bf16[1, 2112]" = l_stack0_.to(dtype = torch.bfloat16);  l_stack0_ = None
            view: "bf16[1, 2112]" = to.view(1, 2112);  to = None
            return view
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 2112]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_utils.py:307 in torch_dynamo_resume_in_triton_w8a8_block_fp8_linear_at_302, code: return output.to(dtype=input_2d.dtype).view(*output_shape)
                to: "bf16[1, 2112]" = l_stack0_.to(dtype = torch.bfloat16);  l_stack0_ = None
                view: "bf16[1, 2112]" = to.view(1, 2112);  to = None
                return view
                