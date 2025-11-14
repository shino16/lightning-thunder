# Rank: 1, Graph 2

class GraphModule(torch.nn.Module):
    def forward(self, l_input_: "bf16[1, 5120]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_input_);  l_input_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_input_: "bf16[1, 5120]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_utils.py:566 in apply_fp8_linear, code: input_2d = input.view(-1, input.shape[-1])
            input_2d: "bf16[1, 5120]" = l_input_.view(-1, 5120);  l_input_ = None
            return input_2d
            
        class _model(torch.nn.Module):
            def forward(self, l_input_: "bf16[1, 5120]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/fp8_utils.py:566 in apply_fp8_linear, code: input_2d = input.view(-1, input.shape[-1])
                input_2d: "bf16[1, 5120]" = l_input_.view(-1, 5120);  l_input_ = None
                return input_2d
                