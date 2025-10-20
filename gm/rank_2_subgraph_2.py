class GraphModule(torch.nn.Module):
    def forward(self, l_self_parameters_weight_: "bf16[1280, 2048]", l_input_: "bf16[1, 2048]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_input_, l_self_parameters_weight_);  l_input_ = l_self_parameters_weight_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_input_: "bf16[1, 2048]", l_self_parameters_weight_: "bf16[1280, 2048]"):
             # File: /usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel: "bf16[1, 1280]" = torch._C._nn.linear(l_input_, l_self_parameters_weight_, None);  l_input_ = l_self_parameters_weight_ = None
            return output_parallel
            
        class _model(torch.nn.Module):
            def forward(self, l_input_: "bf16[1, 2048]", l_self_parameters_weight_: "bf16[1280, 2048]"):
                 # File: /usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel: "bf16[1, 1280]" = torch._C._nn.linear(l_input_, l_self_parameters_weight_, None);  l_input_ = l_self_parameters_weight_ = None
                return output_parallel
                
