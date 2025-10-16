# Rank: 1, Graph 8

class GraphModule(torch.nn.Module):
    def forward(self, l_self_parameters_weight_: "bf16[1024, 512]", l_input_: "bf16[1, 512]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_input_, l_self_parameters_weight_);  l_input_ = l_self_parameters_weight_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_input_: "bf16[1, 512]", l_self_parameters_weight_: "bf16[1024, 512]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel: "bf16[1, 1024]" = torch._C._nn.linear(l_input_, l_self_parameters_weight_, None);  l_input_ = l_self_parameters_weight_ = None
            return output_parallel
            
        class _model(torch.nn.Module):
            def forward(self, l_input_: "bf16[1, 512]", l_self_parameters_weight_: "bf16[1024, 512]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel: "bf16[1, 1024]" = torch._C._nn.linear(l_input_, l_self_parameters_weight_, None);  l_input_ = l_self_parameters_weight_ = None
                return output_parallel
                