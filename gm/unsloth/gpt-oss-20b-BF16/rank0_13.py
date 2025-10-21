# Rank: 0, Graph 13

class GraphModule(torch.nn.Module):
    def forward(self, l_self_modules_o_proj_parameters_bias_: "bf16[2880]", l_self_modules_o_proj_parameters_weight_: "bf16[2880, 1024]", l_stack0_: "bf16[1, 1024]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_, l_self_modules_o_proj_parameters_weight_, l_self_modules_o_proj_parameters_bias_);  l_stack0_ = l_self_modules_o_proj_parameters_weight_ = l_self_modules_o_proj_parameters_bias_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 1024]", l_self_modules_o_proj_parameters_weight_: "bf16[2880, 1024]", l_self_modules_o_proj_parameters_bias_: "bf16[2880]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel: "bf16[1, 2880]" = torch._C._nn.linear(l_stack0_, l_self_modules_o_proj_parameters_weight_, l_self_modules_o_proj_parameters_bias_);  l_stack0_ = l_self_modules_o_proj_parameters_weight_ = l_self_modules_o_proj_parameters_bias_ = None
            return output_parallel
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 1024]", l_self_modules_o_proj_parameters_weight_: "bf16[2880, 1024]", l_self_modules_o_proj_parameters_bias_: "bf16[2880]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel: "bf16[1, 2880]" = torch._C._nn.linear(l_stack0_, l_self_modules_o_proj_parameters_weight_, l_self_modules_o_proj_parameters_bias_);  l_stack0_ = l_self_modules_o_proj_parameters_weight_ = l_self_modules_o_proj_parameters_bias_ = None
                return output_parallel
                