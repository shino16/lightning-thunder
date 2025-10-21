# Rank: 0, Graph 15

class GraphModule(torch.nn.Module):
    def forward(self, l_hidden_states_: "bf16[1, 2880]", l_self_modules_router_parameters_bias_: "bf16[32]", l_self_modules_router_parameters_weight_: "bf16[32, 2880]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_hidden_states_, l_self_modules_router_parameters_weight_, l_self_modules_router_parameters_bias_);  l_hidden_states_ = l_self_modules_router_parameters_weight_ = l_self_modules_router_parameters_bias_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_hidden_states_: "bf16[1, 2880]", l_self_modules_router_parameters_weight_: "bf16[32, 2880]", l_self_modules_router_parameters_bias_: "bf16[32]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output: "bf16[1, 32]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_router_parameters_weight_, l_self_modules_router_parameters_bias_);  l_hidden_states_ = l_self_modules_router_parameters_weight_ = l_self_modules_router_parameters_bias_ = None
            return output
            
        class _model(torch.nn.Module):
            def forward(self, l_hidden_states_: "bf16[1, 2880]", l_self_modules_router_parameters_weight_: "bf16[32, 2880]", l_self_modules_router_parameters_bias_: "bf16[32]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output: "bf16[1, 32]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_router_parameters_weight_, l_self_modules_router_parameters_bias_);  l_hidden_states_ = l_self_modules_router_parameters_weight_ = l_self_modules_router_parameters_bias_ = None
                return output
                