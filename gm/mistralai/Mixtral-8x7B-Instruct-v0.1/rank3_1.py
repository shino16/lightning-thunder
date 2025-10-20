# Rank: 3, Graph 1

class GraphModule(torch.nn.Module):
    def forward(self, l_hidden_states_: "bf16[1, 4096]", l_self_modules_input_layernorm_parameters_weight_: "bf16[4096]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_hidden_states_, l_self_modules_input_layernorm_parameters_weight_);  l_hidden_states_ = l_self_modules_input_layernorm_parameters_weight_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_hidden_states_: "bf16[1, 4096]", l_self_modules_input_layernorm_parameters_weight_: "bf16[4096]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
            x: "f32[1, 4096]" = l_hidden_states_.to(torch.float32);  l_hidden_states_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
            pow_1: "f32[1, 4096]" = x.pow(2)
            variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
            add: "f32[1, 1]" = variance + 1e-05;  variance = None
            rsqrt: "f32[1, 1]" = torch.rsqrt(add);  add = None
            x_1: "f32[1, 4096]" = x * rsqrt;  x = rsqrt = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
            mul_1: "f32[1, 4096]" = x_1 * l_self_modules_input_layernorm_parameters_weight_;  x_1 = l_self_modules_input_layernorm_parameters_weight_ = None
            x_2: "bf16[1, 4096]" = mul_1.to(torch.bfloat16);  mul_1 = None
            return x_2
            
        class _model(torch.nn.Module):
            def forward(self, l_hidden_states_: "bf16[1, 4096]", l_self_modules_input_layernorm_parameters_weight_: "bf16[4096]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
                x: "f32[1, 4096]" = l_hidden_states_.to(torch.float32);  l_hidden_states_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
                pow_1: "f32[1, 4096]" = x.pow(2)
                variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
                add: "f32[1, 1]" = variance + 1e-05;  variance = None
                rsqrt: "f32[1, 1]" = torch.rsqrt(add);  add = None
                x_1: "f32[1, 4096]" = x * rsqrt;  x = rsqrt = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
                mul_1: "f32[1, 4096]" = x_1 * l_self_modules_input_layernorm_parameters_weight_;  x_1 = l_self_modules_input_layernorm_parameters_weight_ = None
                x_2: "bf16[1, 4096]" = mul_1.to(torch.bfloat16);  mul_1 = None
                return x_2
                