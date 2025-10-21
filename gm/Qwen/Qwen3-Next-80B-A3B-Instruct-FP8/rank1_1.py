# Rank: 1, Graph 1

class GraphModule(torch.nn.Module):
    def forward(self, l_hidden_states_: "bf16[1, 2048]", l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2048]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_hidden_states_, l_self_layer_communicator_input_layernorm_parameters_weight_);  l_hidden_states_ = l_self_layer_communicator_input_layernorm_parameters_weight_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_hidden_states_: "bf16[1, 2048]", l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2048]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:272 in forward_native, code: x = x.float()
            x: "f32[1, 2048]" = l_hidden_states_.float();  l_hidden_states_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:273 in forward_native, code: variance = x.pow(2).mean(dim=-1, keepdim=True)
            pow_1: "f32[1, 2048]" = x.pow(2)
            variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:274 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
            add: "f32[1, 1]" = variance + 1e-06;  variance = None
            rsqrt: "f32[1, 1]" = torch.rsqrt(add);  add = None
            x_1: "f32[1, 2048]" = x * rsqrt;  x = rsqrt = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:275 in forward_native, code: x = x * (1.0 + self.weight.float())
            float_2: "f32[2048]" = l_self_layer_communicator_input_layernorm_parameters_weight_.float();  l_self_layer_communicator_input_layernorm_parameters_weight_ = None
            add_1: "f32[2048]" = 1.0 + float_2;  float_2 = None
            x_2: "f32[1, 2048]" = x_1 * add_1;  x_1 = add_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:276 in forward_native, code: x = x.to(orig_dtype)
            x_3: "bf16[1, 2048]" = x_2.to(torch.bfloat16);  x_2 = None
            return x_3
            
        class _model(torch.nn.Module):
            def forward(self, l_hidden_states_: "bf16[1, 2048]", l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2048]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:272 in forward_native, code: x = x.float()
                x: "f32[1, 2048]" = l_hidden_states_.float();  l_hidden_states_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:273 in forward_native, code: variance = x.pow(2).mean(dim=-1, keepdim=True)
                pow_1: "f32[1, 2048]" = x.pow(2)
                variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:274 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
                add: "f32[1, 1]" = variance + 1e-06;  variance = None
                rsqrt: "f32[1, 1]" = torch.rsqrt(add);  add = None
                x_1: "f32[1, 2048]" = x * rsqrt;  x = rsqrt = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:275 in forward_native, code: x = x * (1.0 + self.weight.float())
                float_2: "f32[2048]" = l_self_layer_communicator_input_layernorm_parameters_weight_.float();  l_self_layer_communicator_input_layernorm_parameters_weight_ = None
                add_1: "f32[2048]" = 1.0 + float_2;  float_2 = None
                x_2: "f32[1, 2048]" = x_1 * add_1;  x_1 = add_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:276 in forward_native, code: x = x.to(orig_dtype)
                x_3: "bf16[1, 2048]" = x_2.to(torch.bfloat16);  x_2 = None
                return x_3
                