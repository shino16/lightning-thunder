# Rank: 0, Graph 9

class GraphModule(torch.nn.Module):
    def forward(self, l_args_0_: "f16[1, 5120]", l_args_1_: "f16[1, 5120]", l_self_parameters_weight_: "f16[5120]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_args_0_, l_args_1_, l_self_parameters_weight_);  l_args_0_ = l_args_1_ = l_self_parameters_weight_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_args_0_: "f16[1, 5120]", l_args_1_: "f16[1, 5120]", l_self_parameters_weight_: "f16[5120]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
            x: "f32[1, 5120]" = l_args_0_.to(torch.float32);  l_args_0_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:169 in forward_native, code: x = x + residual.to(torch.float32)
            to_1: "f32[1, 5120]" = l_args_1_.to(torch.float32);  l_args_1_ = None
            x_1: "f32[1, 5120]" = x + to_1;  x = to_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:170 in forward_native, code: residual = x.to(orig_dtype)
            residual: "f16[1, 5120]" = x_1.to(torch.float16)
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
            pow_1: "f32[1, 5120]" = x_1.pow(2)
            variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
            add_1: "f32[1, 1]" = variance + 1e-05;  variance = None
            rsqrt: "f32[1, 1]" = torch.rsqrt(add_1);  add_1 = None
            x_2: "f32[1, 5120]" = x_1 * rsqrt;  x_1 = rsqrt = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
            mul_1: "f32[1, 5120]" = x_2 * l_self_parameters_weight_;  x_2 = l_self_parameters_weight_ = None
            x_3: "f16[1, 5120]" = mul_1.to(torch.float16);  mul_1 = None
            return (x_3, residual)
            
        class _model(torch.nn.Module):
            def forward(self, l_args_0_: "f16[1, 5120]", l_args_1_: "f16[1, 5120]", l_self_parameters_weight_: "f16[5120]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
                x: "f32[1, 5120]" = l_args_0_.to(torch.float32);  l_args_0_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:169 in forward_native, code: x = x + residual.to(torch.float32)
                to_1: "f32[1, 5120]" = l_args_1_.to(torch.float32);  l_args_1_ = None
                x_1: "f32[1, 5120]" = x + to_1;  x = to_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:170 in forward_native, code: residual = x.to(orig_dtype)
                residual: "f16[1, 5120]" = x_1.to(torch.float16)
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
                pow_1: "f32[1, 5120]" = x_1.pow(2)
                variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
                add_1: "f32[1, 1]" = variance + 1e-05;  variance = None
                rsqrt: "f32[1, 1]" = torch.rsqrt(add_1);  add_1 = None
                x_2: "f32[1, 5120]" = x_1 * rsqrt;  x_1 = rsqrt = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
                mul_1: "f32[1, 5120]" = x_2 * l_self_parameters_weight_;  x_2 = l_self_parameters_weight_ = None
                x_3: "f16[1, 5120]" = mul_1.to(torch.float16);  mul_1 = None
                return (x_3, residual)
                