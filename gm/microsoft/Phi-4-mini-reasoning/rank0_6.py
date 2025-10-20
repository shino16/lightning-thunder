# Rank: 0, Graph 6

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 3072]", l_residual_: "bf16[1, 3072]", l_self_modules_post_attention_layernorm_parameters_weight_: "bf16[3072]", l_self_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[4096, 3072]", l_self_modules_mlp_modules_down_proj_parameters_weight_: "bf16[3072, 2048]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_, l_residual_, l_self_modules_post_attention_layernorm_parameters_weight_, l_self_modules_mlp_modules_gate_up_proj_parameters_weight_, l_self_modules_mlp_modules_down_proj_parameters_weight_);  l_stack0_ = l_residual_ = l_self_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_mlp_modules_gate_up_proj_parameters_weight_ = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        inductor_1 = self.inductor_1(getitem);  inductor_1 = None
        return (getitem, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 3072]", l_residual_: "bf16[1, 3072]", l_self_modules_post_attention_layernorm_parameters_weight_: "bf16[3072]", l_self_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[4096, 3072]", l_self_modules_mlp_modules_down_proj_parameters_weight_: "bf16[3072, 2048]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
            x: "f32[1, 3072]" = l_stack0_.to(torch.float32);  l_stack0_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:169 in forward_native, code: x = x + residual.to(torch.float32)
            to_1: "f32[1, 3072]" = l_residual_.to(torch.float32);  l_residual_ = None
            x_1: "f32[1, 3072]" = x + to_1;  x = to_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:170 in forward_native, code: residual = x.to(orig_dtype)
            residual: "bf16[1, 3072]" = x_1.to(torch.bfloat16)
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
            pow_1: "f32[1, 3072]" = x_1.pow(2)
            variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
            add_1: "f32[1, 1]" = variance + 1e-05;  variance = None
            rsqrt: "f32[1, 1]" = torch.rsqrt(add_1);  add_1 = None
            x_2: "f32[1, 3072]" = x_1 * rsqrt;  x_1 = rsqrt = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
            mul_1: "f32[1, 3072]" = x_2 * l_self_modules_post_attention_layernorm_parameters_weight_;  x_2 = l_self_modules_post_attention_layernorm_parameters_weight_ = None
            x_3: "bf16[1, 3072]" = mul_1.to(torch.bfloat16);  mul_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel: "bf16[1, 4096]" = torch._C._nn.linear(x_3, l_self_modules_mlp_modules_gate_up_proj_parameters_weight_, None);  x_3 = l_self_modules_mlp_modules_gate_up_proj_parameters_weight_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/activation.py:64 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
            getitem: "bf16[1, 2048]" = output_parallel[(Ellipsis, slice(None, 2048, None))]
            silu: "bf16[1, 2048]" = torch.nn.functional.silu(getitem);  getitem = None
            getitem_1: "bf16[1, 2048]" = output_parallel[(Ellipsis, slice(2048, None, None))];  output_parallel = None
            x_4: "bf16[1, 2048]" = silu * getitem_1;  silu = getitem_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel_1: "bf16[1, 3072]" = torch._C._nn.linear(x_4, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  x_4 = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
            return (output_parallel_1, residual)
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 3072]", l_residual_: "bf16[1, 3072]", l_self_modules_post_attention_layernorm_parameters_weight_: "bf16[3072]", l_self_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[4096, 3072]", l_self_modules_mlp_modules_down_proj_parameters_weight_: "bf16[3072, 2048]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
                x: "f32[1, 3072]" = l_stack0_.to(torch.float32);  l_stack0_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:169 in forward_native, code: x = x + residual.to(torch.float32)
                to_1: "f32[1, 3072]" = l_residual_.to(torch.float32);  l_residual_ = None
                x_1: "f32[1, 3072]" = x + to_1;  x = to_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:170 in forward_native, code: residual = x.to(orig_dtype)
                residual: "bf16[1, 3072]" = x_1.to(torch.bfloat16)
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
                pow_1: "f32[1, 3072]" = x_1.pow(2)
                variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
                add_1: "f32[1, 1]" = variance + 1e-05;  variance = None
                rsqrt: "f32[1, 1]" = torch.rsqrt(add_1);  add_1 = None
                x_2: "f32[1, 3072]" = x_1 * rsqrt;  x_1 = rsqrt = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
                mul_1: "f32[1, 3072]" = x_2 * l_self_modules_post_attention_layernorm_parameters_weight_;  x_2 = l_self_modules_post_attention_layernorm_parameters_weight_ = None
                x_3: "bf16[1, 3072]" = mul_1.to(torch.bfloat16);  mul_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel: "bf16[1, 4096]" = torch._C._nn.linear(x_3, l_self_modules_mlp_modules_gate_up_proj_parameters_weight_, None);  x_3 = l_self_modules_mlp_modules_gate_up_proj_parameters_weight_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/activation.py:64 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
                getitem: "bf16[1, 2048]" = output_parallel[(Ellipsis, slice(None, 2048, None))]
                silu: "bf16[1, 2048]" = torch.nn.functional.silu(getitem);  getitem = None
                getitem_1: "bf16[1, 2048]" = output_parallel[(Ellipsis, slice(2048, None, None))];  output_parallel = None
                x_4: "bf16[1, 2048]" = silu * getitem_1;  silu = getitem_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel_1: "bf16[1, 3072]" = torch._C._nn.linear(x_4, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  x_4 = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
                return (output_parallel_1, residual)
                
    class inductor_1(torch.nn.Module):
        def forward(self, output_parallel_1: "bf16[1, 3072]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_1 = output_parallel_1.is_cpu;  getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(output_parallel_1, group_name = 'tp:0');  output_parallel_1 = inplace_all_reduce = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, output_parallel_1: "bf16[1, 3072]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_1 = output_parallel_1.is_cpu;  getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(output_parallel_1, group_name = 'tp:0');  output_parallel_1 = inplace_all_reduce = None
                return ()
                