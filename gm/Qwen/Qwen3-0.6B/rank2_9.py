# Rank: 2, Graph 9

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 1024]", l_residual_: "bf16[1, 1024]", l_self_layer_communicator_post_attention_layernorm_parameters_weight_: "bf16[1024]", l_self_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[1536, 1024]", l_self_modules_mlp_modules_down_proj_parameters_weight_: "bf16[1024, 768]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_stack0_);  inductor_0 = None
        thunder_1 = self.thunder_1(l_stack0_, l_residual_, l_self_layer_communicator_post_attention_layernorm_parameters_weight_, l_self_modules_mlp_modules_gate_up_proj_parameters_weight_, l_self_modules_mlp_modules_down_proj_parameters_weight_);  l_stack0_ = l_residual_ = l_self_layer_communicator_post_attention_layernorm_parameters_weight_ = l_self_modules_mlp_modules_gate_up_proj_parameters_weight_ = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
        getitem = thunder_1[0]
        getitem_1 = thunder_1[1];  thunder_1 = None
        inductor_2 = self.inductor_2(getitem);  inductor_2 = None
        return (getitem, getitem_1)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 1024]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_1 = l_stack0_.is_cpu;  getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(l_stack0_, group_name = 'tp:0');  l_stack0_ = inplace_all_reduce = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 1024]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_1 = l_stack0_.is_cpu;  getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(l_stack0_, group_name = 'tp:0');  l_stack0_ = inplace_all_reduce = None
                return ()
                
    class thunder_1(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 1024]", l_residual_: "bf16[1, 1024]", l_self_layer_communicator_post_attention_layernorm_parameters_weight_: "bf16[1024]", l_self_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[1536, 1024]", l_self_modules_mlp_modules_down_proj_parameters_weight_: "bf16[1024, 768]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
            x: "f32[1, 1024]" = l_stack0_.to(torch.float32);  l_stack0_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:169 in forward_native, code: x = x + residual.to(torch.float32)
            to_1: "f32[1, 1024]" = l_residual_.to(torch.float32);  l_residual_ = None
            x_1: "f32[1, 1024]" = x + to_1;  x = to_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:170 in forward_native, code: residual = x.to(orig_dtype)
            residual: "bf16[1, 1024]" = x_1.to(torch.bfloat16)
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
            pow_1: "f32[1, 1024]" = x_1.pow(2)
            variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
            add_1: "f32[1, 1]" = variance + 1e-06;  variance = None
            rsqrt: "f32[1, 1]" = torch.rsqrt(add_1);  add_1 = None
            x_2: "f32[1, 1024]" = x_1 * rsqrt;  x_1 = rsqrt = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
            mul_1: "f32[1, 1024]" = x_2 * l_self_layer_communicator_post_attention_layernorm_parameters_weight_;  x_2 = l_self_layer_communicator_post_attention_layernorm_parameters_weight_ = None
            x_3: "bf16[1, 1024]" = mul_1.to(torch.bfloat16);  mul_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel: "bf16[1, 1536]" = torch._C._nn.linear(x_3, l_self_modules_mlp_modules_gate_up_proj_parameters_weight_, None);  x_3 = l_self_modules_mlp_modules_gate_up_proj_parameters_weight_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/activation.py:64 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
            getitem: "bf16[1, 768]" = output_parallel[(Ellipsis, slice(None, 768, None))]
            silu: "bf16[1, 768]" = torch.nn.functional.silu(getitem);  getitem = None
            getitem_1: "bf16[1, 768]" = output_parallel[(Ellipsis, slice(768, None, None))];  output_parallel = None
            x_4: "bf16[1, 768]" = silu * getitem_1;  silu = getitem_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel_1: "bf16[1, 1024]" = torch._C._nn.linear(x_4, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  x_4 = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
            return (output_parallel_1, residual)
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 1024]", l_residual_: "bf16[1, 1024]", l_self_layer_communicator_post_attention_layernorm_parameters_weight_: "bf16[1024]", l_self_modules_mlp_modules_gate_up_proj_parameters_weight_: "bf16[1536, 1024]", l_self_modules_mlp_modules_down_proj_parameters_weight_: "bf16[1024, 768]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
                x: "f32[1, 1024]" = l_stack0_.to(torch.float32);  l_stack0_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:169 in forward_native, code: x = x + residual.to(torch.float32)
                to_1: "f32[1, 1024]" = l_residual_.to(torch.float32);  l_residual_ = None
                x_1: "f32[1, 1024]" = x + to_1;  x = to_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:170 in forward_native, code: residual = x.to(orig_dtype)
                residual: "bf16[1, 1024]" = x_1.to(torch.bfloat16)
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
                pow_1: "f32[1, 1024]" = x_1.pow(2)
                variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
                add_1: "f32[1, 1]" = variance + 1e-06;  variance = None
                rsqrt: "f32[1, 1]" = torch.rsqrt(add_1);  add_1 = None
                x_2: "f32[1, 1024]" = x_1 * rsqrt;  x_1 = rsqrt = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
                mul_1: "f32[1, 1024]" = x_2 * l_self_layer_communicator_post_attention_layernorm_parameters_weight_;  x_2 = l_self_layer_communicator_post_attention_layernorm_parameters_weight_ = None
                x_3: "bf16[1, 1024]" = mul_1.to(torch.bfloat16);  mul_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel: "bf16[1, 1536]" = torch._C._nn.linear(x_3, l_self_modules_mlp_modules_gate_up_proj_parameters_weight_, None);  x_3 = l_self_modules_mlp_modules_gate_up_proj_parameters_weight_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/activation.py:64 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
                getitem: "bf16[1, 768]" = output_parallel[(Ellipsis, slice(None, 768, None))]
                silu: "bf16[1, 768]" = torch.nn.functional.silu(getitem);  getitem = None
                getitem_1: "bf16[1, 768]" = output_parallel[(Ellipsis, slice(768, None, None))];  output_parallel = None
                x_4: "bf16[1, 768]" = silu * getitem_1;  silu = getitem_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel_1: "bf16[1, 1024]" = torch._C._nn.linear(x_4, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  x_4 = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
                return (output_parallel_1, residual)
                
    class inductor_2(torch.nn.Module):
        def forward(self, output_parallel_1: "bf16[1, 1024]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_2 = output_parallel_1.is_cpu;  getattr_2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce_1 = torch.ops.sglang.inplace_all_reduce(output_parallel_1, group_name = 'tp:0');  output_parallel_1 = inplace_all_reduce_1 = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, output_parallel_1: "bf16[1, 1024]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_2 = output_parallel_1.is_cpu;  getattr_2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce_1 = torch.ops.sglang.inplace_all_reduce(output_parallel_1, group_name = 'tp:0');  output_parallel_1 = inplace_all_reduce_1 = None
                return ()
                