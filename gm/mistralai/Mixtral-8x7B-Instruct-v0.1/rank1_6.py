# Rank: 1, Graph 6

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 4096]", l_residual_: "bf16[1, 4096]", l_self_modules_post_attention_layernorm_parameters_weight_: "bf16[4096]", l_self_modules_block_sparse_moe_modules_gate_parameters_weight_: "bf16[8, 4096]", l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_: "bf16[8, 7168, 4096]", l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_: "bf16[8, 4096, 3584]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_, l_residual_, l_self_modules_post_attention_layernorm_parameters_weight_, l_self_modules_block_sparse_moe_modules_gate_parameters_weight_, l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_, l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_);  l_stack0_ = l_residual_ = l_self_modules_post_attention_layernorm_parameters_weight_ = l_self_modules_block_sparse_moe_modules_gate_parameters_weight_ = l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_ = l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        inductor_1 = self.inductor_1(getitem);  inductor_1 = None
        thunder_2 = self.thunder_2(getitem);  getitem = None
        return (thunder_2, getitem_1)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 4096]", l_residual_: "bf16[1, 4096]", l_self_modules_post_attention_layernorm_parameters_weight_: "bf16[4096]", l_self_modules_block_sparse_moe_modules_gate_parameters_weight_: "bf16[8, 4096]", l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_: "bf16[8, 7168, 4096]", l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_: "bf16[8, 4096, 3584]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
            x: "f32[1, 4096]" = l_stack0_.to(torch.float32);  l_stack0_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:169 in forward_native, code: x = x + residual.to(torch.float32)
            to_1: "f32[1, 4096]" = l_residual_.to(torch.float32);  l_residual_ = None
            x_1: "f32[1, 4096]" = x + to_1;  x = to_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:170 in forward_native, code: residual = x.to(orig_dtype)
            residual: "bf16[1, 4096]" = x_1.to(torch.bfloat16)
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
            pow_1: "f32[1, 4096]" = x_1.pow(2)
            variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
            add_1: "f32[1, 1]" = variance + 1e-05;  variance = None
            rsqrt: "f32[1, 1]" = torch.rsqrt(add_1);  add_1 = None
            x_2: "f32[1, 4096]" = x_1 * rsqrt;  x_1 = rsqrt = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
            mul_1: "f32[1, 4096]" = x_2 * l_self_modules_post_attention_layernorm_parameters_weight_;  x_2 = l_self_modules_post_attention_layernorm_parameters_weight_ = None
            x_3: "bf16[1, 4096]" = mul_1.to(torch.bfloat16);  mul_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/mixtral.py:112 in forward, code: hidden_states = hidden_states.view(-1, self.hidden_size)
            hidden_states: "bf16[1, 4096]" = x_3.view(-1, 4096);  x_3 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output: "bf16[1, 8]" = torch._C._nn.linear(hidden_states, l_self_modules_block_sparse_moe_modules_gate_parameters_weight_, None);  l_self_modules_block_sparse_moe_modules_gate_parameters_weight_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:396 in fused_topk_torch_native, code: topk_weights = torch.empty(
            topk_weights: "f32[1, 2]" = torch.empty(1, 2, dtype = torch.float32, device = device(type='cuda', index=1));  topk_weights = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:399 in fused_topk_torch_native, code: topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
            topk_ids: "i32[1, 2]" = torch.empty(1, 2, dtype = torch.int32, device = device(type='cuda', index=1));  topk_ids = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:400 in fused_topk_torch_native, code: topk_weights = F.softmax(gating_output.float(), dim=-1)
            float_1: "f32[1, 8]" = output.float();  output = None
            topk_weights_1: "f32[1, 8]" = torch.nn.functional.softmax(float_1, dim = -1);  float_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:401 in fused_topk_torch_native, code: topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
            topk = torch.topk(topk_weights_1, 2, dim = -1);  topk_weights_1 = None
            topk_weights_2: "f32[1, 2]" = topk[0]
            topk_ids_1: "i64[1, 2]" = topk[1];  topk = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:404 in fused_topk_torch_native, code: topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            sum_1: "f32[1, 1]" = topk_weights_2.sum(dim = -1, keepdim = True)
            topk_weights_3: "f32[1, 2]" = topk_weights_2 / sum_1;  topk_weights_2 = sum_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:28 in fused_moe_forward_native, code: w13_weights = layer.w13_weight[topk_ids]
            w13_weights: "bf16[1, 2, 7168, 4096]" = l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_[topk_ids_1];  l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:29 in fused_moe_forward_native, code: w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
            chunk = torch.chunk(w13_weights, 2, dim = 2);  w13_weights = None
            w1_weights: "bf16[1, 2, 3584, 4096]" = chunk[0]
            w3_weights: "bf16[1, 2, 3584, 4096]" = chunk[1];  chunk = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:30 in fused_moe_forward_native, code: w2_weights = layer.w2_weight[topk_ids]
            w2_weights: "bf16[1, 2, 4096, 3584]" = l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_[topk_ids_1];  l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_ = topk_ids_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:31 in fused_moe_forward_native, code: x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
            x1: "bf16[1, 2, 3584]" = torch.functional.einsum('ti,taoi -> tao', hidden_states, w1_weights);  w1_weights = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:33 in fused_moe_forward_native, code: x1 = F.silu(x1)
            x1_1: "bf16[1, 2, 3584]" = torch.nn.functional.silu(x1);  x1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:38 in fused_moe_forward_native, code: x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
            x3: "bf16[1, 2, 3584]" = torch.functional.einsum('ti, taoi -> tao', hidden_states, w3_weights);  hidden_states = w3_weights = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:39 in fused_moe_forward_native, code: expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
            mul_2: "bf16[1, 2, 3584]" = x1_1 * x3;  x1_1 = x3 = None
            expert_outs: "bf16[1, 2, 4096]" = torch.functional.einsum('tao, taio -> tai', mul_2, w2_weights);  mul_2 = w2_weights = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:40 in fused_moe_forward_native, code: return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))
            to_4: "bf16[1, 2]" = topk_weights_3.to(torch.bfloat16);  topk_weights_3 = None
            combine_input: "bf16[1, 4096]" = torch.functional.einsum('tai,ta -> ti', expert_outs, to_4);  expert_outs = to_4 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:840 in forward, code: final_hidden_states = final_hidden_states[
            getitem_6: "bf16[1, 4096]" = combine_input[(Ellipsis, slice(None, 4096, None))];  combine_input = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:842 in forward, code: ].contiguous()
            final_hidden_states: "bf16[1, 4096]" = getitem_6.contiguous();  getitem_6 = None
            return (final_hidden_states, residual)
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 4096]", l_residual_: "bf16[1, 4096]", l_self_modules_post_attention_layernorm_parameters_weight_: "bf16[4096]", l_self_modules_block_sparse_moe_modules_gate_parameters_weight_: "bf16[8, 4096]", l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_: "bf16[8, 7168, 4096]", l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_: "bf16[8, 4096, 3584]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:167 in forward_native, code: x = x.to(torch.float32)
                x: "f32[1, 4096]" = l_stack0_.to(torch.float32);  l_stack0_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:169 in forward_native, code: x = x + residual.to(torch.float32)
                to_1: "f32[1, 4096]" = l_residual_.to(torch.float32);  l_residual_ = None
                x_1: "f32[1, 4096]" = x + to_1;  x = to_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:170 in forward_native, code: residual = x.to(orig_dtype)
                residual: "bf16[1, 4096]" = x_1.to(torch.bfloat16)
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:190 in forward_native, code: variance = x_var.pow(2).mean(dim=-1, keepdim=True)
                pow_1: "f32[1, 4096]" = x_1.pow(2)
                variance: "f32[1, 1]" = pow_1.mean(dim = -1, keepdim = True);  pow_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:191 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
                add_1: "f32[1, 1]" = variance + 1e-05;  variance = None
                rsqrt: "f32[1, 1]" = torch.rsqrt(add_1);  add_1 = None
                x_2: "f32[1, 4096]" = x_1 * rsqrt;  x_1 = rsqrt = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:192 in forward_native, code: x = (x * self.weight).to(orig_dtype)
                mul_1: "f32[1, 4096]" = x_2 * l_self_modules_post_attention_layernorm_parameters_weight_;  x_2 = l_self_modules_post_attention_layernorm_parameters_weight_ = None
                x_3: "bf16[1, 4096]" = mul_1.to(torch.bfloat16);  mul_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/mixtral.py:112 in forward, code: hidden_states = hidden_states.view(-1, self.hidden_size)
                hidden_states: "bf16[1, 4096]" = x_3.view(-1, 4096);  x_3 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output: "bf16[1, 8]" = torch._C._nn.linear(hidden_states, l_self_modules_block_sparse_moe_modules_gate_parameters_weight_, None);  l_self_modules_block_sparse_moe_modules_gate_parameters_weight_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:396 in fused_topk_torch_native, code: topk_weights = torch.empty(
                topk_weights: "f32[1, 2]" = torch.empty(1, 2, dtype = torch.float32, device = device(type='cuda', index=1));  topk_weights = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:399 in fused_topk_torch_native, code: topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
                topk_ids: "i32[1, 2]" = torch.empty(1, 2, dtype = torch.int32, device = device(type='cuda', index=1));  topk_ids = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:400 in fused_topk_torch_native, code: topk_weights = F.softmax(gating_output.float(), dim=-1)
                float_1: "f32[1, 8]" = output.float();  output = None
                topk_weights_1: "f32[1, 8]" = torch.nn.functional.softmax(float_1, dim = -1);  float_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:401 in fused_topk_torch_native, code: topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
                topk = torch.topk(topk_weights_1, 2, dim = -1);  topk_weights_1 = None
                topk_weights_2: "f32[1, 2]" = topk[0]
                topk_ids_1: "i64[1, 2]" = topk[1];  topk = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:404 in fused_topk_torch_native, code: topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                sum_1: "f32[1, 1]" = topk_weights_2.sum(dim = -1, keepdim = True)
                topk_weights_3: "f32[1, 2]" = topk_weights_2 / sum_1;  topk_weights_2 = sum_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:28 in fused_moe_forward_native, code: w13_weights = layer.w13_weight[topk_ids]
                w13_weights: "bf16[1, 2, 7168, 4096]" = l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_[topk_ids_1];  l_self_modules_block_sparse_moe_modules_experts_parameters_w13_weight_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:29 in fused_moe_forward_native, code: w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
                chunk = torch.chunk(w13_weights, 2, dim = 2);  w13_weights = None
                w1_weights: "bf16[1, 2, 3584, 4096]" = chunk[0]
                w3_weights: "bf16[1, 2, 3584, 4096]" = chunk[1];  chunk = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:30 in fused_moe_forward_native, code: w2_weights = layer.w2_weight[topk_ids]
                w2_weights: "bf16[1, 2, 4096, 3584]" = l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_[topk_ids_1];  l_self_modules_block_sparse_moe_modules_experts_parameters_w2_weight_ = topk_ids_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:31 in fused_moe_forward_native, code: x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
                x1: "bf16[1, 2, 3584]" = torch.functional.einsum('ti,taoi -> tao', hidden_states, w1_weights);  w1_weights = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:33 in fused_moe_forward_native, code: x1 = F.silu(x1)
                x1_1: "bf16[1, 2, 3584]" = torch.nn.functional.silu(x1);  x1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:38 in fused_moe_forward_native, code: x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
                x3: "bf16[1, 2, 3584]" = torch.functional.einsum('ti, taoi -> tao', hidden_states, w3_weights);  hidden_states = w3_weights = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:39 in fused_moe_forward_native, code: expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
                mul_2: "bf16[1, 2, 3584]" = x1_1 * x3;  x1_1 = x3 = None
                expert_outs: "bf16[1, 2, 4096]" = torch.functional.einsum('tao, taio -> tai', mul_2, w2_weights);  mul_2 = w2_weights = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:40 in fused_moe_forward_native, code: return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))
                to_4: "bf16[1, 2]" = topk_weights_3.to(torch.bfloat16);  topk_weights_3 = None
                combine_input: "bf16[1, 4096]" = torch.functional.einsum('tai,ta -> ti', expert_outs, to_4);  expert_outs = to_4 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:840 in forward, code: final_hidden_states = final_hidden_states[
                getitem_6: "bf16[1, 4096]" = combine_input[(Ellipsis, slice(None, 4096, None))];  combine_input = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:842 in forward, code: ].contiguous()
                final_hidden_states: "bf16[1, 4096]" = getitem_6.contiguous();  getitem_6 = None
                return (final_hidden_states, residual)
                
    class inductor_1(torch.nn.Module):
        def forward(self, final_hidden_states: "bf16[1, 4096]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_1 = final_hidden_states.is_cpu;  getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(final_hidden_states, group_name = 'tp:0');  final_hidden_states = inplace_all_reduce = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, final_hidden_states: "bf16[1, 4096]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_1 = final_hidden_states.is_cpu;  getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(final_hidden_states, group_name = 'tp:0');  final_hidden_states = inplace_all_reduce = None
                return ()
                
    class thunder_2(torch.nn.Module):
        def forward(self, final_hidden_states: "bf16[1, 4096]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/mixtral.py:119 in forward, code: return final_hidden_states.view(orig_shape)
            hidden_states_1: "bf16[1, 4096]" = final_hidden_states.view((1, 4096));  final_hidden_states = None
            return hidden_states_1
            
        class _model(torch.nn.Module):
            def forward(self, final_hidden_states: "bf16[1, 4096]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/mixtral.py:119 in forward, code: return final_hidden_states.view(orig_shape)
                hidden_states_1: "bf16[1, 4096]" = final_hidden_states.view((1, 4096));  final_hidden_states = None
                return hidden_states_1
                