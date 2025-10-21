class GraphModule(torch.nn.Module):
    def forward(self, L_hidden_states_: "bf16[1, 2048]", L_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2048]", L_self_modules_linear_attn_modules_in_proj_qkvz_parameters_weight_: "bf16[1536, 2048]", L_self_modules_linear_attn_modules_in_proj_ba_parameters_weight_: "bf16[8, 2048]", L_self_modules_linear_attn_modules_conv1d_parameters_weight_: "bf16[1024, 1, 4]", L_self_modules_linear_attn_parameters_A_log_: "f32[4]", L_self_modules_linear_attn_parameters_dt_bias_: "bf16[4]", L_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_0_: "bf16[36, 513, 1024, 3]", L_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_1_: "f32[36, 513, 4, 128, 128]", L_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_mamba_cache_indices: "i32[1]", L_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_query_start_loc: "i32[2]", L_self_modules_linear_attn_modules_norm_parameters_weight_: "bf16[128]", L_self_modules_linear_attn_modules_out_proj_parameters_weight_: "bf16[2048, 512]", L_self_layer_communicator_post_attention_layernorm_parameters_weight_: "bf16[2048]", L_self_modules_mlp_modules_shared_expert_modules_gate_up_proj_parameters_weight_: "bf16[128, 2048]", L_self_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_: "bf16[2048, 64]", L_self_modules_mlp_modules_shared_expert_gate_parameters_weight_: "bf16[1, 2048]", L_self_modules_mlp_modules_gate_parameters_weight_: "bf16[512, 2048]", L_self_modules_mlp_modules_experts_parameters_w13_weight_: "bf16[512, 128, 2048]", L_self_modules_mlp_modules_experts_parameters_w2_weight_: "bf16[512, 2048, 64]"):
        l_hidden_states_ = L_hidden_states_
        l_self_layer_communicator_input_layernorm_parameters_weight_ = L_self_layer_communicator_input_layernorm_parameters_weight_
        l_self_modules_linear_attn_modules_in_proj_qkvz_parameters_weight_ = L_self_modules_linear_attn_modules_in_proj_qkvz_parameters_weight_
        l_self_modules_linear_attn_modules_in_proj_ba_parameters_weight_ = L_self_modules_linear_attn_modules_in_proj_ba_parameters_weight_
        l_self_modules_linear_attn_modules_conv1d_parameters_weight_ = L_self_modules_linear_attn_modules_conv1d_parameters_weight_
        l_self_modules_linear_attn_parameters_a_log_ = L_self_modules_linear_attn_parameters_A_log_
        l_self_modules_linear_attn_parameters_dt_bias_ = L_self_modules_linear_attn_parameters_dt_bias_
        l_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_0_ = L_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_0_
        l_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_1_ = L_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_1_
        l_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_mamba_cache_indices = L_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_mamba_cache_indices
        l_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_query_start_loc = L_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_query_start_loc
        l_self_modules_linear_attn_modules_norm_parameters_weight_ = L_self_modules_linear_attn_modules_norm_parameters_weight_
        l_self_modules_linear_attn_modules_out_proj_parameters_weight_ = L_self_modules_linear_attn_modules_out_proj_parameters_weight_
        l_self_layer_communicator_post_attention_layernorm_parameters_weight_ = L_self_layer_communicator_post_attention_layernorm_parameters_weight_
        l_self_modules_mlp_modules_shared_expert_modules_gate_up_proj_parameters_weight_ = L_self_modules_mlp_modules_shared_expert_modules_gate_up_proj_parameters_weight_
        l_self_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_ = L_self_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_
        l_self_modules_mlp_modules_shared_expert_gate_parameters_weight_ = L_self_modules_mlp_modules_shared_expert_gate_parameters_weight_
        l_self_modules_mlp_modules_gate_parameters_weight_ = L_self_modules_mlp_modules_gate_parameters_weight_
        l_self_modules_mlp_modules_experts_parameters_w13_weight_ = L_self_modules_mlp_modules_experts_parameters_w13_weight_
        l_self_modules_mlp_modules_experts_parameters_w2_weight_ = L_self_modules_mlp_modules_experts_parameters_w2_weight_
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:272 in forward_native, code: x = x.float()
        x: "f32[1, 2048]" = l_hidden_states_.float()
        
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
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:399 in _forward_input_proj, code: current_stream = torch.cuda.current_stream()
        current_stream = torch.cuda.current_stream()
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:400 in _forward_input_proj, code: self.alt_stream.wait_stream(current_stream)
        stream = torch.cuda.streams.Stream(stream_id = 35, device_index = 5, device_type = 1)
        wait_stream = stream.wait_stream(current_stream);  wait_stream = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
        output_parallel: "bf16[1, 1536]" = torch._C._nn.linear(x_3, l_self_modules_linear_attn_modules_in_proj_qkvz_parameters_weight_, None);  l_self_modules_linear_attn_modules_in_proj_qkvz_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:402 in _forward_input_proj, code: with torch.cuda.stream(self.alt_stream):
        current_stream_1 = torch.cuda.current_stream(None)
        set_stream = torch.cuda.set_stream(stream);  set_stream = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
        output_parallel_1: "bf16[1, 8]" = torch._C._nn.linear(x_3, l_self_modules_linear_attn_modules_in_proj_ba_parameters_weight_, None);  x_3 = l_self_modules_linear_attn_modules_in_proj_ba_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:402 in _forward_input_proj, code: with torch.cuda.stream(self.alt_stream):
        set_stream_1 = torch.cuda.set_stream(current_stream_1);  current_stream_1 = set_stream_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:404 in _forward_input_proj, code: current_stream.wait_stream(self.alt_stream)
        wait_stream_1 = current_stream.wait_stream(stream);  current_stream = stream = wait_stream_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:158 in fused_qkvzba_split_reshape_cat, code: mixed_qkv = torch.empty(
        mixed_qkv: "bf16[1, 1024]" = torch.empty([1, 1024], dtype = torch.bfloat16, device = device(type='cuda', index=5))
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:163 in fused_qkvzba_split_reshape_cat, code: z = torch.empty(
        z: "bf16[1, 4, 128]" = torch.empty([1, 4, 128], dtype = torch.bfloat16, device = device(type='cuda', index=5))
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:168 in fused_qkvzba_split_reshape_cat, code: b = torch.empty(
        b: "bf16[1, 4]" = torch.empty([1, 4], dtype = torch.bfloat16, device = device(type='cuda', index=5))
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:173 in fused_qkvzba_split_reshape_cat, code: a = torch.empty_like(b)
        a: "bf16[1, 4]" = torch.empty_like(b)
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:175 in fused_qkvzba_split_reshape_cat, code: fused_qkvzba_split_reshape_cat_kernel[grid](
        triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 31, constant_args_idx = 36, grid = [(1, 2, 1)], tma_descriptor_metadata = {}, kwargs = {'mixed_qkv': mixed_qkv, 'z': z, 'b': b, 'a': a, 'mixed_qkvz': output_parallel, 'mixed_ba': output_parallel_1});  output_parallel = output_parallel_1 = triton_kernel_wrapper_mutation = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:442 in forward, code: conv_weights = self.conv1d.weight.view(
        conv_weights: "bf16[1024, 4]" = l_self_modules_linear_attn_modules_conv1d_parameters_weight_.view(1024, 4);  l_self_modules_linear_attn_modules_conv1d_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/mem_cache/memory_pool.py:187 in get_mamba_params, code: return [self.mamba_cache[i][layer_id] for i in range(len(self.mamba_cache))]
        conv_states: "bf16[513, 1024, 3]" = l_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_0_[0];  l_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_0_ = None
        ssm_states: "f32[513, 4, 128, 128]" = l_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_1_[0];  l_kwargs_forward_batch_attn_backend_attn_backend_list_1_req_to_token_pool_mamba_pool_mamba_cache_1_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py:949 in causal_conv1d_update, code: x = x.unsqueeze(-1)
        x_4: "bf16[1, 1024, 1]" = mixed_qkv.unsqueeze(-1);  mixed_qkv = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py:1003 in causal_conv1d_update, code: _causal_conv1d_update_kernel[grid](
        triton_kernel_wrapper_mutation_1 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 37, grid = [(1, 4, 1)], tma_descriptor_metadata = {}, kwargs = {'x_ptr': x_4, 'w_ptr': conv_weights, 'conv_state_ptr': conv_states, 'conv_state_indices_ptr': l_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_mamba_cache_indices, 'intermediate_conv_window_ptr': x_4, 'o_ptr': x_4});  conv_weights = conv_states = triton_kernel_wrapper_mutation_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py:1051 in causal_conv1d_update, code: out = out.squeeze(-1)
        out: "bf16[1, 1024]" = x_4.squeeze(-1);  x_4 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:231 in forward_decode, code: query, key, value = torch.split(
        split = torch.functional.split(out, [256, 256, 512], dim = -1);  out = None
        query: "bf16[1, 256]" = split[0]
        key: "bf16[1, 256]" = split[1]
        value: "bf16[1, 512]" = split[2];  split = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:243 in forward_decode, code: query = query.view(1, seq_len, num_heads, head_k_dim)
        query_1: "bf16[1, 1, 2, 128]" = query.view(1, 1, 2, 128);  query = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:244 in forward_decode, code: key = key.view(1, seq_len, num_heads, head_k_dim)
        key_1: "bf16[1, 1, 2, 128]" = key.view(1, 1, 2, 128);  key = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:245 in forward_decode, code: value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)
        value_1: "bf16[1, 1, 4, 128]" = value.view(1, 1, 4, 128);  value = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/utils.py:146 in wrapper, code: k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
        contiguous: "f32[4]" = l_self_modules_linear_attn_parameters_a_log_.contiguous();  l_self_modules_linear_attn_parameters_a_log_ = None
        contiguous_1: "bf16[4]" = l_self_modules_linear_attn_parameters_dt_bias_.contiguous();  l_self_modules_linear_attn_parameters_dt_bias_ = None
        contiguous_2: "bf16[1, 1, 2, 128]" = query_1.contiguous();  query_1 = None
        contiguous_3: "bf16[1, 1, 2, 128]" = key_1.contiguous();  key_1 = None
        contiguous_4: "bf16[1, 1, 4, 128]" = value_1.contiguous();  value_1 = None
        contiguous_5: "bf16[1, 4]" = a.contiguous();  a = None
        contiguous_6: "bf16[1, 4]" = b.contiguous();  b = None
        contiguous_7: "f32[513, 4, 128, 128]" = ssm_states.contiguous();  ssm_states = None
        contiguous_8: "i32[1]" = l_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_mamba_cache_indices.contiguous();  l_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_mamba_cache_indices = None
        contiguous_9: "i32[2]" = l_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_query_start_loc.contiguous();  l_kwargs_forward_batch_attn_backend_attn_backend_list_1_forward_metadata_query_start_loc = None
        
        # No stacktrace found for following nodes
        _cuda_exchange_device = torch._C._cuda_exchangeDevice(5)
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py:201 in fused_sigmoid_gating_delta_rule_update, code: o = q.new_empty(NK, *v.shape)
        o: "bf16[1, 1, 1, 4, 128]" = contiguous_2.new_empty(1, 1, 1, 4, 128)
        
         # File: /home/mshinokawa/.local/lib/python3.12/site-packages/triton/runtime/autotuner.py:453 in run, code: return self.fn.run(*args, **kwargs)
        triton_kernel_wrapper_mutation_2 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 32, constant_args_idx = 38, grid = [(1, 16, 4)], tma_descriptor_metadata = {}, kwargs = {'A_log': contiguous, 'a': contiguous_5, 'dt_bias': contiguous_1, 'q': contiguous_2, 'k': contiguous_3, 'v': contiguous_4, 'b': contiguous_6, 'o': o, 'h0_source': contiguous_7, 'h0_indices': contiguous_8, 'cu_seqlens': contiguous_9});  contiguous = contiguous_5 = contiguous_1 = contiguous_2 = contiguous_3 = contiguous_4 = contiguous_6 = contiguous_7 = contiguous_8 = contiguous_9 = triton_kernel_wrapper_mutation_2 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py:231 in fused_sigmoid_gating_delta_rule_update, code: o = o.squeeze(0)
        o_1: "bf16[1, 1, 4, 128]" = o.squeeze(0);  o = None
        
        # No stacktrace found for following nodes
        _cuda_maybe_exchange_device = torch._C._cuda_maybeExchangeDevice(_cuda_exchange_device);  _cuda_exchange_device = _cuda_maybe_exchange_device = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:478 in forward, code: core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        core_attn_out: "bf16[4, 128]" = o_1.reshape(-1, 128);  o_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:479 in forward, code: z = z.reshape(-1, z.shape[-1])
        z_1: "bf16[4, 128]" = z.reshape(-1, 128);  z = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/layernorm_gated.py:202 in forward, code: x = x.reshape(-1, x.shape[-1])
        x_5: "bf16[4, 128]" = core_attn_out.reshape(-1, 128);  core_attn_out = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/layernorm_gated.py:207 in forward, code: z = z.reshape(-1, z.shape[-1])
        z_2: "bf16[4, 128]" = z_1.reshape(-1, 128);  z_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/layernorm_gated.py:210 in forward, code: weight = weight.contiguous()
        weight: "bf16[128]" = l_self_modules_linear_attn_modules_norm_parameters_weight_.contiguous();  l_self_modules_linear_attn_modules_norm_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/layernorm_gated.py:145 in _layer_norm_fwd, code: out = torch.empty_like(x)
        out_1: "bf16[4, 128]" = torch.empty_like(x_5)
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/layernorm_gated.py:152 in _layer_norm_fwd, code: rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        rstd: "f32[4]" = torch.empty((4,), dtype = torch.float32, device = device(type='cuda', index=5))
        
        # No stacktrace found for following nodes
        _cuda_exchange_device_1 = torch._C._cuda_exchangeDevice(5)
        
         # File: /home/mshinokawa/.local/lib/python3.12/site-packages/triton/runtime/autotuner.py:453 in run, code: return self.fn.run(*args, **kwargs)
        triton_kernel_wrapper_mutation_3 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 33, constant_args_idx = 39, grid = [(4, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': x_5, 'Y': out_1, 'W': weight, 'Z': z_2, 'Rstd': rstd});  x_5 = weight = z_2 = rstd = triton_kernel_wrapper_mutation_3 = None
        
        # No stacktrace found for following nodes
        _cuda_maybe_exchange_device_1 = torch._C._cuda_maybeExchangeDevice(_cuda_exchange_device_1);  _cuda_exchange_device_1 = _cuda_maybe_exchange_device_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/fla/layernorm_gated.py:223 in forward, code: return y.reshape(x_shape_og)
        core_attn_out_1: "bf16[4, 128]" = out_1.reshape((4, 128));  out_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:481 in forward, code: core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out_2: "bf16[1, 4, 128]" = core_attn_out_1.reshape((1, 4, 128));  core_attn_out_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen3_next.py:482 in forward, code: core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        core_attn_out_3: "bf16[1, 512]" = core_attn_out_2.reshape(1, -1);  core_attn_out_2 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
        output_parallel_2: "bf16[1, 2048]" = torch._C._nn.linear(core_attn_out_3, l_self_modules_linear_attn_modules_out_proj_parameters_weight_, None);  core_attn_out_3 = l_self_modules_linear_attn_modules_out_proj_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
        getattr_1 = output_parallel_2.is_cpu;  getattr_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
        inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(output_parallel_2, group_name = 'tp:0');  inplace_all_reduce = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:269 in forward_native, code: x = x + residual
        x_6: "bf16[1, 2048]" = output_parallel_2 + l_hidden_states_;  output_parallel_2 = l_hidden_states_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:272 in forward_native, code: x = x.float()
        x_7: "f32[1, 2048]" = x_6.float()
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:273 in forward_native, code: variance = x.pow(2).mean(dim=-1, keepdim=True)
        pow_2: "f32[1, 2048]" = x_7.pow(2)
        variance_1: "f32[1, 1]" = pow_2.mean(dim = -1, keepdim = True);  pow_2 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:274 in forward_native, code: x = x * torch.rsqrt(variance + self.variance_epsilon)
        add_3: "f32[1, 1]" = variance_1 + 1e-06;  variance_1 = None
        rsqrt_1: "f32[1, 1]" = torch.rsqrt(add_3);  add_3 = None
        x_8: "f32[1, 2048]" = x_7 * rsqrt_1;  x_7 = rsqrt_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:275 in forward_native, code: x = x * (1.0 + self.weight.float())
        float_4: "f32[2048]" = l_self_layer_communicator_post_attention_layernorm_parameters_weight_.float();  l_self_layer_communicator_post_attention_layernorm_parameters_weight_ = None
        add_4: "f32[2048]" = 1.0 + float_4;  float_4 = None
        x_9: "f32[1, 2048]" = x_8 * add_4;  x_8 = add_4 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:276 in forward_native, code: x = x.to(orig_dtype)
        x_10: "bf16[1, 2048]" = x_9.to(torch.bfloat16);  x_9 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:214 in forward, code: hidden_states = hidden_states.view(-1, hidden_dim)
        hidden_states: "bf16[1, 2048]" = x_10.view(-1, 2048);  x_10 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:218 in forward, code: self.alt_stream is not None
        stream_1 = torch.cuda.streams.Stream(stream_id = 35, device_index = 5, device_type = 1)
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:196 in forward_normal_dual_stream, code: current_stream = torch.cuda.current_stream()
        current_stream_2 = torch.cuda.current_stream()
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:197 in forward_normal_dual_stream, code: self.alt_stream.wait_stream(current_stream)
        wait_stream_2 = stream_1.wait_stream(current_stream_2);  wait_stream_2 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:198 in forward_normal_dual_stream, code: shared_output = self._forward_shared_experts(hidden_states.clone())
        clone: "bf16[1, 2048]" = hidden_states.clone()
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
        output_parallel_3: "bf16[1, 128]" = torch._C._nn.linear(clone, l_self_modules_mlp_modules_shared_expert_modules_gate_up_proj_parameters_weight_, None);  l_self_modules_mlp_modules_shared_expert_modules_gate_up_proj_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/activation.py:64 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
        getitem_5: "bf16[1, 64]" = output_parallel_3[(Ellipsis, slice(None, 64, None))]
        silu: "bf16[1, 64]" = torch.nn.functional.silu(getitem_5);  getitem_5 = None
        getitem_6: "bf16[1, 64]" = output_parallel_3[(Ellipsis, slice(64, None, None))];  output_parallel_3 = None
        x_11: "bf16[1, 64]" = silu * getitem_6;  silu = getitem_6 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
        output_parallel_4: "bf16[1, 2048]" = torch._C._nn.linear(x_11, l_self_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_, None);  x_11 = l_self_modules_mlp_modules_shared_expert_modules_down_proj_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:182 in _forward_shared_experts, code: F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
        linear_5: "bf16[1, 1]" = torch._C._nn.linear(clone, l_self_modules_mlp_modules_shared_expert_gate_parameters_weight_, None);  clone = l_self_modules_mlp_modules_shared_expert_gate_parameters_weight_ = None
        sigmoid: "bf16[1, 1]" = torch.nn.functional.sigmoid(linear_5);  linear_5 = None
        shared_output: "bf16[1, 2048]" = sigmoid * output_parallel_4;  sigmoid = output_parallel_4 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:200 in forward_normal_dual_stream, code: with torch.cuda.stream(self.alt_stream):
        current_stream_3 = torch.cuda.current_stream(None)
        set_stream_2 = torch.cuda.set_stream(stream_1);  set_stream_2 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
        output: "bf16[1, 512]" = torch._C._nn.linear(hidden_states, l_self_modules_mlp_modules_gate_parameters_weight_, None);  l_self_modules_mlp_modules_gate_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:396 in fused_topk_torch_native, code: topk_weights = torch.empty(
        topk_weights: "f32[1, 10]" = torch.empty(1, 10, dtype = torch.float32, device = device(type='cuda', index=5));  topk_weights = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:399 in fused_topk_torch_native, code: topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        topk_ids: "i32[1, 10]" = torch.empty(1, 10, dtype = torch.int32, device = device(type='cuda', index=5));  topk_ids = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:400 in fused_topk_torch_native, code: topk_weights = F.softmax(gating_output.float(), dim=-1)
        float_5: "f32[1, 512]" = output.float();  output = None
        topk_weights_1: "f32[1, 512]" = torch.nn.functional.softmax(float_5, dim = -1);  float_5 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:401 in fused_topk_torch_native, code: topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
        topk = torch.topk(topk_weights_1, 10, dim = -1);  topk_weights_1 = None
        topk_weights_2: "f32[1, 10]" = topk[0]
        topk_ids_1: "i64[1, 10]" = topk[1];  topk = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:404 in fused_topk_torch_native, code: topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        sum_1: "f32[1, 1]" = topk_weights_2.sum(dim = -1, keepdim = True)
        topk_weights_3: "f32[1, 10]" = topk_weights_2 / sum_1;  topk_weights_2 = sum_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:28 in fused_moe_forward_native, code: w13_weights = layer.w13_weight[topk_ids]
        w13_weights: "bf16[1, 10, 128, 2048]" = l_self_modules_mlp_modules_experts_parameters_w13_weight_[topk_ids_1];  l_self_modules_mlp_modules_experts_parameters_w13_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:29 in fused_moe_forward_native, code: w1_weights, w3_weights = torch.chunk(w13_weights, 2, dim=2)
        chunk = torch.chunk(w13_weights, 2, dim = 2);  w13_weights = None
        w1_weights: "bf16[1, 10, 64, 2048]" = chunk[0]
        w3_weights: "bf16[1, 10, 64, 2048]" = chunk[1];  chunk = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:30 in fused_moe_forward_native, code: w2_weights = layer.w2_weight[topk_ids]
        w2_weights: "bf16[1, 10, 2048, 64]" = l_self_modules_mlp_modules_experts_parameters_w2_weight_[topk_ids_1];  l_self_modules_mlp_modules_experts_parameters_w2_weight_ = topk_ids_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:31 in fused_moe_forward_native, code: x1 = torch.einsum("ti,taoi -> tao", x, w1_weights)
        x1: "bf16[1, 10, 64]" = torch.functional.einsum('ti,taoi -> tao', hidden_states, w1_weights);  w1_weights = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:33 in fused_moe_forward_native, code: x1 = F.silu(x1)
        x1_1: "bf16[1, 10, 64]" = torch.nn.functional.silu(x1);  x1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:38 in fused_moe_forward_native, code: x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        x3: "bf16[1, 10, 64]" = torch.functional.einsum('ti, taoi -> tao', hidden_states, w3_weights);  hidden_states = w3_weights = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:39 in fused_moe_forward_native, code: expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
        mul_6: "bf16[1, 10, 64]" = x1_1 * x3;  x1_1 = x3 = None
        expert_outs: "bf16[1, 10, 2048]" = torch.functional.einsum('tao, taio -> tai', mul_6, w2_weights);  mul_6 = w2_weights = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_native.py:40 in fused_moe_forward_native, code: return torch.einsum("tai,ta -> ti", expert_outs, topk_weights.to(expert_outs.dtype))
        to_2: "bf16[1, 10]" = topk_weights_3.to(torch.bfloat16);  topk_weights_3 = None
        combine_input: "bf16[1, 2048]" = torch.functional.einsum('tai,ta -> ti', expert_outs, to_2);  expert_outs = to_2 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:840 in forward, code: final_hidden_states = final_hidden_states[
        getitem_13: "bf16[1, 2048]" = combine_input[(Ellipsis, slice(None, 2048, None))];  combine_input = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:842 in forward, code: ].contiguous()
        final_hidden_states: "bf16[1, 2048]" = getitem_13.contiguous();  getitem_13 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:200 in forward_normal_dual_stream, code: with torch.cuda.stream(self.alt_stream):
        set_stream_3 = torch.cuda.set_stream(current_stream_3);  current_stream_3 = set_stream_3 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:203 in forward_normal_dual_stream, code: current_stream.wait_stream(self.alt_stream)
        wait_stream_3 = current_stream_2.wait_stream(stream_1);  current_stream_2 = stream_1 = wait_stream_3 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:231 in forward, code: final_hidden_states = final_hidden_states + shared_output
        final_hidden_states_1: "bf16[1, 2048]" = final_hidden_states + shared_output;  final_hidden_states = shared_output = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
        getattr_2 = final_hidden_states_1.is_cpu;  getattr_2 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
        inplace_all_reduce_1 = torch.ops.sglang.inplace_all_reduce(final_hidden_states_1, group_name = 'tp:0');  inplace_all_reduce_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/models/qwen2_moe.py:235 in forward, code: return final_hidden_states.view(num_tokens, hidden_dim)
        hidden_states_1: "bf16[1, 2048]" = final_hidden_states_1.view(1, 2048);  final_hidden_states_1 = None
        return (hidden_states_1, x_6)
        