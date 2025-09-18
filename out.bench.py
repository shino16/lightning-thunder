Loading model with {'name': 'Gemma-2-27b', 'hf_config': {'org': 'google', 'name': 'gemma-2-27b'}, 'scale_embeddings': True, 'attention_scores_scalar': 144, 'block_size': 8192, 'sliding_window_size': 4096, 'sliding_window_layer_placing': 2, 'vocab_size': 256000, 'padding_multiple': 512, 'padded_vocab_size': 256000, 'n_layer': 2, 'n_head': 32, 'head_size': 128, 'n_embd': 2, 'rotary_percentage': 1.0, 'parallel_residual': False, 'bias': False, 'lm_head_bias': False, 'n_query_groups': 16, 'shared_attention_norm': False, 'norm_class_name': 'RMSNorm', 'post_attention_norm': True, 'post_mlp_norm': True, 'norm_eps': 1e-05, 'mlp_class_name': 'GemmaMLP', 'gelu_approximate': 'tanh', 'intermediate_size': 16, 'rope_condense_ratio': 1, 'rope_base': 10000, 'rope_adjustments': None, 'n_expert': 0, 'n_expert_per_token': 0, 'attention_logit_softcapping': 50.0, 'final_logit_softcapping': 30.0, 'rope_n_elem': 128}
Time to instantiate model: 0.06 seconds.
Model Flops/Throughput calculation failed for model Gemma-2-27b. Skipping throughput metric collection.
iter 0: loss 13.0000, iter time: 10503.50ms, t: 8192
Model name: Gemma-2-27b
Seq Length: 8192
Micro BS: 1
Global BS: 1
Number of Layers: 2
Number of parameters: 0.00B
Distributed Mode: none
Compiler: dynamo_thunder
Low Precision Mode: none
Average iter time: 10503.50 ms
Memory used: 47.52 GB
Saved for backward size: 4001.07 MiB
Saved for backward number of tensors: 6
class GraphModule(torch.nn.Module):
    def forward(self, x_1: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        wrap_body_0 = self.wrap_body_0

         # File: <debug>:0 in <debug>, code: memory_events = [snapshot - 6610432 bytes], total 6610432 bytes allocated
        memory_events_wrap_body_0 = thunder_dev_utils_debug_memory_transform_memory_events_wrap_body_0();  memory_events_wrap_body_0 = None

         # File: <debug>:0 in <debug>, code: memory_events = [], total 6610432 bytes allocated
        memory_events_x_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_1();  memory_events_x_1 = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None
        memory_events_cos = thunder_dev_utils_debug_memory_transform_memory_events_cos();  memory_events_cos = None
        memory_events_sin = thunder_dev_utils_debug_memory_transform_memory_events_sin();  memory_events_sin = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, x_1, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, cos, sin, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_, use_reentrant = False);  wrap_body_0 = x_1 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = cos = sin = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

         # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, segment_alloc - 134217728 bytes, alloc - 134217728 bytes, segment_alloc - 33554432 bytes, alloc - 33554432 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 33554432 bytes, alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 33554432 bytes, alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 134217728 bytes, segment_alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 134217728 bytes, segment_alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, free_requested - 8589934592 bytes, free_completed - 8589934592 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 32768 bytes, alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 32768 bytes, alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, segment_alloc - 2097152 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 17490273792 bytes allocated
        memory_events_tag_activation_checkpoint = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint();  memory_events_tag_activation_checkpoint = None
        return tag_activation_checkpoint

        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

    class wrap_body_0(torch.nn.Module):
        def forward(self, x_1: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x: "f32[1, 8192, 2]" = x_1.float()

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul: "f32[1, 8192, 2]" = x * x
            norm_x: "f32[1, 8192, 1]" = torch.mean(mul, dim = -1, keepdim = True);  mul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add: "f32[1, 8192, 1]" = norm_x + 1e-05;  norm_x = None
            rsqrt: "f32[1, 8192, 1]" = torch.rsqrt(add);  add = None
            x_normed: "f32[1, 8192, 2]" = x * rsqrt;  x = rsqrt = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_2: "f32[2]" = weight.float();  weight = None
            mul_2: "f32[1, 8192, 2]" = x_normed * float_2;  x_normed = float_2 = None
            x_normed_1: "bf16[1, 8192, 2]" = mul_2.to(dtype = torch.bfloat16);  mul_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:269 in forward, code: qkv = self.attn(x)
            qkv: "bf16[1, 8192, 8192]" = torch._C._nn.linear(x_normed_1, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, None);  x_normed_1 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:274 in forward, code: qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv_1: "bf16[1, 8192, 16, 4, 128]" = qkv.view(1, 8192, 16, 4, 128);  qkv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:275 in forward, code: qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
            qkv_2: "bf16[1, 16, 4, 8192, 128]" = qkv_1.permute(0, 2, 3, 1, 4);  qkv_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            split = qkv_2.split((2, 1, 1), dim = 2);  qkv_2 = None
            q: "bf16[1, 16, 2, 8192, 128]" = split[0]
            k: "bf16[1, 16, 1, 8192, 128]" = split[1]
            v: "bf16[1, 16, 1, 8192, 128]" = split[2];  split = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:284 in forward, code: k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            k_1: "bf16[1, 16, 2, 8192, 128]" = k.expand(1, 16, 2, 8192, 128);  k = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:285 in forward, code: v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v_1: "bf16[1, 16, 2, 8192, 128]" = v.expand(1, 16, 2, 8192, 128);  v = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:287 in forward, code: q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            q_1: "bf16[1, 32, 8192, 128]" = q.reshape(1, -1, 8192, 128);  q = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:288 in forward, code: k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            k_2: "bf16[1, 32, 8192, 128]" = k_1.reshape(1, -1, 8192, 128);  k_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:289 in forward, code: v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
            v_2: "bf16[1, 32, 8192, 128]" = v_1.reshape(1, -1, 8192, 128);  v_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:291 in forward, code: q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            getitem_3: "bf16[1, 32, 8192, 128]" = q_1[(Ellipsis, slice(None, 128, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(None, 64, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(64, None, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg: "bf16[1, 32, 8192, 64]" = -x2;  x2 = None
            rotated: "bf16[1, 32, 8192, 128]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_1: "bf16[1, 8192, 128]" = cos.unsqueeze(-3)

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_1: "bf16[1, 8192, 128]" = sin.unsqueeze(-3)

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_3: "bf16[1, 32, 8192, 128]" = getitem_3 * cos_1;  getitem_3 = cos_1 = None
            mul_4: "bf16[1, 32, 8192, 128]" = rotated * sin_1;  rotated = sin_1 = None
            roped: "bf16[1, 32, 8192, 128]" = mul_3 + mul_4;  mul_3 = mul_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            q_roped: "bf16[1, 32, 8192, 128]" = roped.to(dtype = torch.bfloat16);  roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:292 in forward, code: k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            getitem_6: "bf16[1, 32, 8192, 128]" = k_2[(Ellipsis, slice(None, 128, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(None, 64, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(64, None, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg_1: "bf16[1, 32, 8192, 64]" = -x2_1;  x2_1 = None
            rotated_1: "bf16[1, 32, 8192, 128]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_2: "bf16[1, 8192, 128]" = cos.unsqueeze(-3);  cos = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_2: "bf16[1, 8192, 128]" = sin.unsqueeze(-3);  sin = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_5: "bf16[1, 32, 8192, 128]" = getitem_6 * cos_2;  getitem_6 = cos_2 = None
            mul_6: "bf16[1, 32, 8192, 128]" = rotated_1 * sin_2;  rotated_1 = sin_2 = None
            roped_1: "bf16[1, 32, 8192, 128]" = mul_5 + mul_6;  mul_5 = mul_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            k_roped: "bf16[1, 32, 8192, 128]" = roped_1.to(dtype = torch.bfloat16);  roped_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            getitem_9: "bf16[1, 32, 8192, 0]" = q_1[(Ellipsis, slice(128, None, None))];  q_1 = None
            q_2: "bf16[1, 32, 8192, 128]" = torch.cat((q_roped, getitem_9), dim = -1);  q_roped = getitem_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            getitem_10: "bf16[1, 32, 8192, 0]" = k_2[(Ellipsis, slice(128, None, None))];  k_2 = None
            k_3: "bf16[1, 32, 8192, 128]" = torch.cat((k_roped, getitem_10), dim = -1);  k_roped = getitem_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:313 in forward, code: mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
            ones: "bf16[8192, 8192]" = torch.ones(8192, 8192, dtype = torch.bfloat16, device = device(type='cuda', index=0))
            mask: "bf16[8192, 8192]" = ones.triu(diagonal = 1);  ones = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:314 in forward, code: mask.masked_fill_(mask.bool(), float("-inf"))
            bool_1: "b8[8192, 8192]" = mask.bool()
            masked_fill_: "bf16[8192, 8192]" = mask.masked_fill_(bool_1, -inf);  bool_1 = masked_fill_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:315 in forward, code: sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            ones_like: "bf16[8192, 8192]" = torch.ones_like(mask)
            sliding_window_bias: "bf16[8192, 8192]" = ones_like.tril(diagonal = -4096);  ones_like = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:316 in forward, code: sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            bool_2: "b8[8192, 8192]" = sliding_window_bias.bool()
            masked_fill__1: "bf16[8192, 8192]" = sliding_window_bias.masked_fill_(bool_2, -inf);  bool_2 = masked_fill__1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:317 in forward, code: mask += sliding_window_bias
            mask += sliding_window_bias;  mask_1: "bf16[8192, 8192]" = mask;  mask = sliding_window_bias = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            getattr_1: "bf16[1, 32, 128, 8192]" = k_3.mT;  k_3 = None
            matmul: "bf16[1, 32, 8192, 8192]" = q_2 @ getattr_1;  q_2 = getattr_1 = None
            scores: "bf16[1, 32, 8192, 8192]" = matmul * 0.08333333333333333;  matmul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            truediv: "bf16[1, 32, 8192, 8192]" = scores / 50.0;  scores = None
            tanh: "bf16[1, 32, 8192, 8192]" = torch.tanh(truediv);  truediv = None
            scores_1: "bf16[1, 32, 8192, 8192]" = tanh * 50.0;  tanh = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:341 in scaled_dot_product_attention, code: scores = scores + mask
            scores_2: "bf16[1, 32, 8192, 8192]" = scores_1 + mask_1;  scores_1 = mask_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            softmax: "f32[1, 32, 8192, 8192]" = torch.nn.functional.softmax(scores_2, dim = -1, dtype = torch.float32);  scores_2 = None
            scores_3: "bf16[1, 32, 8192, 8192]" = softmax.to(dtype = torch.bfloat16);  softmax = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:343 in scaled_dot_product_attention, code: y = scores @ v
            y: "bf16[1, 32, 8192, 128]" = scores_3 @ v_2;  scores_3 = v_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:348 in scaled_dot_product_attention, code: return y.transpose(1, 2)
            y_1: "bf16[1, 8192, 32, 128]" = y.transpose(1, 2);  y = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:321 in forward, code: y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
            y_2: "bf16[1, 8192, 4096]" = y_1.reshape(1, 8192, 4096);  y_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:324 in forward, code: return self.proj(y)
            attention_output: "bf16[1, 8192, 2]" = torch._C._nn.linear(y_2, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, None);  y_2 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_2: "f32[1, 8192, 2]" = attention_output.float();  attention_output = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_9: "f32[1, 8192, 2]" = x_2 * x_2
            norm_x_1: "f32[1, 8192, 1]" = torch.mean(mul_9, dim = -1, keepdim = True);  mul_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_5: "f32[1, 8192, 1]" = norm_x_1 + 1e-05;  norm_x_1 = None
            rsqrt_1: "f32[1, 8192, 1]" = torch.rsqrt(add_5);  add_5 = None
            x_normed_2: "f32[1, 8192, 2]" = x_2 * rsqrt_1;  x_2 = rsqrt_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_1: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_4: "f32[2]" = weight_1.float();  weight_1 = None
            mul_11: "f32[1, 8192, 2]" = x_normed_2 * float_4;  x_normed_2 = float_4 = None
            attention_output_1: "bf16[1, 8192, 2]" = mul_11.to(dtype = torch.bfloat16);  mul_11 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:236 in forward, code: x = attention_output + x
            x_3: "bf16[1, 8192, 2]" = attention_output_1 + x_1;  attention_output_1 = x_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_4: "f32[1, 8192, 2]" = x_3.float()

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_12: "f32[1, 8192, 2]" = x_4 * x_4
            norm_x_2: "f32[1, 8192, 1]" = torch.mean(mul_12, dim = -1, keepdim = True);  mul_12 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_8: "f32[1, 8192, 1]" = norm_x_2 + 1e-05;  norm_x_2 = None
            rsqrt_2: "f32[1, 8192, 1]" = torch.rsqrt(add_8);  add_8 = None
            x_normed_3: "f32[1, 8192, 2]" = x_4 * rsqrt_2;  x_4 = rsqrt_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_2: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_6: "f32[2]" = weight_2.float();  weight_2 = None
            mul_14: "f32[1, 8192, 2]" = x_normed_3 * float_6;  x_normed_3 = float_6 = None
            to_5: "bf16[1, 8192, 2]" = mul_14.to(dtype = torch.bfloat16);  mul_14 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:406 in forward, code: x_fc_1 = self.fc_1(x)
            x_fc_1: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, None);  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:407 in forward, code: x_fc_2 = self.fc_2(x)
            x_fc_2: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, None);  to_5 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            gelu: "bf16[1, 8192, 16]" = torch._C._nn.gelu(x_fc_1, approximate = 'tanh');  x_fc_1 = None
            x_5: "bf16[1, 8192, 16]" = gelu * x_fc_2;  gelu = x_fc_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:409 in forward, code: return self.proj(x)
            linear_4: "bf16[1, 8192, 2]" = torch._C._nn.linear(x_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, None);  x_5 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_6: "f32[1, 8192, 2]" = linear_4.float();  linear_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_16: "f32[1, 8192, 2]" = x_6 * x_6
            norm_x_3: "f32[1, 8192, 1]" = torch.mean(mul_16, dim = -1, keepdim = True);  mul_16 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_10: "f32[1, 8192, 1]" = norm_x_3 + 1e-05;  norm_x_3 = None
            rsqrt_3: "f32[1, 8192, 1]" = torch.rsqrt(add_10);  add_10 = None
            x_normed_4: "f32[1, 8192, 2]" = x_6 * rsqrt_3;  x_6 = rsqrt_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_3: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_8: "f32[2]" = weight_3.float();  weight_3 = None
            mul_18: "f32[1, 8192, 2]" = x_normed_4 * float_8;  x_normed_4 = float_8 = None
            to_6: "bf16[1, 8192, 2]" = mul_18.to(dtype = torch.bfloat16);  mul_18 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:237 in forward, code: x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
            x_7: "bf16[1, 8192, 2]" = to_6 + x_3;  to_6 = x_3 = None
            return (x_7,)

class GraphModule(torch.nn.Module):
    def forward(self, x_1: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        wrap_body_0 = self.wrap_body_0

         # File: <debug>:0 in <debug>, code: memory_events = [snapshot - 6610432 bytes], total 6610432 bytes allocated
        memory_events_wrap_body_0 = thunder_dev_utils_debug_memory_transform_memory_events_wrap_body_0();  memory_events_wrap_body_0 = None

         # File: <debug>:0 in <debug>, code: memory_events = [], total 6610432 bytes allocated
        memory_events_x_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_1();  memory_events_x_1 = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None
        memory_events_cos = thunder_dev_utils_debug_memory_transform_memory_events_cos();  memory_events_cos = None
        memory_events_sin = thunder_dev_utils_debug_memory_transform_memory_events_sin();  memory_events_sin = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, x_1, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, cos, sin, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_, use_reentrant = False);  wrap_body_0 = x_1 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = cos = sin = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

         # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, segment_alloc - 134217728 bytes, alloc - 134217728 bytes, segment_alloc - 33554432 bytes, alloc - 33554432 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 33554432 bytes, alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, segment_alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 33554432 bytes, alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 134217728 bytes, segment_alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 134217728 bytes, segment_alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, free_requested - 8589934592 bytes, free_completed - 8589934592 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 32768 bytes, alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 32768 bytes, alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, segment_alloc - 2097152 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 17490273792 bytes allocated
        memory_events_tag_activation_checkpoint = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint();  memory_events_tag_activation_checkpoint = None
        return tag_activation_checkpoint

        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

    class wrap_body_0(torch.nn.Module):
        def forward(self, x_1: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x: "f32[1, 8192, 2]" = x_1.float()

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul: "f32[1, 8192, 2]" = x * x
            norm_x: "f32[1, 8192, 1]" = torch.mean(mul, dim = -1, keepdim = True);  mul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add: "f32[1, 8192, 1]" = norm_x + 1e-05;  norm_x = None
            rsqrt: "f32[1, 8192, 1]" = torch.rsqrt(add);  add = None
            x_normed: "f32[1, 8192, 2]" = x * rsqrt;  x = rsqrt = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_2: "f32[2]" = weight.float();  weight = None
            mul_2: "f32[1, 8192, 2]" = x_normed * float_2;  x_normed = float_2 = None
            x_normed_1: "bf16[1, 8192, 2]" = mul_2.to(dtype = torch.bfloat16);  mul_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:269 in forward, code: qkv = self.attn(x)
            qkv: "bf16[1, 8192, 8192]" = torch._C._nn.linear(x_normed_1, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, None);  x_normed_1 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:274 in forward, code: qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv_1: "bf16[1, 8192, 16, 4, 128]" = qkv.view(1, 8192, 16, 4, 128);  qkv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:275 in forward, code: qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
            qkv_2: "bf16[1, 16, 4, 8192, 128]" = qkv_1.permute(0, 2, 3, 1, 4);  qkv_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            split = qkv_2.split((2, 1, 1), dim = 2);  qkv_2 = None
            q: "bf16[1, 16, 2, 8192, 128]" = split[0]
            k: "bf16[1, 16, 1, 8192, 128]" = split[1]
            v: "bf16[1, 16, 1, 8192, 128]" = split[2];  split = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:284 in forward, code: k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            k_1: "bf16[1, 16, 2, 8192, 128]" = k.expand(1, 16, 2, 8192, 128);  k = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:285 in forward, code: v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v_1: "bf16[1, 16, 2, 8192, 128]" = v.expand(1, 16, 2, 8192, 128);  v = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:287 in forward, code: q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            q_1: "bf16[1, 32, 8192, 128]" = q.reshape(1, -1, 8192, 128);  q = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:288 in forward, code: k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            k_2: "bf16[1, 32, 8192, 128]" = k_1.reshape(1, -1, 8192, 128);  k_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:289 in forward, code: v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
            v_2: "bf16[1, 32, 8192, 128]" = v_1.reshape(1, -1, 8192, 128);  v_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:291 in forward, code: q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            getitem_3: "bf16[1, 32, 8192, 128]" = q_1[(Ellipsis, slice(None, 128, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(None, 64, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(64, None, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg: "bf16[1, 32, 8192, 64]" = -x2;  x2 = None
            rotated: "bf16[1, 32, 8192, 128]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_1: "bf16[1, 8192, 128]" = cos.unsqueeze(-3)

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_1: "bf16[1, 8192, 128]" = sin.unsqueeze(-3)

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_3: "bf16[1, 32, 8192, 128]" = getitem_3 * cos_1;  getitem_3 = cos_1 = None
            mul_4: "bf16[1, 32, 8192, 128]" = rotated * sin_1;  rotated = sin_1 = None
            roped: "bf16[1, 32, 8192, 128]" = mul_3 + mul_4;  mul_3 = mul_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            q_roped: "bf16[1, 32, 8192, 128]" = roped.to(dtype = torch.bfloat16);  roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:292 in forward, code: k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            getitem_6: "bf16[1, 32, 8192, 128]" = k_2[(Ellipsis, slice(None, 128, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(None, 64, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(64, None, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg_1: "bf16[1, 32, 8192, 64]" = -x2_1;  x2_1 = None
            rotated_1: "bf16[1, 32, 8192, 128]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_2: "bf16[1, 8192, 128]" = cos.unsqueeze(-3);  cos = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_2: "bf16[1, 8192, 128]" = sin.unsqueeze(-3);  sin = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_5: "bf16[1, 32, 8192, 128]" = getitem_6 * cos_2;  getitem_6 = cos_2 = None
            mul_6: "bf16[1, 32, 8192, 128]" = rotated_1 * sin_2;  rotated_1 = sin_2 = None
            roped_1: "bf16[1, 32, 8192, 128]" = mul_5 + mul_6;  mul_5 = mul_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            k_roped: "bf16[1, 32, 8192, 128]" = roped_1.to(dtype = torch.bfloat16);  roped_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            getitem_9: "bf16[1, 32, 8192, 0]" = q_1[(Ellipsis, slice(128, None, None))];  q_1 = None
            q_2: "bf16[1, 32, 8192, 128]" = torch.cat((q_roped, getitem_9), dim = -1);  q_roped = getitem_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            getitem_10: "bf16[1, 32, 8192, 0]" = k_2[(Ellipsis, slice(128, None, None))];  k_2 = None
            k_3: "bf16[1, 32, 8192, 128]" = torch.cat((k_roped, getitem_10), dim = -1);  k_roped = getitem_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:313 in forward, code: mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
            ones: "bf16[8192, 8192]" = torch.ones(8192, 8192, dtype = torch.bfloat16, device = device(type='cuda', index=0))
            mask: "bf16[8192, 8192]" = ones.triu(diagonal = 1);  ones = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:314 in forward, code: mask.masked_fill_(mask.bool(), float("-inf"))
            bool_1: "b8[8192, 8192]" = mask.bool()
            masked_fill_: "bf16[8192, 8192]" = mask.masked_fill_(bool_1, -inf);  bool_1 = masked_fill_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:315 in forward, code: sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            ones_like: "bf16[8192, 8192]" = torch.ones_like(mask)
            sliding_window_bias: "bf16[8192, 8192]" = ones_like.tril(diagonal = -4096);  ones_like = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:316 in forward, code: sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            bool_2: "b8[8192, 8192]" = sliding_window_bias.bool()
            masked_fill__1: "bf16[8192, 8192]" = sliding_window_bias.masked_fill_(bool_2, -inf);  bool_2 = masked_fill__1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:317 in forward, code: mask += sliding_window_bias
            mask += sliding_window_bias;  mask_1: "bf16[8192, 8192]" = mask;  mask = sliding_window_bias = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            getattr_1: "bf16[1, 32, 128, 8192]" = k_3.mT;  k_3 = None
            matmul: "bf16[1, 32, 8192, 8192]" = q_2 @ getattr_1;  q_2 = getattr_1 = None
            scores: "bf16[1, 32, 8192, 8192]" = matmul * 0.08333333333333333;  matmul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            truediv: "bf16[1, 32, 8192, 8192]" = scores / 50.0;  scores = None
            tanh: "bf16[1, 32, 8192, 8192]" = torch.tanh(truediv);  truediv = None
            scores_1: "bf16[1, 32, 8192, 8192]" = tanh * 50.0;  tanh = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:341 in scaled_dot_product_attention, code: scores = scores + mask
            scores_2: "bf16[1, 32, 8192, 8192]" = scores_1 + mask_1;  scores_1 = mask_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            softmax: "f32[1, 32, 8192, 8192]" = torch.nn.functional.softmax(scores_2, dim = -1, dtype = torch.float32);  scores_2 = None
            scores_3: "bf16[1, 32, 8192, 8192]" = softmax.to(dtype = torch.bfloat16);  softmax = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:343 in scaled_dot_product_attention, code: y = scores @ v
            y: "bf16[1, 32, 8192, 128]" = scores_3 @ v_2;  scores_3 = v_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:348 in scaled_dot_product_attention, code: return y.transpose(1, 2)
            y_1: "bf16[1, 8192, 32, 128]" = y.transpose(1, 2);  y = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:321 in forward, code: y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
            y_2: "bf16[1, 8192, 4096]" = y_1.reshape(1, 8192, 4096);  y_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:324 in forward, code: return self.proj(y)
            attention_output: "bf16[1, 8192, 2]" = torch._C._nn.linear(y_2, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, None);  y_2 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_2: "f32[1, 8192, 2]" = attention_output.float();  attention_output = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_9: "f32[1, 8192, 2]" = x_2 * x_2
            norm_x_1: "f32[1, 8192, 1]" = torch.mean(mul_9, dim = -1, keepdim = True);  mul_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_5: "f32[1, 8192, 1]" = norm_x_1 + 1e-05;  norm_x_1 = None
            rsqrt_1: "f32[1, 8192, 1]" = torch.rsqrt(add_5);  add_5 = None
            x_normed_2: "f32[1, 8192, 2]" = x_2 * rsqrt_1;  x_2 = rsqrt_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_1: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_4: "f32[2]" = weight_1.float();  weight_1 = None
            mul_11: "f32[1, 8192, 2]" = x_normed_2 * float_4;  x_normed_2 = float_4 = None
            attention_output_1: "bf16[1, 8192, 2]" = mul_11.to(dtype = torch.bfloat16);  mul_11 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:236 in forward, code: x = attention_output + x
            x_3: "bf16[1, 8192, 2]" = attention_output_1 + x_1;  attention_output_1 = x_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_4: "f32[1, 8192, 2]" = x_3.float()

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_12: "f32[1, 8192, 2]" = x_4 * x_4
            norm_x_2: "f32[1, 8192, 1]" = torch.mean(mul_12, dim = -1, keepdim = True);  mul_12 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_8: "f32[1, 8192, 1]" = norm_x_2 + 1e-05;  norm_x_2 = None
            rsqrt_2: "f32[1, 8192, 1]" = torch.rsqrt(add_8);  add_8 = None
            x_normed_3: "f32[1, 8192, 2]" = x_4 * rsqrt_2;  x_4 = rsqrt_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_2: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_6: "f32[2]" = weight_2.float();  weight_2 = None
            mul_14: "f32[1, 8192, 2]" = x_normed_3 * float_6;  x_normed_3 = float_6 = None
            to_5: "bf16[1, 8192, 2]" = mul_14.to(dtype = torch.bfloat16);  mul_14 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:406 in forward, code: x_fc_1 = self.fc_1(x)
            x_fc_1: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, None);  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:407 in forward, code: x_fc_2 = self.fc_2(x)
            x_fc_2: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, None);  to_5 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            gelu: "bf16[1, 8192, 16]" = torch._C._nn.gelu(x_fc_1, approximate = 'tanh');  x_fc_1 = None
            x_5: "bf16[1, 8192, 16]" = gelu * x_fc_2;  gelu = x_fc_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:409 in forward, code: return self.proj(x)
            linear_4: "bf16[1, 8192, 2]" = torch._C._nn.linear(x_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, None);  x_5 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_6: "f32[1, 8192, 2]" = linear_4.float();  linear_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_16: "f32[1, 8192, 2]" = x_6 * x_6
            norm_x_3: "f32[1, 8192, 1]" = torch.mean(mul_16, dim = -1, keepdim = True);  mul_16 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_10: "f32[1, 8192, 1]" = norm_x_3 + 1e-05;  norm_x_3 = None
            rsqrt_3: "f32[1, 8192, 1]" = torch.rsqrt(add_10);  add_10 = None
            x_normed_4: "f32[1, 8192, 2]" = x_6 * rsqrt_3;  x_6 = rsqrt_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_3: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_8: "f32[2]" = weight_3.float();  weight_3 = None
            mul_18: "f32[1, 8192, 2]" = x_normed_4 * float_8;  x_normed_4 = float_8 = None
            to_6: "bf16[1, 8192, 2]" = mul_18.to(dtype = torch.bfloat16);  mul_18 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:237 in forward, code: x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
            x_7: "bf16[1, 8192, 2]" = to_6 + x_3;  to_6 = x_3 = None
            return (x_7,)

class GraphModule(torch.nn.Module):
    def forward(self, x_2: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        wrap_body_1 = self.wrap_body_1

         # File: <debug>:0 in <debug>, code: memory_events = [], total 17490241024 bytes allocated
        memory_events_wrap_body_1 = thunder_dev_utils_debug_memory_transform_memory_events_wrap_body_1();  memory_events_wrap_body_1 = None
        memory_events_x_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_2();  memory_events_x_2 = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None
        memory_events_cos = thunder_dev_utils_debug_memory_transform_memory_events_cos();  memory_events_cos = None
        memory_events_sin = thunder_dev_utils_debug_memory_transform_memory_events_sin();  memory_events_sin = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        tag_activation_checkpoint_1 = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_1, x_2, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, cos, sin, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_, use_reentrant = False);  wrap_body_1 = x_2 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = cos = sin = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

         # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 134217728 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 33554432 bytes, alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 33554432 bytes, alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, free_requested - 8589934592 bytes, free_completed - 8589934592 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 32768 bytes, alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 32768 bytes, alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 34940349952 bytes allocated
        memory_events_tag_activation_checkpoint_1 = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint_1();  memory_events_tag_activation_checkpoint_1 = None
        return tag_activation_checkpoint_1

        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

    class wrap_body_1(torch.nn.Module):
        def forward(self, x_2: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x: "f32[1, 8192, 2]" = x_2.float()

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul: "f32[1, 8192, 2]" = x * x
            norm_x: "f32[1, 8192, 1]" = torch.mean(mul, dim = -1, keepdim = True);  mul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add: "f32[1, 8192, 1]" = norm_x + 1e-05;  norm_x = None
            rsqrt: "f32[1, 8192, 1]" = torch.rsqrt(add);  add = None
            x_normed: "f32[1, 8192, 2]" = x * rsqrt;  x = rsqrt = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_2: "f32[2]" = weight.float();  weight = None
            mul_2: "f32[1, 8192, 2]" = x_normed * float_2;  x_normed = float_2 = None
            x_normed_1: "bf16[1, 8192, 2]" = mul_2.to(dtype = torch.bfloat16);  mul_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:269 in forward, code: qkv = self.attn(x)
            qkv: "bf16[1, 8192, 8192]" = torch._C._nn.linear(x_normed_1, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, None);  x_normed_1 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:274 in forward, code: qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv_1: "bf16[1, 8192, 16, 4, 128]" = qkv.view(1, 8192, 16, 4, 128);  qkv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:275 in forward, code: qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
            qkv_2: "bf16[1, 16, 4, 8192, 128]" = qkv_1.permute(0, 2, 3, 1, 4);  qkv_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            split = qkv_2.split((2, 1, 1), dim = 2);  qkv_2 = None
            q: "bf16[1, 16, 2, 8192, 128]" = split[0]
            k: "bf16[1, 16, 1, 8192, 128]" = split[1]
            v: "bf16[1, 16, 1, 8192, 128]" = split[2];  split = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:284 in forward, code: k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            k_1: "bf16[1, 16, 2, 8192, 128]" = k.expand(1, 16, 2, 8192, 128);  k = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:285 in forward, code: v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v_1: "bf16[1, 16, 2, 8192, 128]" = v.expand(1, 16, 2, 8192, 128);  v = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:287 in forward, code: q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            q_1: "bf16[1, 32, 8192, 128]" = q.reshape(1, -1, 8192, 128);  q = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:288 in forward, code: k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            k_2: "bf16[1, 32, 8192, 128]" = k_1.reshape(1, -1, 8192, 128);  k_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:289 in forward, code: v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
            v_2: "bf16[1, 32, 8192, 128]" = v_1.reshape(1, -1, 8192, 128);  v_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:291 in forward, code: q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            getitem_3: "bf16[1, 32, 8192, 128]" = q_1[(Ellipsis, slice(None, 128, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(None, 64, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(64, None, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg: "bf16[1, 32, 8192, 64]" = -x2;  x2 = None
            rotated: "bf16[1, 32, 8192, 128]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_1: "bf16[1, 8192, 128]" = cos.unsqueeze(-3)

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_1: "bf16[1, 8192, 128]" = sin.unsqueeze(-3)

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_3: "bf16[1, 32, 8192, 128]" = getitem_3 * cos_1;  getitem_3 = cos_1 = None
            mul_4: "bf16[1, 32, 8192, 128]" = rotated * sin_1;  rotated = sin_1 = None
            roped: "bf16[1, 32, 8192, 128]" = mul_3 + mul_4;  mul_3 = mul_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            q_roped: "bf16[1, 32, 8192, 128]" = roped.to(dtype = torch.bfloat16);  roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:292 in forward, code: k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            getitem_6: "bf16[1, 32, 8192, 128]" = k_2[(Ellipsis, slice(None, 128, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(None, 64, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(64, None, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg_1: "bf16[1, 32, 8192, 64]" = -x2_1;  x2_1 = None
            rotated_1: "bf16[1, 32, 8192, 128]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_2: "bf16[1, 8192, 128]" = cos.unsqueeze(-3);  cos = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_2: "bf16[1, 8192, 128]" = sin.unsqueeze(-3);  sin = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_5: "bf16[1, 32, 8192, 128]" = getitem_6 * cos_2;  getitem_6 = cos_2 = None
            mul_6: "bf16[1, 32, 8192, 128]" = rotated_1 * sin_2;  rotated_1 = sin_2 = None
            roped_1: "bf16[1, 32, 8192, 128]" = mul_5 + mul_6;  mul_5 = mul_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            k_roped: "bf16[1, 32, 8192, 128]" = roped_1.to(dtype = torch.bfloat16);  roped_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            getitem_9: "bf16[1, 32, 8192, 0]" = q_1[(Ellipsis, slice(128, None, None))];  q_1 = None
            q_2: "bf16[1, 32, 8192, 128]" = torch.cat((q_roped, getitem_9), dim = -1);  q_roped = getitem_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            getitem_10: "bf16[1, 32, 8192, 0]" = k_2[(Ellipsis, slice(128, None, None))];  k_2 = None
            k_3: "bf16[1, 32, 8192, 128]" = torch.cat((k_roped, getitem_10), dim = -1);  k_roped = getitem_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            getattr_1: "bf16[1, 32, 128, 8192]" = k_3.mT;  k_3 = None
            matmul: "bf16[1, 32, 8192, 8192]" = q_2 @ getattr_1;  q_2 = getattr_1 = None
            scores: "bf16[1, 32, 8192, 8192]" = matmul * 0.08333333333333333;  matmul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            truediv: "bf16[1, 32, 8192, 8192]" = scores / 50.0;  scores = None
            tanh: "bf16[1, 32, 8192, 8192]" = torch.tanh(truediv);  truediv = None
            scores_1: "bf16[1, 32, 8192, 8192]" = tanh * 50.0;  tanh = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:339 in scaled_dot_product_attention, code: mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
            ones: "bf16[8192, 8192]" = torch.ones(8192, 8192, dtype = torch.bfloat16, device = device(type='cuda', index=0))
            mask: "bf16[8192, 8192]" = ones.triu(diagonal = 1);  ones = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:340 in scaled_dot_product_attention, code: mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            bool_1: "b8[8192, 8192]" = mask.bool()
            masked_fill_: "bf16[8192, 8192]" = mask.masked_fill_(bool_1, -3.3895313892515355e+38);  bool_1 = masked_fill_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:341 in scaled_dot_product_attention, code: scores = scores + mask
            scores_2: "bf16[1, 32, 8192, 8192]" = scores_1 + mask;  scores_1 = mask = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            softmax: "f32[1, 32, 8192, 8192]" = torch.nn.functional.softmax(scores_2, dim = -1, dtype = torch.float32);  scores_2 = None
            scores_3: "bf16[1, 32, 8192, 8192]" = softmax.to(dtype = torch.bfloat16);  softmax = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:343 in scaled_dot_product_attention, code: y = scores @ v
            y: "bf16[1, 32, 8192, 128]" = scores_3 @ v_2;  scores_3 = v_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:348 in scaled_dot_product_attention, code: return y.transpose(1, 2)
            y_1: "bf16[1, 8192, 32, 128]" = y.transpose(1, 2);  y = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:321 in forward, code: y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
            y_2: "bf16[1, 8192, 4096]" = y_1.reshape(1, 8192, 4096);  y_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:324 in forward, code: return self.proj(y)
            attention_output: "bf16[1, 8192, 2]" = torch._C._nn.linear(y_2, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, None);  y_2 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_3: "f32[1, 8192, 2]" = attention_output.float();  attention_output = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_9: "f32[1, 8192, 2]" = x_3 * x_3
            norm_x_1: "f32[1, 8192, 1]" = torch.mean(mul_9, dim = -1, keepdim = True);  mul_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_5: "f32[1, 8192, 1]" = norm_x_1 + 1e-05;  norm_x_1 = None
            rsqrt_1: "f32[1, 8192, 1]" = torch.rsqrt(add_5);  add_5 = None
            x_normed_2: "f32[1, 8192, 2]" = x_3 * rsqrt_1;  x_3 = rsqrt_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_1: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_4: "f32[2]" = weight_1.float();  weight_1 = None
            mul_11: "f32[1, 8192, 2]" = x_normed_2 * float_4;  x_normed_2 = float_4 = None
            attention_output_1: "bf16[1, 8192, 2]" = mul_11.to(dtype = torch.bfloat16);  mul_11 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:236 in forward, code: x = attention_output + x
            x_4: "bf16[1, 8192, 2]" = attention_output_1 + x_2;  attention_output_1 = x_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_5: "f32[1, 8192, 2]" = x_4.float()

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_12: "f32[1, 8192, 2]" = x_5 * x_5
            norm_x_2: "f32[1, 8192, 1]" = torch.mean(mul_12, dim = -1, keepdim = True);  mul_12 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_8: "f32[1, 8192, 1]" = norm_x_2 + 1e-05;  norm_x_2 = None
            rsqrt_2: "f32[1, 8192, 1]" = torch.rsqrt(add_8);  add_8 = None
            x_normed_3: "f32[1, 8192, 2]" = x_5 * rsqrt_2;  x_5 = rsqrt_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_2: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_6: "f32[2]" = weight_2.float();  weight_2 = None
            mul_14: "f32[1, 8192, 2]" = x_normed_3 * float_6;  x_normed_3 = float_6 = None
            to_5: "bf16[1, 8192, 2]" = mul_14.to(dtype = torch.bfloat16);  mul_14 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:406 in forward, code: x_fc_1 = self.fc_1(x)
            x_fc_1: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, None);  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:407 in forward, code: x_fc_2 = self.fc_2(x)
            x_fc_2: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, None);  to_5 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            gelu: "bf16[1, 8192, 16]" = torch._C._nn.gelu(x_fc_1, approximate = 'tanh');  x_fc_1 = None
            x_6: "bf16[1, 8192, 16]" = gelu * x_fc_2;  gelu = x_fc_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:409 in forward, code: return self.proj(x)
            linear_4: "bf16[1, 8192, 2]" = torch._C._nn.linear(x_6, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, None);  x_6 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_7: "f32[1, 8192, 2]" = linear_4.float();  linear_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_16: "f32[1, 8192, 2]" = x_7 * x_7
            norm_x_3: "f32[1, 8192, 1]" = torch.mean(mul_16, dim = -1, keepdim = True);  mul_16 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_10: "f32[1, 8192, 1]" = norm_x_3 + 1e-05;  norm_x_3 = None
            rsqrt_3: "f32[1, 8192, 1]" = torch.rsqrt(add_10);  add_10 = None
            x_normed_4: "f32[1, 8192, 2]" = x_7 * rsqrt_3;  x_7 = rsqrt_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_3: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_8: "f32[2]" = weight_3.float();  weight_3 = None
            mul_18: "f32[1, 8192, 2]" = x_normed_4 * float_8;  x_normed_4 = float_8 = None
            to_6: "bf16[1, 8192, 2]" = mul_18.to(dtype = torch.bfloat16);  mul_18 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:237 in forward, code: x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
            x_8: "bf16[1, 8192, 2]" = to_6 + x_4;  to_6 = x_4 = None
            return (x_8,)

class GraphModule(torch.nn.Module):
    def forward(self, x_2: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        wrap_body_1 = self.wrap_body_1

         # File: <debug>:0 in <debug>, code: memory_events = [], total 17490241024 bytes allocated
        memory_events_wrap_body_1 = thunder_dev_utils_debug_memory_transform_memory_events_wrap_body_1();  memory_events_wrap_body_1 = None
        memory_events_x_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_2();  memory_events_x_2 = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None
        memory_events_cos = thunder_dev_utils_debug_memory_transform_memory_events_cos();  memory_events_cos = None
        memory_events_sin = thunder_dev_utils_debug_memory_transform_memory_events_sin();  memory_events_sin = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None
        memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = thunder_dev_utils_debug_memory_transform_memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_();  memory_events_l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        tag_activation_checkpoint_1 = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_1, x_2, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, cos, sin, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_, use_reentrant = False);  wrap_body_1 = x_2 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = cos = sin = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

         # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 134217728 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 33554432 bytes, alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 33554432 bytes, alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, free_requested - 8589934592 bytes, free_completed - 8589934592 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes, alloc - 4294967296 bytes, alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, alloc - 32768 bytes, alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 262144 bytes, alloc - 32768 bytes, alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, alloc - 65536 bytes, alloc - 8 bytes, alloc - 65536 bytes, alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes, alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 34940349952 bytes allocated
        memory_events_tag_activation_checkpoint_1 = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint_1();  memory_events_tag_activation_checkpoint_1 = None
        return tag_activation_checkpoint_1

        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

    class wrap_body_1(torch.nn.Module):
        def forward(self, x_2: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x: "f32[1, 8192, 2]" = x_2.float()

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul: "f32[1, 8192, 2]" = x * x
            norm_x: "f32[1, 8192, 1]" = torch.mean(mul, dim = -1, keepdim = True);  mul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add: "f32[1, 8192, 1]" = norm_x + 1e-05;  norm_x = None
            rsqrt: "f32[1, 8192, 1]" = torch.rsqrt(add);  add = None
            x_normed: "f32[1, 8192, 2]" = x * rsqrt;  x = rsqrt = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_2: "f32[2]" = weight.float();  weight = None
            mul_2: "f32[1, 8192, 2]" = x_normed * float_2;  x_normed = float_2 = None
            x_normed_1: "bf16[1, 8192, 2]" = mul_2.to(dtype = torch.bfloat16);  mul_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:269 in forward, code: qkv = self.attn(x)
            qkv: "bf16[1, 8192, 8192]" = torch._C._nn.linear(x_normed_1, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, None);  x_normed_1 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:274 in forward, code: qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv_1: "bf16[1, 8192, 16, 4, 128]" = qkv.view(1, 8192, 16, 4, 128);  qkv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:275 in forward, code: qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
            qkv_2: "bf16[1, 16, 4, 8192, 128]" = qkv_1.permute(0, 2, 3, 1, 4);  qkv_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            split = qkv_2.split((2, 1, 1), dim = 2);  qkv_2 = None
            q: "bf16[1, 16, 2, 8192, 128]" = split[0]
            k: "bf16[1, 16, 1, 8192, 128]" = split[1]
            v: "bf16[1, 16, 1, 8192, 128]" = split[2];  split = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:284 in forward, code: k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            k_1: "bf16[1, 16, 2, 8192, 128]" = k.expand(1, 16, 2, 8192, 128);  k = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:285 in forward, code: v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v_1: "bf16[1, 16, 2, 8192, 128]" = v.expand(1, 16, 2, 8192, 128);  v = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:287 in forward, code: q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            q_1: "bf16[1, 32, 8192, 128]" = q.reshape(1, -1, 8192, 128);  q = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:288 in forward, code: k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            k_2: "bf16[1, 32, 8192, 128]" = k_1.reshape(1, -1, 8192, 128);  k_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:289 in forward, code: v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
            v_2: "bf16[1, 32, 8192, 128]" = v_1.reshape(1, -1, 8192, 128);  v_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:291 in forward, code: q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            getitem_3: "bf16[1, 32, 8192, 128]" = q_1[(Ellipsis, slice(None, 128, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(None, 64, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(64, None, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg: "bf16[1, 32, 8192, 64]" = -x2;  x2 = None
            rotated: "bf16[1, 32, 8192, 128]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_1: "bf16[1, 8192, 128]" = cos.unsqueeze(-3)

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_1: "bf16[1, 8192, 128]" = sin.unsqueeze(-3)

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_3: "bf16[1, 32, 8192, 128]" = getitem_3 * cos_1;  getitem_3 = cos_1 = None
            mul_4: "bf16[1, 32, 8192, 128]" = rotated * sin_1;  rotated = sin_1 = None
            roped: "bf16[1, 32, 8192, 128]" = mul_3 + mul_4;  mul_3 = mul_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            q_roped: "bf16[1, 32, 8192, 128]" = roped.to(dtype = torch.bfloat16);  roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:292 in forward, code: k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            getitem_6: "bf16[1, 32, 8192, 128]" = k_2[(Ellipsis, slice(None, 128, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(None, 64, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(64, None, None))]

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg_1: "bf16[1, 32, 8192, 64]" = -x2_1;  x2_1 = None
            rotated_1: "bf16[1, 32, 8192, 128]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_2: "bf16[1, 8192, 128]" = cos.unsqueeze(-3);  cos = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_2: "bf16[1, 8192, 128]" = sin.unsqueeze(-3);  sin = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_5: "bf16[1, 32, 8192, 128]" = getitem_6 * cos_2;  getitem_6 = cos_2 = None
            mul_6: "bf16[1, 32, 8192, 128]" = rotated_1 * sin_2;  rotated_1 = sin_2 = None
            roped_1: "bf16[1, 32, 8192, 128]" = mul_5 + mul_6;  mul_5 = mul_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            k_roped: "bf16[1, 32, 8192, 128]" = roped_1.to(dtype = torch.bfloat16);  roped_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            getitem_9: "bf16[1, 32, 8192, 0]" = q_1[(Ellipsis, slice(128, None, None))];  q_1 = None
            q_2: "bf16[1, 32, 8192, 128]" = torch.cat((q_roped, getitem_9), dim = -1);  q_roped = getitem_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            getitem_10: "bf16[1, 32, 8192, 0]" = k_2[(Ellipsis, slice(128, None, None))];  k_2 = None
            k_3: "bf16[1, 32, 8192, 128]" = torch.cat((k_roped, getitem_10), dim = -1);  k_roped = getitem_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            getattr_1: "bf16[1, 32, 128, 8192]" = k_3.mT;  k_3 = None
            matmul: "bf16[1, 32, 8192, 8192]" = q_2 @ getattr_1;  q_2 = getattr_1 = None
            scores: "bf16[1, 32, 8192, 8192]" = matmul * 0.08333333333333333;  matmul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            truediv: "bf16[1, 32, 8192, 8192]" = scores / 50.0;  scores = None
            tanh: "bf16[1, 32, 8192, 8192]" = torch.tanh(truediv);  truediv = None
            scores_1: "bf16[1, 32, 8192, 8192]" = tanh * 50.0;  tanh = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:339 in scaled_dot_product_attention, code: mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
            ones: "bf16[8192, 8192]" = torch.ones(8192, 8192, dtype = torch.bfloat16, device = device(type='cuda', index=0))
            mask: "bf16[8192, 8192]" = ones.triu(diagonal = 1);  ones = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:340 in scaled_dot_product_attention, code: mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            bool_1: "b8[8192, 8192]" = mask.bool()
            masked_fill_: "bf16[8192, 8192]" = mask.masked_fill_(bool_1, -3.3895313892515355e+38);  bool_1 = masked_fill_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:341 in scaled_dot_product_attention, code: scores = scores + mask
            scores_2: "bf16[1, 32, 8192, 8192]" = scores_1 + mask;  scores_1 = mask = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            softmax: "f32[1, 32, 8192, 8192]" = torch.nn.functional.softmax(scores_2, dim = -1, dtype = torch.float32);  scores_2 = None
            scores_3: "bf16[1, 32, 8192, 8192]" = softmax.to(dtype = torch.bfloat16);  softmax = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:343 in scaled_dot_product_attention, code: y = scores @ v
            y: "bf16[1, 32, 8192, 128]" = scores_3 @ v_2;  scores_3 = v_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:348 in scaled_dot_product_attention, code: return y.transpose(1, 2)
            y_1: "bf16[1, 8192, 32, 128]" = y.transpose(1, 2);  y = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:321 in forward, code: y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
            y_2: "bf16[1, 8192, 4096]" = y_1.reshape(1, 8192, 4096);  y_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:324 in forward, code: return self.proj(y)
            attention_output: "bf16[1, 8192, 2]" = torch._C._nn.linear(y_2, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, None);  y_2 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_3: "f32[1, 8192, 2]" = attention_output.float();  attention_output = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_9: "f32[1, 8192, 2]" = x_3 * x_3
            norm_x_1: "f32[1, 8192, 1]" = torch.mean(mul_9, dim = -1, keepdim = True);  mul_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_5: "f32[1, 8192, 1]" = norm_x_1 + 1e-05;  norm_x_1 = None
            rsqrt_1: "f32[1, 8192, 1]" = torch.rsqrt(add_5);  add_5 = None
            x_normed_2: "f32[1, 8192, 2]" = x_3 * rsqrt_1;  x_3 = rsqrt_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_1: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_4: "f32[2]" = weight_1.float();  weight_1 = None
            mul_11: "f32[1, 8192, 2]" = x_normed_2 * float_4;  x_normed_2 = float_4 = None
            attention_output_1: "bf16[1, 8192, 2]" = mul_11.to(dtype = torch.bfloat16);  mul_11 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:236 in forward, code: x = attention_output + x
            x_4: "bf16[1, 8192, 2]" = attention_output_1 + x_2;  attention_output_1 = x_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_5: "f32[1, 8192, 2]" = x_4.float()

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_12: "f32[1, 8192, 2]" = x_5 * x_5
            norm_x_2: "f32[1, 8192, 1]" = torch.mean(mul_12, dim = -1, keepdim = True);  mul_12 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_8: "f32[1, 8192, 1]" = norm_x_2 + 1e-05;  norm_x_2 = None
            rsqrt_2: "f32[1, 8192, 1]" = torch.rsqrt(add_8);  add_8 = None
            x_normed_3: "f32[1, 8192, 2]" = x_5 * rsqrt_2;  x_5 = rsqrt_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_2: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_6: "f32[2]" = weight_2.float();  weight_2 = None
            mul_14: "f32[1, 8192, 2]" = x_normed_3 * float_6;  x_normed_3 = float_6 = None
            to_5: "bf16[1, 8192, 2]" = mul_14.to(dtype = torch.bfloat16);  mul_14 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:406 in forward, code: x_fc_1 = self.fc_1(x)
            x_fc_1: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, None);  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:407 in forward, code: x_fc_2 = self.fc_2(x)
            x_fc_2: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, None);  to_5 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            gelu: "bf16[1, 8192, 16]" = torch._C._nn.gelu(x_fc_1, approximate = 'tanh');  x_fc_1 = None
            x_6: "bf16[1, 8192, 16]" = gelu * x_fc_2;  gelu = x_fc_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:409 in forward, code: return self.proj(x)
            linear_4: "bf16[1, 8192, 2]" = torch._C._nn.linear(x_6, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, None);  x_6 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_7: "f32[1, 8192, 2]" = linear_4.float();  linear_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_16: "f32[1, 8192, 2]" = x_7 * x_7
            norm_x_3: "f32[1, 8192, 1]" = torch.mean(mul_16, dim = -1, keepdim = True);  mul_16 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_10: "f32[1, 8192, 1]" = norm_x_3 + 1e-05;  norm_x_3 = None
            rsqrt_3: "f32[1, 8192, 1]" = torch.rsqrt(add_10);  add_10 = None
            x_normed_4: "f32[1, 8192, 2]" = x_7 * rsqrt_3;  x_7 = rsqrt_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_3: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_8: "f32[2]" = weight_3.float();  weight_3 = None
            mul_18: "f32[1, 8192, 2]" = x_normed_4 * float_8;  x_normed_4 = float_8 = None
            to_6: "bf16[1, 8192, 2]" = mul_18.to(dtype = torch.bfloat16);  mul_18 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:237 in forward, code: x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
            x_8: "bf16[1, 8192, 2]" = to_6 + x_4;  to_6 = x_4 = None
            return (x_8,)

