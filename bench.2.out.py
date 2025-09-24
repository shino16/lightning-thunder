Loading model with {'name': 'Gemma-2-27b-like', 'hf_config': {'org': 'google', 'name': 'gemma-2-27b'}, 'scale_embeddings': True, 'attention_scores_scalar': 144, 'block_size': 8192, 'sliding_window_size': 4096, 'sliding_window_layer_placing': 2, 'vocab_size': 256000, 'padding_multiple': 512, 'padded_vocab_size': 256000, 'n_layer': 2, 'n_head': 32, 'head_size': 128, 'n_embd': 2, 'rotary_percentage': 1.0, 'parallel_residual': False, 'bias': False, 'lm_head_bias': False, 'n_query_groups': 16, 'shared_attention_norm': False, 'norm_class_name': 'RMSNorm', 'post_attention_norm': True, 'post_mlp_norm': True, 'norm_eps': 1e-05, 'mlp_class_name': 'GemmaMLP', 'gelu_approximate': 'tanh', 'intermediate_size': 16, 'rope_condense_ratio': 1, 'rope_base': 10000, 'rope_adjustments': None, 'n_expert': 0, 'n_expert_per_token': 0, 'attention_logit_softcapping': 50.0, 'final_logit_softcapping': 30.0, 'rope_n_elem': 128}
Time to instantiate model: 0.06 seconds.
Model Flops/Throughput calculation failed for model Gemma-2-27b-like. Skipping throughput metric collection.
iter 0: loss 13.0000, iter time: 5195.87ms, t: 8192
Model name: Gemma-2-27b-like
Seq Length: 8192
Micro BS: 1
Global BS: 1
Number of Layers: 2
Number of parameters: 0.00B
Distributed Mode: none
Compiler: dynamo_thunder
Low Precision Mode: none
Average iter time: 5195.87 ms
Memory used: 60.58 GB
Saved for backward size: 4001.07 MiB
Saved for backward number of tensors: 6
class GraphModule(torch.nn.Module):
    def forward(self, x_1: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        wrap_body_0 = self.wrap_body_0

         # File: <debug>:0 in <debug>, code: memory_events = [], total 6610432 bytes allocated
        memory_events_wrap_body_0 = thunder_dev_utils_debug_memory_transform_memory_events_wrap_body_0();  memory_events_wrap_body_0 = None
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

         # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 134217728 bytes, free_completed - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 17490273792 bytes allocated
        memory_events_tag_activation_checkpoint = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint();  memory_events_tag_activation_checkpoint = None
        return tag_activation_checkpoint

        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

    class wrap_body_0(torch.nn.Module):
        def forward(self, x_1: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
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

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x: "f32[1, 8192, 2]" = x_1.float()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 6675968 bytes allocated
            memory_events_x = thunder_dev_utils_debug_memory_transform_memory_events_x();  memory_events_x = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul: "f32[1, 8192, 2]" = x * x

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 6741504 bytes allocated
            memory_events_mul = thunder_dev_utils_debug_memory_transform_memory_events_mul();  memory_events_mul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x: "f32[1, 8192, 1]" = torch.mean(mul, dim = -1, keepdim = True);  mul = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 6708736 bytes allocated
            memory_events_norm_x = thunder_dev_utils_debug_memory_transform_memory_events_norm_x();  memory_events_norm_x = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add: "f32[1, 8192, 1]" = norm_x + 1e-05;  norm_x = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 6708736 bytes allocated
            memory_events_add = thunder_dev_utils_debug_memory_transform_memory_events_add();  memory_events_add = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt: "f32[1, 8192, 1]" = torch.rsqrt(add);  add = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 6708736 bytes allocated
            memory_events_rsqrt = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt();  memory_events_rsqrt = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed: "f32[1, 8192, 2]" = x * rsqrt;  x = rsqrt = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 6774272 bytes allocated
            memory_events_x_normed = thunder_dev_utils_debug_memory_transform_memory_events_x_normed();  memory_events_x_normed = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 6774784 bytes allocated
            memory_events_weight = thunder_dev_utils_debug_memory_transform_memory_events_weight();  memory_events_weight = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_2: "f32[2]" = weight.float();  weight = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 6774784 bytes allocated
            memory_events_float_2 = thunder_dev_utils_debug_memory_transform_memory_events_float_2();  memory_events_float_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_2: "f32[1, 8192, 2]" = x_normed * float_2;  x_normed = float_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 6840320 bytes allocated
            memory_events_mul_2 = thunder_dev_utils_debug_memory_transform_memory_events_mul_2();  memory_events_mul_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            x_normed_1: "bf16[1, 8192, 2]" = mul_2.to(dtype = torch.bfloat16);  mul_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 6807552 bytes allocated
            memory_events_x_normed_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_1();  memory_events_x_normed_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:269 in forward, code: qkv = self.attn(x)
            qkv: "bf16[1, 8192, 8192]" = torch._C._nn.linear(x_normed_1, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, None);  x_normed_1 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 134217728 bytes, alloc - 134217728 bytes, segment_alloc - 33554432 bytes, alloc - 33554432 bytes], total 174579712 bytes allocated
            memory_events_qkv = thunder_dev_utils_debug_memory_transform_memory_events_qkv();  memory_events_qkv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:274 in forward, code: qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv_1: "bf16[1, 8192, 16, 4, 128]" = qkv.view(1, 8192, 16, 4, 128);  qkv = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_qkv_1 = thunder_dev_utils_debug_memory_transform_memory_events_qkv_1();  memory_events_qkv_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:275 in forward, code: qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
            qkv_2: "bf16[1, 16, 4, 8192, 128]" = qkv_1.permute(0, 2, 3, 1, 4);  qkv_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_qkv_2 = thunder_dev_utils_debug_memory_transform_memory_events_qkv_2();  memory_events_qkv_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            split = qkv_2.split((2, 1, 1), dim = 2);  qkv_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_split = thunder_dev_utils_debug_memory_transform_memory_events_split();  memory_events_split = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            q: "bf16[1, 16, 2, 8192, 128]" = split[0]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_q = thunder_dev_utils_debug_memory_transform_memory_events_q();  memory_events_q = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            k: "bf16[1, 16, 1, 8192, 128]" = split[1]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_k = thunder_dev_utils_debug_memory_transform_memory_events_k();  memory_events_k = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            v: "bf16[1, 16, 1, 8192, 128]" = split[2];  split = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_v = thunder_dev_utils_debug_memory_transform_memory_events_v();  memory_events_v = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:284 in forward, code: k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            k_1: "bf16[1, 16, 2, 8192, 128]" = k.expand(1, 16, 2, 8192, 128);  k = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_k_1 = thunder_dev_utils_debug_memory_transform_memory_events_k_1();  memory_events_k_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:285 in forward, code: v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v_1: "bf16[1, 16, 2, 8192, 128]" = v.expand(1, 16, 2, 8192, 128);  v = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_v_1 = thunder_dev_utils_debug_memory_transform_memory_events_v_1();  memory_events_v_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:287 in forward, code: q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            q_1: "bf16[1, 32, 8192, 128]" = q.reshape(1, -1, 8192, 128);  q = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes], total 241688576 bytes allocated
            memory_events_q_1 = thunder_dev_utils_debug_memory_transform_memory_events_q_1();  memory_events_q_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:288 in forward, code: k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            k_2: "bf16[1, 32, 8192, 128]" = k_1.reshape(1, -1, 8192, 128);  k_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes], total 308797440 bytes allocated
            memory_events_k_2 = thunder_dev_utils_debug_memory_transform_memory_events_k_2();  memory_events_k_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:289 in forward, code: v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
            v_2: "bf16[1, 32, 8192, 128]" = v_1.reshape(1, -1, 8192, 128);  v_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 241688576 bytes allocated
            memory_events_v_2 = thunder_dev_utils_debug_memory_transform_memory_events_v_2();  memory_events_v_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:291 in forward, code: q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            getitem_3: "bf16[1, 32, 8192, 128]" = q_1[(Ellipsis, slice(None, 128, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 241688576 bytes allocated
            memory_events_getitem_3 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_3();  memory_events_getitem_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(None, 64, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 241688576 bytes allocated
            memory_events_x1 = thunder_dev_utils_debug_memory_transform_memory_events_x1();  memory_events_x1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(64, None, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 241688576 bytes allocated
            memory_events_x2 = thunder_dev_utils_debug_memory_transform_memory_events_x2();  memory_events_x2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg: "bf16[1, 32, 8192, 64]" = -x2;  x2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 33554432 bytes], total 275243008 bytes allocated
            memory_events_neg = thunder_dev_utils_debug_memory_transform_memory_events_neg();  memory_events_neg = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            rotated: "bf16[1, 32, 8192, 128]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes], total 308797440 bytes allocated
            memory_events_rotated = thunder_dev_utils_debug_memory_transform_memory_events_rotated();  memory_events_rotated = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_1: "bf16[1, 8192, 128]" = cos.unsqueeze(-3)

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_cos_1 = thunder_dev_utils_debug_memory_transform_memory_events_cos_1();  memory_events_cos_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_1: "bf16[1, 8192, 128]" = sin.unsqueeze(-3)

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_sin_1 = thunder_dev_utils_debug_memory_transform_memory_events_sin_1();  memory_events_sin_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_3: "bf16[1, 32, 8192, 128]" = getitem_3 * cos_1;  getitem_3 = cos_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes], total 375906304 bytes allocated
            memory_events_mul_3 = thunder_dev_utils_debug_memory_transform_memory_events_mul_3();  memory_events_mul_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_4: "bf16[1, 32, 8192, 128]" = rotated * sin_1;  rotated = sin_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 375906304 bytes allocated
            memory_events_mul_4 = thunder_dev_utils_debug_memory_transform_memory_events_mul_4();  memory_events_mul_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            roped: "bf16[1, 32, 8192, 128]" = mul_3 + mul_4;  mul_3 = mul_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 308797440 bytes allocated
            memory_events_roped = thunder_dev_utils_debug_memory_transform_memory_events_roped();  memory_events_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            q_roped: "bf16[1, 32, 8192, 128]" = roped.to(dtype = torch.bfloat16);  roped = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_q_roped = thunder_dev_utils_debug_memory_transform_memory_events_q_roped();  memory_events_q_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:292 in forward, code: k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            getitem_6: "bf16[1, 32, 8192, 128]" = k_2[(Ellipsis, slice(None, 128, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_getitem_6 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_6();  memory_events_getitem_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(None, 64, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_x1_1 = thunder_dev_utils_debug_memory_transform_memory_events_x1_1();  memory_events_x1_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(64, None, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_x2_1 = thunder_dev_utils_debug_memory_transform_memory_events_x2_1();  memory_events_x2_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg_1: "bf16[1, 32, 8192, 64]" = -x2_1;  x2_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 33554432 bytes], total 342351872 bytes allocated
            memory_events_neg_1 = thunder_dev_utils_debug_memory_transform_memory_events_neg_1();  memory_events_neg_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            rotated_1: "bf16[1, 32, 8192, 128]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes], total 375906304 bytes allocated
            memory_events_rotated_1 = thunder_dev_utils_debug_memory_transform_memory_events_rotated_1();  memory_events_rotated_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_2: "bf16[1, 8192, 128]" = cos.unsqueeze(-3);  cos = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 375906304 bytes allocated
            memory_events_cos_2 = thunder_dev_utils_debug_memory_transform_memory_events_cos_2();  memory_events_cos_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_2: "bf16[1, 8192, 128]" = sin.unsqueeze(-3);  sin = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 375906304 bytes allocated
            memory_events_sin_2 = thunder_dev_utils_debug_memory_transform_memory_events_sin_2();  memory_events_sin_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_5: "bf16[1, 32, 8192, 128]" = getitem_6 * cos_2;  getitem_6 = cos_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 443015168 bytes allocated
            memory_events_mul_5 = thunder_dev_utils_debug_memory_transform_memory_events_mul_5();  memory_events_mul_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_6: "bf16[1, 32, 8192, 128]" = rotated_1 * sin_2;  rotated_1 = sin_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 443015168 bytes allocated
            memory_events_mul_6 = thunder_dev_utils_debug_memory_transform_memory_events_mul_6();  memory_events_mul_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            roped_1: "bf16[1, 32, 8192, 128]" = mul_5 + mul_6;  mul_5 = mul_6 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 375906304 bytes allocated
            memory_events_roped_1 = thunder_dev_utils_debug_memory_transform_memory_events_roped_1();  memory_events_roped_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            k_roped: "bf16[1, 32, 8192, 128]" = roped_1.to(dtype = torch.bfloat16);  roped_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 375906304 bytes allocated
            memory_events_k_roped = thunder_dev_utils_debug_memory_transform_memory_events_k_roped();  memory_events_k_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            getitem_9: "bf16[1, 32, 8192, 0]" = q_1[(Ellipsis, slice(128, None, None))];  q_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 375906304 bytes allocated
            memory_events_getitem_9 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_9();  memory_events_getitem_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            q_2: "bf16[1, 32, 8192, 128]" = torch.cat((q_roped, getitem_9), dim = -1);  q_roped = getitem_9 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 308797440 bytes allocated
            memory_events_q_2 = thunder_dev_utils_debug_memory_transform_memory_events_q_2();  memory_events_q_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            getitem_10: "bf16[1, 32, 8192, 0]" = k_2[(Ellipsis, slice(128, None, None))];  k_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_getitem_10 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_10();  memory_events_getitem_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            k_3: "bf16[1, 32, 8192, 128]" = torch.cat((k_roped, getitem_10), dim = -1);  k_roped = getitem_10 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 241688576 bytes allocated
            memory_events_k_3 = thunder_dev_utils_debug_memory_transform_memory_events_k_3();  memory_events_k_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:313 in forward, code: mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
            ones: "bf16[8192, 8192]" = torch.ones(8192, 8192, dtype = torch.bfloat16, device = device(type='cuda', index=0))

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes], total 375906304 bytes allocated
            memory_events_ones = thunder_dev_utils_debug_memory_transform_memory_events_ones();  memory_events_ones = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:313 in forward, code: mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
            mask: "bf16[8192, 8192]" = ones.triu(diagonal = 1);  ones = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 375906304 bytes allocated
            memory_events_mask = thunder_dev_utils_debug_memory_transform_memory_events_mask();  memory_events_mask = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:314 in forward, code: mask.masked_fill_(mask.bool(), float("-inf"))
            bool_1: "b8[8192, 8192]" = mask.bool()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 443015168 bytes allocated
            memory_events_bool_1 = thunder_dev_utils_debug_memory_transform_memory_events_bool_1();  memory_events_bool_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:314 in forward, code: mask.masked_fill_(mask.bool(), float("-inf"))
            masked_fill_: "bf16[8192, 8192]" = mask.masked_fill_(bool_1, -inf);  bool_1 = masked_fill_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 375906304 bytes allocated
            memory_events_masked_fill_ = thunder_dev_utils_debug_memory_transform_memory_events_masked_fill_();  memory_events_masked_fill_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:315 in forward, code: sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            ones_like: "bf16[8192, 8192]" = torch.ones_like(mask)

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes], total 510124032 bytes allocated
            memory_events_ones_like = thunder_dev_utils_debug_memory_transform_memory_events_ones_like();  memory_events_ones_like = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:315 in forward, code: sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            sliding_window_bias: "bf16[8192, 8192]" = ones_like.tril(diagonal = -4096);  ones_like = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 510124032 bytes allocated
            memory_events_sliding_window_bias = thunder_dev_utils_debug_memory_transform_memory_events_sliding_window_bias();  memory_events_sliding_window_bias = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:316 in forward, code: sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            bool_2: "b8[8192, 8192]" = sliding_window_bias.bool()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 577232896 bytes allocated
            memory_events_bool_2 = thunder_dev_utils_debug_memory_transform_memory_events_bool_2();  memory_events_bool_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:316 in forward, code: sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            masked_fill__1: "bf16[8192, 8192]" = sliding_window_bias.masked_fill_(bool_2, -inf);  bool_2 = masked_fill__1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 510124032 bytes allocated
            memory_events_masked_fill__1 = thunder_dev_utils_debug_memory_transform_memory_events_masked_fill__1();  memory_events_masked_fill__1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:317 in forward, code: mask += sliding_window_bias
            mask += sliding_window_bias;  mask_1: "bf16[8192, 8192]" = mask;  mask = sliding_window_bias = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 510124032 bytes allocated
            memory_events_mask_1 = thunder_dev_utils_debug_memory_transform_memory_events_mask_1();  memory_events_mask_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            getattr_1: "bf16[1, 32, 128, 8192]" = k_3.mT;  k_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 510124032 bytes allocated
            memory_events_getattr_1 = thunder_dev_utils_debug_memory_transform_memory_events_getattr_1();  memory_events_getattr_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            matmul: "bf16[1, 32, 8192, 8192]" = q_2 @ getattr_1;  q_2 = getattr_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes], total 4805091328 bytes allocated
            memory_events_matmul = thunder_dev_utils_debug_memory_transform_memory_events_matmul();  memory_events_matmul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            scores: "bf16[1, 32, 8192, 8192]" = matmul * 0.08333333333333333;  matmul = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 4805091328 bytes allocated
            memory_events_scores = thunder_dev_utils_debug_memory_transform_memory_events_scores();  memory_events_scores = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            truediv: "bf16[1, 32, 8192, 8192]" = scores / 50.0;  scores = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 4805091328 bytes allocated
            memory_events_truediv = thunder_dev_utils_debug_memory_transform_memory_events_truediv();  memory_events_truediv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            tanh: "bf16[1, 32, 8192, 8192]" = torch.tanh(truediv);  truediv = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 4805091328 bytes allocated
            memory_events_tanh = thunder_dev_utils_debug_memory_transform_memory_events_tanh();  memory_events_tanh = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            scores_1: "bf16[1, 32, 8192, 8192]" = tanh * 50.0;  tanh = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 9100058624 bytes allocated
            memory_events_scores_1 = thunder_dev_utils_debug_memory_transform_memory_events_scores_1();  memory_events_scores_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:341 in scaled_dot_product_attention, code: scores = scores + mask
            scores_2: "bf16[1, 32, 8192, 8192]" = scores_1 + mask_1;  scores_1 = mask_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 9100058624 bytes allocated
            memory_events_scores_2 = thunder_dev_utils_debug_memory_transform_memory_events_scores_2();  memory_events_scores_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            softmax: "f32[1, 32, 8192, 8192]" = torch.nn.functional.softmax(scores_2, dim = -1, dtype = torch.float32);  scores_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, free_requested - 8589934592 bytes, free_completed - 8589934592 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 13395025920 bytes allocated
            memory_events_softmax = thunder_dev_utils_debug_memory_transform_memory_events_softmax();  memory_events_softmax = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            scores_3: "bf16[1, 32, 8192, 8192]" = softmax.to(dtype = torch.bfloat16);  softmax = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 17689993216 bytes allocated
            memory_events_scores_3 = thunder_dev_utils_debug_memory_transform_memory_events_scores_3();  memory_events_scores_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:343 in scaled_dot_product_attention, code: y = scores @ v
            y: "bf16[1, 32, 8192, 128]" = scores_3 @ v_2;  scores_3 = v_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17757102080 bytes allocated
            memory_events_y = thunder_dev_utils_debug_memory_transform_memory_events_y();  memory_events_y = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:348 in scaled_dot_product_attention, code: return y.transpose(1, 2)
            y_1: "bf16[1, 8192, 32, 128]" = y.transpose(1, 2);  y = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17757102080 bytes allocated
            memory_events_y_1 = thunder_dev_utils_debug_memory_transform_memory_events_y_1();  memory_events_y_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:321 in forward, code: y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
            y_2: "bf16[1, 8192, 4096]" = y_1.reshape(1, 8192, 4096);  y_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17757102080 bytes allocated
            memory_events_y_2 = thunder_dev_utils_debug_memory_transform_memory_events_y_2();  memory_events_y_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:324 in forward, code: return self.proj(y)
            attention_output: "bf16[1, 8192, 2]" = torch._C._nn.linear(y_2, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, None);  y_2 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes], total 17757134848 bytes allocated
            memory_events_attention_output = thunder_dev_utils_debug_memory_transform_memory_events_attention_output();  memory_events_attention_output = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_2: "f32[1, 8192, 2]" = attention_output.float();  attention_output = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757167616 bytes allocated
            memory_events_x_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_2();  memory_events_x_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_9: "f32[1, 8192, 2]" = x_2 * x_2

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757233152 bytes allocated
            memory_events_mul_9 = thunder_dev_utils_debug_memory_transform_memory_events_mul_9();  memory_events_mul_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_1: "f32[1, 8192, 1]" = torch.mean(mul_9, dim = -1, keepdim = True);  mul_9 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17757200384 bytes allocated
            memory_events_norm_x_1 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_1();  memory_events_norm_x_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_5: "f32[1, 8192, 1]" = norm_x_1 + 1e-05;  norm_x_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757200384 bytes allocated
            memory_events_add_5 = thunder_dev_utils_debug_memory_transform_memory_events_add_5();  memory_events_add_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_1: "f32[1, 8192, 1]" = torch.rsqrt(add_5);  add_5 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757200384 bytes allocated
            memory_events_rsqrt_1 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_1();  memory_events_rsqrt_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_2: "f32[1, 8192, 2]" = x_2 * rsqrt_1;  x_2 = rsqrt_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757265920 bytes allocated
            memory_events_x_normed_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_2();  memory_events_x_normed_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_1: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17757266432 bytes allocated
            memory_events_weight_1 = thunder_dev_utils_debug_memory_transform_memory_events_weight_1();  memory_events_weight_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_4: "f32[2]" = weight_1.float();  weight_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 17757266432 bytes allocated
            memory_events_float_4 = thunder_dev_utils_debug_memory_transform_memory_events_float_4();  memory_events_float_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_11: "f32[1, 8192, 2]" = x_normed_2 * float_4;  x_normed_2 = float_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757331968 bytes allocated
            memory_events_mul_11 = thunder_dev_utils_debug_memory_transform_memory_events_mul_11();  memory_events_mul_11 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            attention_output_1: "bf16[1, 8192, 2]" = mul_11.to(dtype = torch.bfloat16);  mul_11 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17757299200 bytes allocated
            memory_events_attention_output_1 = thunder_dev_utils_debug_memory_transform_memory_events_attention_output_1();  memory_events_attention_output_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:236 in forward, code: x = attention_output + x
            x_3: "bf16[1, 8192, 2]" = attention_output_1 + x_1;  attention_output_1 = x_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757299200 bytes allocated
            memory_events_x_3 = thunder_dev_utils_debug_memory_transform_memory_events_x_3();  memory_events_x_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_4: "f32[1, 8192, 2]" = x_3.float()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757364736 bytes allocated
            memory_events_x_4 = thunder_dev_utils_debug_memory_transform_memory_events_x_4();  memory_events_x_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_12: "f32[1, 8192, 2]" = x_4 * x_4

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757430272 bytes allocated
            memory_events_mul_12 = thunder_dev_utils_debug_memory_transform_memory_events_mul_12();  memory_events_mul_12 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_2: "f32[1, 8192, 1]" = torch.mean(mul_12, dim = -1, keepdim = True);  mul_12 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17757397504 bytes allocated
            memory_events_norm_x_2 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_2();  memory_events_norm_x_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_8: "f32[1, 8192, 1]" = norm_x_2 + 1e-05;  norm_x_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757397504 bytes allocated
            memory_events_add_8 = thunder_dev_utils_debug_memory_transform_memory_events_add_8();  memory_events_add_8 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_2: "f32[1, 8192, 1]" = torch.rsqrt(add_8);  add_8 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757397504 bytes allocated
            memory_events_rsqrt_2 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_2();  memory_events_rsqrt_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_3: "f32[1, 8192, 2]" = x_4 * rsqrt_2;  x_4 = rsqrt_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757463040 bytes allocated
            memory_events_x_normed_3 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_3();  memory_events_x_normed_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_2: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17757463552 bytes allocated
            memory_events_weight_2 = thunder_dev_utils_debug_memory_transform_memory_events_weight_2();  memory_events_weight_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_6: "f32[2]" = weight_2.float();  weight_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 17757463552 bytes allocated
            memory_events_float_6 = thunder_dev_utils_debug_memory_transform_memory_events_float_6();  memory_events_float_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_14: "f32[1, 8192, 2]" = x_normed_3 * float_6;  x_normed_3 = float_6 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757529088 bytes allocated
            memory_events_mul_14 = thunder_dev_utils_debug_memory_transform_memory_events_mul_14();  memory_events_mul_14 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            to_5: "bf16[1, 8192, 2]" = mul_14.to(dtype = torch.bfloat16);  mul_14 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17757496320 bytes allocated
            memory_events_to_5 = thunder_dev_utils_debug_memory_transform_memory_events_to_5();  memory_events_to_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:406 in forward, code: x_fc_1 = self.fc_1(x)
            x_fc_1: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, None);  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 17757758464 bytes allocated
            memory_events_x_fc_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_fc_1();  memory_events_x_fc_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:407 in forward, code: x_fc_2 = self.fc_2(x)
            x_fc_2: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, None);  to_5 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 17758020608 bytes allocated
            memory_events_x_fc_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_fc_2();  memory_events_x_fc_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            gelu: "bf16[1, 8192, 16]" = torch._C._nn.gelu(x_fc_1, approximate = 'tanh');  x_fc_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 17758282752 bytes allocated
            memory_events_gelu = thunder_dev_utils_debug_memory_transform_memory_events_gelu();  memory_events_gelu = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            x_5: "bf16[1, 8192, 16]" = gelu * x_fc_2;  gelu = x_fc_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 17758544896 bytes allocated
            memory_events_x_5 = thunder_dev_utils_debug_memory_transform_memory_events_x_5();  memory_events_x_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:409 in forward, code: return self.proj(x)
            linear_4: "bf16[1, 8192, 2]" = torch._C._nn.linear(x_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, None);  x_5 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes], total 17758577664 bytes allocated
            memory_events_linear_4 = thunder_dev_utils_debug_memory_transform_memory_events_linear_4();  memory_events_linear_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_6: "f32[1, 8192, 2]" = linear_4.float();  linear_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17758610432 bytes allocated
            memory_events_x_6 = thunder_dev_utils_debug_memory_transform_memory_events_x_6();  memory_events_x_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_16: "f32[1, 8192, 2]" = x_6 * x_6

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17758675968 bytes allocated
            memory_events_mul_16 = thunder_dev_utils_debug_memory_transform_memory_events_mul_16();  memory_events_mul_16 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_3: "f32[1, 8192, 1]" = torch.mean(mul_16, dim = -1, keepdim = True);  mul_16 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17758643200 bytes allocated
            memory_events_norm_x_3 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_3();  memory_events_norm_x_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_10: "f32[1, 8192, 1]" = norm_x_3 + 1e-05;  norm_x_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17758643200 bytes allocated
            memory_events_add_10 = thunder_dev_utils_debug_memory_transform_memory_events_add_10();  memory_events_add_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_3: "f32[1, 8192, 1]" = torch.rsqrt(add_10);  add_10 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17758643200 bytes allocated
            memory_events_rsqrt_3 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_3();  memory_events_rsqrt_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_4: "f32[1, 8192, 2]" = x_6 * rsqrt_3;  x_6 = rsqrt_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 2097152 bytes, alloc - 65536 bytes], total 17758708736 bytes allocated
            memory_events_x_normed_4 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_4();  memory_events_x_normed_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_3: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758709248 bytes allocated
            memory_events_weight_3 = thunder_dev_utils_debug_memory_transform_memory_events_weight_3();  memory_events_weight_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_8: "f32[2]" = weight_3.float();  weight_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 17758709248 bytes allocated
            memory_events_float_8 = thunder_dev_utils_debug_memory_transform_memory_events_float_8();  memory_events_float_8 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_18: "f32[1, 8192, 2]" = x_normed_4 * float_8;  x_normed_4 = float_8 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17758774784 bytes allocated
            memory_events_mul_18 = thunder_dev_utils_debug_memory_transform_memory_events_mul_18();  memory_events_mul_18 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            to_6: "bf16[1, 8192, 2]" = mul_18.to(dtype = torch.bfloat16);  mul_18 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17758742016 bytes allocated
            memory_events_to_6 = thunder_dev_utils_debug_memory_transform_memory_events_to_6();  memory_events_to_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:237 in forward, code: x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
            x_7: "bf16[1, 8192, 2]" = to_6 + x_3;  to_6 = x_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17758709248 bytes allocated
            memory_events_x_7 = thunder_dev_utils_debug_memory_transform_memory_events_x_7();  memory_events_x_7 = None
            return (x_7,)

            # No stacktrace found for following nodes
            memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

class GraphModule(torch.nn.Module):
    def forward(self, x_1: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
         # File: /usr/local/lib/python3.12/dist-packages/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py:171 in forward, code: return self.checkpoint_fn(  # type: ignore[misc]
        wrap_body_0 = self.wrap_body_0

         # File: <debug>:0 in <debug>, code: memory_events = [], total 6610432 bytes allocated
        memory_events_wrap_body_0 = thunder_dev_utils_debug_memory_transform_memory_events_wrap_body_0();  memory_events_wrap_body_0 = None
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

         # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 134217728 bytes, free_completed - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 17490273792 bytes allocated
        memory_events_tag_activation_checkpoint = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint();  memory_events_tag_activation_checkpoint = None
        return tag_activation_checkpoint

        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

    class wrap_body_0(torch.nn.Module):
        def forward(self, x_1: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
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

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x: "f32[1, 8192, 2]" = x_1.float()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 6675968 bytes allocated
            memory_events_x = thunder_dev_utils_debug_memory_transform_memory_events_x();  memory_events_x = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul: "f32[1, 8192, 2]" = x * x

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 6741504 bytes allocated
            memory_events_mul = thunder_dev_utils_debug_memory_transform_memory_events_mul();  memory_events_mul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x: "f32[1, 8192, 1]" = torch.mean(mul, dim = -1, keepdim = True);  mul = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 6708736 bytes allocated
            memory_events_norm_x = thunder_dev_utils_debug_memory_transform_memory_events_norm_x();  memory_events_norm_x = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add: "f32[1, 8192, 1]" = norm_x + 1e-05;  norm_x = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 6708736 bytes allocated
            memory_events_add = thunder_dev_utils_debug_memory_transform_memory_events_add();  memory_events_add = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt: "f32[1, 8192, 1]" = torch.rsqrt(add);  add = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 6708736 bytes allocated
            memory_events_rsqrt = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt();  memory_events_rsqrt = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed: "f32[1, 8192, 2]" = x * rsqrt;  x = rsqrt = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 6774272 bytes allocated
            memory_events_x_normed = thunder_dev_utils_debug_memory_transform_memory_events_x_normed();  memory_events_x_normed = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 6774784 bytes allocated
            memory_events_weight = thunder_dev_utils_debug_memory_transform_memory_events_weight();  memory_events_weight = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_2: "f32[2]" = weight.float();  weight = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 6774784 bytes allocated
            memory_events_float_2 = thunder_dev_utils_debug_memory_transform_memory_events_float_2();  memory_events_float_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_2: "f32[1, 8192, 2]" = x_normed * float_2;  x_normed = float_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 6840320 bytes allocated
            memory_events_mul_2 = thunder_dev_utils_debug_memory_transform_memory_events_mul_2();  memory_events_mul_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            x_normed_1: "bf16[1, 8192, 2]" = mul_2.to(dtype = torch.bfloat16);  mul_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 6807552 bytes allocated
            memory_events_x_normed_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_1();  memory_events_x_normed_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:269 in forward, code: qkv = self.attn(x)
            qkv: "bf16[1, 8192, 8192]" = torch._C._nn.linear(x_normed_1, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, None);  x_normed_1 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 134217728 bytes, alloc - 134217728 bytes, segment_alloc - 33554432 bytes, alloc - 33554432 bytes], total 174579712 bytes allocated
            memory_events_qkv = thunder_dev_utils_debug_memory_transform_memory_events_qkv();  memory_events_qkv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:274 in forward, code: qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv_1: "bf16[1, 8192, 16, 4, 128]" = qkv.view(1, 8192, 16, 4, 128);  qkv = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_qkv_1 = thunder_dev_utils_debug_memory_transform_memory_events_qkv_1();  memory_events_qkv_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:275 in forward, code: qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
            qkv_2: "bf16[1, 16, 4, 8192, 128]" = qkv_1.permute(0, 2, 3, 1, 4);  qkv_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_qkv_2 = thunder_dev_utils_debug_memory_transform_memory_events_qkv_2();  memory_events_qkv_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            split = qkv_2.split((2, 1, 1), dim = 2);  qkv_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_split = thunder_dev_utils_debug_memory_transform_memory_events_split();  memory_events_split = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            q: "bf16[1, 16, 2, 8192, 128]" = split[0]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_q = thunder_dev_utils_debug_memory_transform_memory_events_q();  memory_events_q = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            k: "bf16[1, 16, 1, 8192, 128]" = split[1]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_k = thunder_dev_utils_debug_memory_transform_memory_events_k();  memory_events_k = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            v: "bf16[1, 16, 1, 8192, 128]" = split[2];  split = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_v = thunder_dev_utils_debug_memory_transform_memory_events_v();  memory_events_v = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:284 in forward, code: k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            k_1: "bf16[1, 16, 2, 8192, 128]" = k.expand(1, 16, 2, 8192, 128);  k = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_k_1 = thunder_dev_utils_debug_memory_transform_memory_events_k_1();  memory_events_k_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:285 in forward, code: v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v_1: "bf16[1, 16, 2, 8192, 128]" = v.expand(1, 16, 2, 8192, 128);  v = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 174579712 bytes allocated
            memory_events_v_1 = thunder_dev_utils_debug_memory_transform_memory_events_v_1();  memory_events_v_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:287 in forward, code: q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            q_1: "bf16[1, 32, 8192, 128]" = q.reshape(1, -1, 8192, 128);  q = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes], total 241688576 bytes allocated
            memory_events_q_1 = thunder_dev_utils_debug_memory_transform_memory_events_q_1();  memory_events_q_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:288 in forward, code: k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            k_2: "bf16[1, 32, 8192, 128]" = k_1.reshape(1, -1, 8192, 128);  k_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes], total 308797440 bytes allocated
            memory_events_k_2 = thunder_dev_utils_debug_memory_transform_memory_events_k_2();  memory_events_k_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:289 in forward, code: v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
            v_2: "bf16[1, 32, 8192, 128]" = v_1.reshape(1, -1, 8192, 128);  v_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 241688576 bytes allocated
            memory_events_v_2 = thunder_dev_utils_debug_memory_transform_memory_events_v_2();  memory_events_v_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:291 in forward, code: q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            getitem_3: "bf16[1, 32, 8192, 128]" = q_1[(Ellipsis, slice(None, 128, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 241688576 bytes allocated
            memory_events_getitem_3 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_3();  memory_events_getitem_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(None, 64, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 241688576 bytes allocated
            memory_events_x1 = thunder_dev_utils_debug_memory_transform_memory_events_x1();  memory_events_x1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(64, None, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 241688576 bytes allocated
            memory_events_x2 = thunder_dev_utils_debug_memory_transform_memory_events_x2();  memory_events_x2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg: "bf16[1, 32, 8192, 64]" = -x2;  x2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 33554432 bytes], total 275243008 bytes allocated
            memory_events_neg = thunder_dev_utils_debug_memory_transform_memory_events_neg();  memory_events_neg = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            rotated: "bf16[1, 32, 8192, 128]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes], total 308797440 bytes allocated
            memory_events_rotated = thunder_dev_utils_debug_memory_transform_memory_events_rotated();  memory_events_rotated = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_1: "bf16[1, 8192, 128]" = cos.unsqueeze(-3)

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_cos_1 = thunder_dev_utils_debug_memory_transform_memory_events_cos_1();  memory_events_cos_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_1: "bf16[1, 8192, 128]" = sin.unsqueeze(-3)

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_sin_1 = thunder_dev_utils_debug_memory_transform_memory_events_sin_1();  memory_events_sin_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_3: "bf16[1, 32, 8192, 128]" = getitem_3 * cos_1;  getitem_3 = cos_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes], total 375906304 bytes allocated
            memory_events_mul_3 = thunder_dev_utils_debug_memory_transform_memory_events_mul_3();  memory_events_mul_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_4: "bf16[1, 32, 8192, 128]" = rotated * sin_1;  rotated = sin_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 67108864 bytes, alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 375906304 bytes allocated
            memory_events_mul_4 = thunder_dev_utils_debug_memory_transform_memory_events_mul_4();  memory_events_mul_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            roped: "bf16[1, 32, 8192, 128]" = mul_3 + mul_4;  mul_3 = mul_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 308797440 bytes allocated
            memory_events_roped = thunder_dev_utils_debug_memory_transform_memory_events_roped();  memory_events_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            q_roped: "bf16[1, 32, 8192, 128]" = roped.to(dtype = torch.bfloat16);  roped = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_q_roped = thunder_dev_utils_debug_memory_transform_memory_events_q_roped();  memory_events_q_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:292 in forward, code: k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            getitem_6: "bf16[1, 32, 8192, 128]" = k_2[(Ellipsis, slice(None, 128, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_getitem_6 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_6();  memory_events_getitem_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(None, 64, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_x1_1 = thunder_dev_utils_debug_memory_transform_memory_events_x1_1();  memory_events_x1_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(64, None, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_x2_1 = thunder_dev_utils_debug_memory_transform_memory_events_x2_1();  memory_events_x2_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg_1: "bf16[1, 32, 8192, 64]" = -x2_1;  x2_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 33554432 bytes], total 342351872 bytes allocated
            memory_events_neg_1 = thunder_dev_utils_debug_memory_transform_memory_events_neg_1();  memory_events_neg_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            rotated_1: "bf16[1, 32, 8192, 128]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes], total 375906304 bytes allocated
            memory_events_rotated_1 = thunder_dev_utils_debug_memory_transform_memory_events_rotated_1();  memory_events_rotated_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_2: "bf16[1, 8192, 128]" = cos.unsqueeze(-3);  cos = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 375906304 bytes allocated
            memory_events_cos_2 = thunder_dev_utils_debug_memory_transform_memory_events_cos_2();  memory_events_cos_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_2: "bf16[1, 8192, 128]" = sin.unsqueeze(-3);  sin = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 375906304 bytes allocated
            memory_events_sin_2 = thunder_dev_utils_debug_memory_transform_memory_events_sin_2();  memory_events_sin_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_5: "bf16[1, 32, 8192, 128]" = getitem_6 * cos_2;  getitem_6 = cos_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 443015168 bytes allocated
            memory_events_mul_5 = thunder_dev_utils_debug_memory_transform_memory_events_mul_5();  memory_events_mul_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_6: "bf16[1, 32, 8192, 128]" = rotated_1 * sin_2;  rotated_1 = sin_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 443015168 bytes allocated
            memory_events_mul_6 = thunder_dev_utils_debug_memory_transform_memory_events_mul_6();  memory_events_mul_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            roped_1: "bf16[1, 32, 8192, 128]" = mul_5 + mul_6;  mul_5 = mul_6 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 375906304 bytes allocated
            memory_events_roped_1 = thunder_dev_utils_debug_memory_transform_memory_events_roped_1();  memory_events_roped_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            k_roped: "bf16[1, 32, 8192, 128]" = roped_1.to(dtype = torch.bfloat16);  roped_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 375906304 bytes allocated
            memory_events_k_roped = thunder_dev_utils_debug_memory_transform_memory_events_k_roped();  memory_events_k_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            getitem_9: "bf16[1, 32, 8192, 0]" = q_1[(Ellipsis, slice(128, None, None))];  q_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 375906304 bytes allocated
            memory_events_getitem_9 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_9();  memory_events_getitem_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            q_2: "bf16[1, 32, 8192, 128]" = torch.cat((q_roped, getitem_9), dim = -1);  q_roped = getitem_9 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 308797440 bytes allocated
            memory_events_q_2 = thunder_dev_utils_debug_memory_transform_memory_events_q_2();  memory_events_q_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            getitem_10: "bf16[1, 32, 8192, 0]" = k_2[(Ellipsis, slice(128, None, None))];  k_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 308797440 bytes allocated
            memory_events_getitem_10 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_10();  memory_events_getitem_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            k_3: "bf16[1, 32, 8192, 128]" = torch.cat((k_roped, getitem_10), dim = -1);  k_roped = getitem_10 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 241688576 bytes allocated
            memory_events_k_3 = thunder_dev_utils_debug_memory_transform_memory_events_k_3();  memory_events_k_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:313 in forward, code: mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
            ones: "bf16[8192, 8192]" = torch.ones(8192, 8192, dtype = torch.bfloat16, device = device(type='cuda', index=0))

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes], total 375906304 bytes allocated
            memory_events_ones = thunder_dev_utils_debug_memory_transform_memory_events_ones();  memory_events_ones = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:313 in forward, code: mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
            mask: "bf16[8192, 8192]" = ones.triu(diagonal = 1);  ones = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 375906304 bytes allocated
            memory_events_mask = thunder_dev_utils_debug_memory_transform_memory_events_mask();  memory_events_mask = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:314 in forward, code: mask.masked_fill_(mask.bool(), float("-inf"))
            bool_1: "b8[8192, 8192]" = mask.bool()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 443015168 bytes allocated
            memory_events_bool_1 = thunder_dev_utils_debug_memory_transform_memory_events_bool_1();  memory_events_bool_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:314 in forward, code: mask.masked_fill_(mask.bool(), float("-inf"))
            masked_fill_: "bf16[8192, 8192]" = mask.masked_fill_(bool_1, -inf);  bool_1 = masked_fill_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 375906304 bytes allocated
            memory_events_masked_fill_ = thunder_dev_utils_debug_memory_transform_memory_events_masked_fill_();  memory_events_masked_fill_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:315 in forward, code: sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            ones_like: "bf16[8192, 8192]" = torch.ones_like(mask)

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes], total 510124032 bytes allocated
            memory_events_ones_like = thunder_dev_utils_debug_memory_transform_memory_events_ones_like();  memory_events_ones_like = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:315 in forward, code: sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            sliding_window_bias: "bf16[8192, 8192]" = ones_like.tril(diagonal = -4096);  ones_like = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 134217728 bytes, alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 510124032 bytes allocated
            memory_events_sliding_window_bias = thunder_dev_utils_debug_memory_transform_memory_events_sliding_window_bias();  memory_events_sliding_window_bias = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:316 in forward, code: sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            bool_2: "b8[8192, 8192]" = sliding_window_bias.bool()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 577232896 bytes allocated
            memory_events_bool_2 = thunder_dev_utils_debug_memory_transform_memory_events_bool_2();  memory_events_bool_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:316 in forward, code: sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            masked_fill__1: "bf16[8192, 8192]" = sliding_window_bias.masked_fill_(bool_2, -inf);  bool_2 = masked_fill__1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 510124032 bytes allocated
            memory_events_masked_fill__1 = thunder_dev_utils_debug_memory_transform_memory_events_masked_fill__1();  memory_events_masked_fill__1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:317 in forward, code: mask += sliding_window_bias
            mask += sliding_window_bias;  mask_1: "bf16[8192, 8192]" = mask;  mask = sliding_window_bias = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 510124032 bytes allocated
            memory_events_mask_1 = thunder_dev_utils_debug_memory_transform_memory_events_mask_1();  memory_events_mask_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            getattr_1: "bf16[1, 32, 128, 8192]" = k_3.mT;  k_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 510124032 bytes allocated
            memory_events_getattr_1 = thunder_dev_utils_debug_memory_transform_memory_events_getattr_1();  memory_events_getattr_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            matmul: "bf16[1, 32, 8192, 8192]" = q_2 @ getattr_1;  q_2 = getattr_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes], total 4805091328 bytes allocated
            memory_events_matmul = thunder_dev_utils_debug_memory_transform_memory_events_matmul();  memory_events_matmul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            scores: "bf16[1, 32, 8192, 8192]" = matmul * 0.08333333333333333;  matmul = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 4805091328 bytes allocated
            memory_events_scores = thunder_dev_utils_debug_memory_transform_memory_events_scores();  memory_events_scores = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            truediv: "bf16[1, 32, 8192, 8192]" = scores / 50.0;  scores = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 4805091328 bytes allocated
            memory_events_truediv = thunder_dev_utils_debug_memory_transform_memory_events_truediv();  memory_events_truediv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            tanh: "bf16[1, 32, 8192, 8192]" = torch.tanh(truediv);  truediv = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 4805091328 bytes allocated
            memory_events_tanh = thunder_dev_utils_debug_memory_transform_memory_events_tanh();  memory_events_tanh = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            scores_1: "bf16[1, 32, 8192, 8192]" = tanh * 50.0;  tanh = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 9100058624 bytes allocated
            memory_events_scores_1 = thunder_dev_utils_debug_memory_transform_memory_events_scores_1();  memory_events_scores_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:341 in scaled_dot_product_attention, code: scores = scores + mask
            scores_2: "bf16[1, 32, 8192, 8192]" = scores_1 + mask_1;  scores_1 = mask_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 9100058624 bytes allocated
            memory_events_scores_2 = thunder_dev_utils_debug_memory_transform_memory_events_scores_2();  memory_events_scores_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            softmax: "f32[1, 32, 8192, 8192]" = torch.nn.functional.softmax(scores_2, dim = -1, dtype = torch.float32);  scores_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, free_requested - 8589934592 bytes, free_completed - 8589934592 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 13395025920 bytes allocated
            memory_events_softmax = thunder_dev_utils_debug_memory_transform_memory_events_softmax();  memory_events_softmax = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            scores_3: "bf16[1, 32, 8192, 8192]" = softmax.to(dtype = torch.bfloat16);  softmax = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 17689993216 bytes allocated
            memory_events_scores_3 = thunder_dev_utils_debug_memory_transform_memory_events_scores_3();  memory_events_scores_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:343 in scaled_dot_product_attention, code: y = scores @ v
            y: "bf16[1, 32, 8192, 128]" = scores_3 @ v_2;  scores_3 = v_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17757102080 bytes allocated
            memory_events_y = thunder_dev_utils_debug_memory_transform_memory_events_y();  memory_events_y = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:348 in scaled_dot_product_attention, code: return y.transpose(1, 2)
            y_1: "bf16[1, 8192, 32, 128]" = y.transpose(1, 2);  y = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17757102080 bytes allocated
            memory_events_y_1 = thunder_dev_utils_debug_memory_transform_memory_events_y_1();  memory_events_y_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:321 in forward, code: y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
            y_2: "bf16[1, 8192, 4096]" = y_1.reshape(1, 8192, 4096);  y_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17757102080 bytes allocated
            memory_events_y_2 = thunder_dev_utils_debug_memory_transform_memory_events_y_2();  memory_events_y_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:324 in forward, code: return self.proj(y)
            attention_output: "bf16[1, 8192, 2]" = torch._C._nn.linear(y_2, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, None);  y_2 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes], total 17757134848 bytes allocated
            memory_events_attention_output = thunder_dev_utils_debug_memory_transform_memory_events_attention_output();  memory_events_attention_output = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_2: "f32[1, 8192, 2]" = attention_output.float();  attention_output = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757167616 bytes allocated
            memory_events_x_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_2();  memory_events_x_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_9: "f32[1, 8192, 2]" = x_2 * x_2

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757233152 bytes allocated
            memory_events_mul_9 = thunder_dev_utils_debug_memory_transform_memory_events_mul_9();  memory_events_mul_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_1: "f32[1, 8192, 1]" = torch.mean(mul_9, dim = -1, keepdim = True);  mul_9 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17757200384 bytes allocated
            memory_events_norm_x_1 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_1();  memory_events_norm_x_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_5: "f32[1, 8192, 1]" = norm_x_1 + 1e-05;  norm_x_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757200384 bytes allocated
            memory_events_add_5 = thunder_dev_utils_debug_memory_transform_memory_events_add_5();  memory_events_add_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_1: "f32[1, 8192, 1]" = torch.rsqrt(add_5);  add_5 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757200384 bytes allocated
            memory_events_rsqrt_1 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_1();  memory_events_rsqrt_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_2: "f32[1, 8192, 2]" = x_2 * rsqrt_1;  x_2 = rsqrt_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757265920 bytes allocated
            memory_events_x_normed_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_2();  memory_events_x_normed_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_1: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17757266432 bytes allocated
            memory_events_weight_1 = thunder_dev_utils_debug_memory_transform_memory_events_weight_1();  memory_events_weight_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_4: "f32[2]" = weight_1.float();  weight_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 17757266432 bytes allocated
            memory_events_float_4 = thunder_dev_utils_debug_memory_transform_memory_events_float_4();  memory_events_float_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_11: "f32[1, 8192, 2]" = x_normed_2 * float_4;  x_normed_2 = float_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757331968 bytes allocated
            memory_events_mul_11 = thunder_dev_utils_debug_memory_transform_memory_events_mul_11();  memory_events_mul_11 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            attention_output_1: "bf16[1, 8192, 2]" = mul_11.to(dtype = torch.bfloat16);  mul_11 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17757299200 bytes allocated
            memory_events_attention_output_1 = thunder_dev_utils_debug_memory_transform_memory_events_attention_output_1();  memory_events_attention_output_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:236 in forward, code: x = attention_output + x
            x_3: "bf16[1, 8192, 2]" = attention_output_1 + x_1;  attention_output_1 = x_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757299200 bytes allocated
            memory_events_x_3 = thunder_dev_utils_debug_memory_transform_memory_events_x_3();  memory_events_x_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_4: "f32[1, 8192, 2]" = x_3.float()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757364736 bytes allocated
            memory_events_x_4 = thunder_dev_utils_debug_memory_transform_memory_events_x_4();  memory_events_x_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_12: "f32[1, 8192, 2]" = x_4 * x_4

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757430272 bytes allocated
            memory_events_mul_12 = thunder_dev_utils_debug_memory_transform_memory_events_mul_12();  memory_events_mul_12 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_2: "f32[1, 8192, 1]" = torch.mean(mul_12, dim = -1, keepdim = True);  mul_12 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17757397504 bytes allocated
            memory_events_norm_x_2 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_2();  memory_events_norm_x_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_8: "f32[1, 8192, 1]" = norm_x_2 + 1e-05;  norm_x_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757397504 bytes allocated
            memory_events_add_8 = thunder_dev_utils_debug_memory_transform_memory_events_add_8();  memory_events_add_8 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_2: "f32[1, 8192, 1]" = torch.rsqrt(add_8);  add_8 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17757397504 bytes allocated
            memory_events_rsqrt_2 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_2();  memory_events_rsqrt_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_3: "f32[1, 8192, 2]" = x_4 * rsqrt_2;  x_4 = rsqrt_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757463040 bytes allocated
            memory_events_x_normed_3 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_3();  memory_events_x_normed_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_2: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17757463552 bytes allocated
            memory_events_weight_2 = thunder_dev_utils_debug_memory_transform_memory_events_weight_2();  memory_events_weight_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_6: "f32[2]" = weight_2.float();  weight_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 17757463552 bytes allocated
            memory_events_float_6 = thunder_dev_utils_debug_memory_transform_memory_events_float_6();  memory_events_float_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_14: "f32[1, 8192, 2]" = x_normed_3 * float_6;  x_normed_3 = float_6 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17757529088 bytes allocated
            memory_events_mul_14 = thunder_dev_utils_debug_memory_transform_memory_events_mul_14();  memory_events_mul_14 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            to_5: "bf16[1, 8192, 2]" = mul_14.to(dtype = torch.bfloat16);  mul_14 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17757496320 bytes allocated
            memory_events_to_5 = thunder_dev_utils_debug_memory_transform_memory_events_to_5();  memory_events_to_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:406 in forward, code: x_fc_1 = self.fc_1(x)
            x_fc_1: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, None);  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 17757758464 bytes allocated
            memory_events_x_fc_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_fc_1();  memory_events_x_fc_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:407 in forward, code: x_fc_2 = self.fc_2(x)
            x_fc_2: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, None);  to_5 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 17758020608 bytes allocated
            memory_events_x_fc_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_fc_2();  memory_events_x_fc_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            gelu: "bf16[1, 8192, 16]" = torch._C._nn.gelu(x_fc_1, approximate = 'tanh');  x_fc_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 17758282752 bytes allocated
            memory_events_gelu = thunder_dev_utils_debug_memory_transform_memory_events_gelu();  memory_events_gelu = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            x_5: "bf16[1, 8192, 16]" = gelu * x_fc_2;  gelu = x_fc_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 17758544896 bytes allocated
            memory_events_x_5 = thunder_dev_utils_debug_memory_transform_memory_events_x_5();  memory_events_x_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:409 in forward, code: return self.proj(x)
            linear_4: "bf16[1, 8192, 2]" = torch._C._nn.linear(x_5, l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, None);  x_5 = l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes], total 17758577664 bytes allocated
            memory_events_linear_4 = thunder_dev_utils_debug_memory_transform_memory_events_linear_4();  memory_events_linear_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_6: "f32[1, 8192, 2]" = linear_4.float();  linear_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17758610432 bytes allocated
            memory_events_x_6 = thunder_dev_utils_debug_memory_transform_memory_events_x_6();  memory_events_x_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_16: "f32[1, 8192, 2]" = x_6 * x_6

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17758675968 bytes allocated
            memory_events_mul_16 = thunder_dev_utils_debug_memory_transform_memory_events_mul_16();  memory_events_mul_16 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_3: "f32[1, 8192, 1]" = torch.mean(mul_16, dim = -1, keepdim = True);  mul_16 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17758643200 bytes allocated
            memory_events_norm_x_3 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_3();  memory_events_norm_x_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_10: "f32[1, 8192, 1]" = norm_x_3 + 1e-05;  norm_x_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17758643200 bytes allocated
            memory_events_add_10 = thunder_dev_utils_debug_memory_transform_memory_events_add_10();  memory_events_add_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_3: "f32[1, 8192, 1]" = torch.rsqrt(add_10);  add_10 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17758643200 bytes allocated
            memory_events_rsqrt_3 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_3();  memory_events_rsqrt_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_4: "f32[1, 8192, 2]" = x_6 * rsqrt_3;  x_6 = rsqrt_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 2097152 bytes, alloc - 65536 bytes], total 17758708736 bytes allocated
            memory_events_x_normed_4 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_4();  memory_events_x_normed_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_3: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_0_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758709248 bytes allocated
            memory_events_weight_3 = thunder_dev_utils_debug_memory_transform_memory_events_weight_3();  memory_events_weight_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_8: "f32[2]" = weight_3.float();  weight_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 17758709248 bytes allocated
            memory_events_float_8 = thunder_dev_utils_debug_memory_transform_memory_events_float_8();  memory_events_float_8 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_18: "f32[1, 8192, 2]" = x_normed_4 * float_8;  x_normed_4 = float_8 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17758774784 bytes allocated
            memory_events_mul_18 = thunder_dev_utils_debug_memory_transform_memory_events_mul_18();  memory_events_mul_18 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            to_6: "bf16[1, 8192, 2]" = mul_18.to(dtype = torch.bfloat16);  mul_18 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17758742016 bytes allocated
            memory_events_to_6 = thunder_dev_utils_debug_memory_transform_memory_events_to_6();  memory_events_to_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:237 in forward, code: x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
            x_7: "bf16[1, 8192, 2]" = to_6 + x_3;  to_6 = x_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17758709248 bytes allocated
            memory_events_x_7 = thunder_dev_utils_debug_memory_transform_memory_events_x_7();  memory_events_x_7 = None
            return (x_7,)

            # No stacktrace found for following nodes
            memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

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

         # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 34940349952 bytes allocated
        memory_events_tag_activation_checkpoint_1 = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint_1();  memory_events_tag_activation_checkpoint_1 = None
        return tag_activation_checkpoint_1

        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

    class wrap_body_1(torch.nn.Module):
        def forward(self, x_2: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
             # File: <debug>:0 in <debug>, code: memory_events = [], total 17490241024 bytes allocated
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

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x: "f32[1, 8192, 2]" = x_2.float()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17490306560 bytes allocated
            memory_events_x = thunder_dev_utils_debug_memory_transform_memory_events_x();  memory_events_x = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul: "f32[1, 8192, 2]" = x * x

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17490372096 bytes allocated
            memory_events_mul = thunder_dev_utils_debug_memory_transform_memory_events_mul();  memory_events_mul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x: "f32[1, 8192, 1]" = torch.mean(mul, dim = -1, keepdim = True);  mul = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17490339328 bytes allocated
            memory_events_norm_x = thunder_dev_utils_debug_memory_transform_memory_events_norm_x();  memory_events_norm_x = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add: "f32[1, 8192, 1]" = norm_x + 1e-05;  norm_x = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17490339328 bytes allocated
            memory_events_add = thunder_dev_utils_debug_memory_transform_memory_events_add();  memory_events_add = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt: "f32[1, 8192, 1]" = torch.rsqrt(add);  add = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17490339328 bytes allocated
            memory_events_rsqrt = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt();  memory_events_rsqrt = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed: "f32[1, 8192, 2]" = x * rsqrt;  x = rsqrt = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17490404864 bytes allocated
            memory_events_x_normed = thunder_dev_utils_debug_memory_transform_memory_events_x_normed();  memory_events_x_normed = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17490405376 bytes allocated
            memory_events_weight = thunder_dev_utils_debug_memory_transform_memory_events_weight();  memory_events_weight = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_2: "f32[2]" = weight.float();  weight = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 17490405376 bytes allocated
            memory_events_float_2 = thunder_dev_utils_debug_memory_transform_memory_events_float_2();  memory_events_float_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_2: "f32[1, 8192, 2]" = x_normed * float_2;  x_normed = float_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17490470912 bytes allocated
            memory_events_mul_2 = thunder_dev_utils_debug_memory_transform_memory_events_mul_2();  memory_events_mul_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            x_normed_1: "bf16[1, 8192, 2]" = mul_2.to(dtype = torch.bfloat16);  mul_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17490438144 bytes allocated
            memory_events_x_normed_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_1();  memory_events_x_normed_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:269 in forward, code: qkv = self.attn(x)
            qkv: "bf16[1, 8192, 8192]" = torch._C._nn.linear(x_normed_1, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, None);  x_normed_1 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes], total 17624655872 bytes allocated
            memory_events_qkv = thunder_dev_utils_debug_memory_transform_memory_events_qkv();  memory_events_qkv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:274 in forward, code: qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv_1: "bf16[1, 8192, 16, 4, 128]" = qkv.view(1, 8192, 16, 4, 128);  qkv = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_qkv_1 = thunder_dev_utils_debug_memory_transform_memory_events_qkv_1();  memory_events_qkv_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:275 in forward, code: qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
            qkv_2: "bf16[1, 16, 4, 8192, 128]" = qkv_1.permute(0, 2, 3, 1, 4);  qkv_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_qkv_2 = thunder_dev_utils_debug_memory_transform_memory_events_qkv_2();  memory_events_qkv_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            split = qkv_2.split((2, 1, 1), dim = 2);  qkv_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_split = thunder_dev_utils_debug_memory_transform_memory_events_split();  memory_events_split = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            q: "bf16[1, 16, 2, 8192, 128]" = split[0]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_q = thunder_dev_utils_debug_memory_transform_memory_events_q();  memory_events_q = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            k: "bf16[1, 16, 1, 8192, 128]" = split[1]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_k = thunder_dev_utils_debug_memory_transform_memory_events_k();  memory_events_k = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            v: "bf16[1, 16, 1, 8192, 128]" = split[2];  split = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_v = thunder_dev_utils_debug_memory_transform_memory_events_v();  memory_events_v = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:284 in forward, code: k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            k_1: "bf16[1, 16, 2, 8192, 128]" = k.expand(1, 16, 2, 8192, 128);  k = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_k_1 = thunder_dev_utils_debug_memory_transform_memory_events_k_1();  memory_events_k_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:285 in forward, code: v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v_1: "bf16[1, 16, 2, 8192, 128]" = v.expand(1, 16, 2, 8192, 128);  v = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_v_1 = thunder_dev_utils_debug_memory_transform_memory_events_v_1();  memory_events_v_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:287 in forward, code: q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            q_1: "bf16[1, 32, 8192, 128]" = q.reshape(1, -1, 8192, 128);  q = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17691764736 bytes allocated
            memory_events_q_1 = thunder_dev_utils_debug_memory_transform_memory_events_q_1();  memory_events_q_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:288 in forward, code: k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            k_2: "bf16[1, 32, 8192, 128]" = k_1.reshape(1, -1, 8192, 128);  k_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17758873600 bytes allocated
            memory_events_k_2 = thunder_dev_utils_debug_memory_transform_memory_events_k_2();  memory_events_k_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:289 in forward, code: v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
            v_2: "bf16[1, 32, 8192, 128]" = v_1.reshape(1, -1, 8192, 128);  v_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 17691764736 bytes allocated
            memory_events_v_2 = thunder_dev_utils_debug_memory_transform_memory_events_v_2();  memory_events_v_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:291 in forward, code: q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            getitem_3: "bf16[1, 32, 8192, 128]" = q_1[(Ellipsis, slice(None, 128, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17691764736 bytes allocated
            memory_events_getitem_3 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_3();  memory_events_getitem_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(None, 64, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17691764736 bytes allocated
            memory_events_x1 = thunder_dev_utils_debug_memory_transform_memory_events_x1();  memory_events_x1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(64, None, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17691764736 bytes allocated
            memory_events_x2 = thunder_dev_utils_debug_memory_transform_memory_events_x2();  memory_events_x2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg: "bf16[1, 32, 8192, 64]" = -x2;  x2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 33554432 bytes], total 17725319168 bytes allocated
            memory_events_neg = thunder_dev_utils_debug_memory_transform_memory_events_neg();  memory_events_neg = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            rotated: "bf16[1, 32, 8192, 128]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes], total 17758873600 bytes allocated
            memory_events_rotated = thunder_dev_utils_debug_memory_transform_memory_events_rotated();  memory_events_rotated = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_1: "bf16[1, 8192, 128]" = cos.unsqueeze(-3)

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_cos_1 = thunder_dev_utils_debug_memory_transform_memory_events_cos_1();  memory_events_cos_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_1: "bf16[1, 8192, 128]" = sin.unsqueeze(-3)

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_sin_1 = thunder_dev_utils_debug_memory_transform_memory_events_sin_1();  memory_events_sin_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_3: "bf16[1, 32, 8192, 128]" = getitem_3 * cos_1;  getitem_3 = cos_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17825982464 bytes allocated
            memory_events_mul_3 = thunder_dev_utils_debug_memory_transform_memory_events_mul_3();  memory_events_mul_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_4: "bf16[1, 32, 8192, 128]" = rotated * sin_1;  rotated = sin_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17825982464 bytes allocated
            memory_events_mul_4 = thunder_dev_utils_debug_memory_transform_memory_events_mul_4();  memory_events_mul_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            roped: "bf16[1, 32, 8192, 128]" = mul_3 + mul_4;  mul_3 = mul_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17758873600 bytes allocated
            memory_events_roped = thunder_dev_utils_debug_memory_transform_memory_events_roped();  memory_events_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            q_roped: "bf16[1, 32, 8192, 128]" = roped.to(dtype = torch.bfloat16);  roped = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_q_roped = thunder_dev_utils_debug_memory_transform_memory_events_q_roped();  memory_events_q_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:292 in forward, code: k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            getitem_6: "bf16[1, 32, 8192, 128]" = k_2[(Ellipsis, slice(None, 128, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_getitem_6 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_6();  memory_events_getitem_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(None, 64, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_x1_1 = thunder_dev_utils_debug_memory_transform_memory_events_x1_1();  memory_events_x1_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(64, None, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_x2_1 = thunder_dev_utils_debug_memory_transform_memory_events_x2_1();  memory_events_x2_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg_1: "bf16[1, 32, 8192, 64]" = -x2_1;  x2_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 33554432 bytes], total 17792428032 bytes allocated
            memory_events_neg_1 = thunder_dev_utils_debug_memory_transform_memory_events_neg_1();  memory_events_neg_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            rotated_1: "bf16[1, 32, 8192, 128]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes], total 17825982464 bytes allocated
            memory_events_rotated_1 = thunder_dev_utils_debug_memory_transform_memory_events_rotated_1();  memory_events_rotated_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_2: "bf16[1, 8192, 128]" = cos.unsqueeze(-3);  cos = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17825982464 bytes allocated
            memory_events_cos_2 = thunder_dev_utils_debug_memory_transform_memory_events_cos_2();  memory_events_cos_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_2: "bf16[1, 8192, 128]" = sin.unsqueeze(-3);  sin = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17825982464 bytes allocated
            memory_events_sin_2 = thunder_dev_utils_debug_memory_transform_memory_events_sin_2();  memory_events_sin_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_5: "bf16[1, 32, 8192, 128]" = getitem_6 * cos_2;  getitem_6 = cos_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17893091328 bytes allocated
            memory_events_mul_5 = thunder_dev_utils_debug_memory_transform_memory_events_mul_5();  memory_events_mul_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_6: "bf16[1, 32, 8192, 128]" = rotated_1 * sin_2;  rotated_1 = sin_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17893091328 bytes allocated
            memory_events_mul_6 = thunder_dev_utils_debug_memory_transform_memory_events_mul_6();  memory_events_mul_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            roped_1: "bf16[1, 32, 8192, 128]" = mul_5 + mul_6;  mul_5 = mul_6 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17825982464 bytes allocated
            memory_events_roped_1 = thunder_dev_utils_debug_memory_transform_memory_events_roped_1();  memory_events_roped_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            k_roped: "bf16[1, 32, 8192, 128]" = roped_1.to(dtype = torch.bfloat16);  roped_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17825982464 bytes allocated
            memory_events_k_roped = thunder_dev_utils_debug_memory_transform_memory_events_k_roped();  memory_events_k_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            getitem_9: "bf16[1, 32, 8192, 0]" = q_1[(Ellipsis, slice(128, None, None))];  q_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17825982464 bytes allocated
            memory_events_getitem_9 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_9();  memory_events_getitem_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            q_2: "bf16[1, 32, 8192, 128]" = torch.cat((q_roped, getitem_9), dim = -1);  q_roped = getitem_9 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17758873600 bytes allocated
            memory_events_q_2 = thunder_dev_utils_debug_memory_transform_memory_events_q_2();  memory_events_q_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            getitem_10: "bf16[1, 32, 8192, 0]" = k_2[(Ellipsis, slice(128, None, None))];  k_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_getitem_10 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_10();  memory_events_getitem_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            k_3: "bf16[1, 32, 8192, 128]" = torch.cat((k_roped, getitem_10), dim = -1);  k_roped = getitem_10 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17691764736 bytes allocated
            memory_events_k_3 = thunder_dev_utils_debug_memory_transform_memory_events_k_3();  memory_events_k_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            getattr_1: "bf16[1, 32, 128, 8192]" = k_3.mT;  k_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17691764736 bytes allocated
            memory_events_getattr_1 = thunder_dev_utils_debug_memory_transform_memory_events_getattr_1();  memory_events_getattr_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            matmul: "bf16[1, 32, 8192, 8192]" = q_2 @ getattr_1;  q_2 = getattr_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 21986732032 bytes allocated
            memory_events_matmul = thunder_dev_utils_debug_memory_transform_memory_events_matmul();  memory_events_matmul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            scores: "bf16[1, 32, 8192, 8192]" = matmul * 0.08333333333333333;  matmul = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 21986732032 bytes allocated
            memory_events_scores = thunder_dev_utils_debug_memory_transform_memory_events_scores();  memory_events_scores = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            truediv: "bf16[1, 32, 8192, 8192]" = scores / 50.0;  scores = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 21986732032 bytes allocated
            memory_events_truediv = thunder_dev_utils_debug_memory_transform_memory_events_truediv();  memory_events_truediv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            tanh: "bf16[1, 32, 8192, 8192]" = torch.tanh(truediv);  truediv = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 21986732032 bytes allocated
            memory_events_tanh = thunder_dev_utils_debug_memory_transform_memory_events_tanh();  memory_events_tanh = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            scores_1: "bf16[1, 32, 8192, 8192]" = tanh * 50.0;  tanh = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 26281699328 bytes allocated
            memory_events_scores_1 = thunder_dev_utils_debug_memory_transform_memory_events_scores_1();  memory_events_scores_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:339 in scaled_dot_product_attention, code: mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
            ones: "bf16[8192, 8192]" = torch.ones(8192, 8192, dtype = torch.bfloat16, device = device(type='cuda', index=0))

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes], total 26415917056 bytes allocated
            memory_events_ones = thunder_dev_utils_debug_memory_transform_memory_events_ones();  memory_events_ones = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:339 in scaled_dot_product_attention, code: mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
            mask: "bf16[8192, 8192]" = ones.triu(diagonal = 1);  ones = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 26415917056 bytes allocated
            memory_events_mask = thunder_dev_utils_debug_memory_transform_memory_events_mask();  memory_events_mask = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:340 in scaled_dot_product_attention, code: mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            bool_1: "b8[8192, 8192]" = mask.bool()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 26483025920 bytes allocated
            memory_events_bool_1 = thunder_dev_utils_debug_memory_transform_memory_events_bool_1();  memory_events_bool_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:340 in scaled_dot_product_attention, code: mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            masked_fill_: "bf16[8192, 8192]" = mask.masked_fill_(bool_1, -3.3895313892515355e+38);  bool_1 = masked_fill_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 26415917056 bytes allocated
            memory_events_masked_fill_ = thunder_dev_utils_debug_memory_transform_memory_events_masked_fill_();  memory_events_masked_fill_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:341 in scaled_dot_product_attention, code: scores = scores + mask
            scores_2: "bf16[1, 32, 8192, 8192]" = scores_1 + mask;  scores_1 = mask = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 26415917056 bytes allocated
            memory_events_scores_2 = thunder_dev_utils_debug_memory_transform_memory_events_scores_2();  memory_events_scores_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            softmax: "f32[1, 32, 8192, 8192]" = torch.nn.functional.softmax(scores_2, dim = -1, dtype = torch.float32);  scores_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, free_requested - 8589934592 bytes, free_completed - 8589934592 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 30710884352 bytes allocated
            memory_events_softmax = thunder_dev_utils_debug_memory_transform_memory_events_softmax();  memory_events_softmax = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            scores_3: "bf16[1, 32, 8192, 8192]" = softmax.to(dtype = torch.bfloat16);  softmax = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 35005851648 bytes allocated
            memory_events_scores_3 = thunder_dev_utils_debug_memory_transform_memory_events_scores_3();  memory_events_scores_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:343 in scaled_dot_product_attention, code: y = scores @ v
            y: "bf16[1, 32, 8192, 128]" = scores_3 @ v_2;  scores_3 = v_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 35072960512 bytes allocated
            memory_events_y = thunder_dev_utils_debug_memory_transform_memory_events_y();  memory_events_y = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:348 in scaled_dot_product_attention, code: return y.transpose(1, 2)
            y_1: "bf16[1, 8192, 32, 128]" = y.transpose(1, 2);  y = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 35072960512 bytes allocated
            memory_events_y_1 = thunder_dev_utils_debug_memory_transform_memory_events_y_1();  memory_events_y_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:321 in forward, code: y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
            y_2: "bf16[1, 8192, 4096]" = y_1.reshape(1, 8192, 4096);  y_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 35072960512 bytes allocated
            memory_events_y_2 = thunder_dev_utils_debug_memory_transform_memory_events_y_2();  memory_events_y_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:324 in forward, code: return self.proj(y)
            attention_output: "bf16[1, 8192, 2]" = torch._C._nn.linear(y_2, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, None);  y_2 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes], total 35072993280 bytes allocated
            memory_events_attention_output = thunder_dev_utils_debug_memory_transform_memory_events_attention_output();  memory_events_attention_output = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_3: "f32[1, 8192, 2]" = attention_output.float();  attention_output = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073026048 bytes allocated
            memory_events_x_3 = thunder_dev_utils_debug_memory_transform_memory_events_x_3();  memory_events_x_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_9: "f32[1, 8192, 2]" = x_3 * x_3

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073091584 bytes allocated
            memory_events_mul_9 = thunder_dev_utils_debug_memory_transform_memory_events_mul_9();  memory_events_mul_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_1: "f32[1, 8192, 1]" = torch.mean(mul_9, dim = -1, keepdim = True);  mul_9 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35073058816 bytes allocated
            memory_events_norm_x_1 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_1();  memory_events_norm_x_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_5: "f32[1, 8192, 1]" = norm_x_1 + 1e-05;  norm_x_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073058816 bytes allocated
            memory_events_add_5 = thunder_dev_utils_debug_memory_transform_memory_events_add_5();  memory_events_add_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_1: "f32[1, 8192, 1]" = torch.rsqrt(add_5);  add_5 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073058816 bytes allocated
            memory_events_rsqrt_1 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_1();  memory_events_rsqrt_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_2: "f32[1, 8192, 2]" = x_3 * rsqrt_1;  x_3 = rsqrt_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073124352 bytes allocated
            memory_events_x_normed_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_2();  memory_events_x_normed_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_1: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 35073124864 bytes allocated
            memory_events_weight_1 = thunder_dev_utils_debug_memory_transform_memory_events_weight_1();  memory_events_weight_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_4: "f32[2]" = weight_1.float();  weight_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 35073124864 bytes allocated
            memory_events_float_4 = thunder_dev_utils_debug_memory_transform_memory_events_float_4();  memory_events_float_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_11: "f32[1, 8192, 2]" = x_normed_2 * float_4;  x_normed_2 = float_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073190400 bytes allocated
            memory_events_mul_11 = thunder_dev_utils_debug_memory_transform_memory_events_mul_11();  memory_events_mul_11 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            attention_output_1: "bf16[1, 8192, 2]" = mul_11.to(dtype = torch.bfloat16);  mul_11 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35073157632 bytes allocated
            memory_events_attention_output_1 = thunder_dev_utils_debug_memory_transform_memory_events_attention_output_1();  memory_events_attention_output_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:236 in forward, code: x = attention_output + x
            x_4: "bf16[1, 8192, 2]" = attention_output_1 + x_2;  attention_output_1 = x_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073157632 bytes allocated
            memory_events_x_4 = thunder_dev_utils_debug_memory_transform_memory_events_x_4();  memory_events_x_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_5: "f32[1, 8192, 2]" = x_4.float()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073223168 bytes allocated
            memory_events_x_5 = thunder_dev_utils_debug_memory_transform_memory_events_x_5();  memory_events_x_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_12: "f32[1, 8192, 2]" = x_5 * x_5

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073288704 bytes allocated
            memory_events_mul_12 = thunder_dev_utils_debug_memory_transform_memory_events_mul_12();  memory_events_mul_12 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_2: "f32[1, 8192, 1]" = torch.mean(mul_12, dim = -1, keepdim = True);  mul_12 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35073255936 bytes allocated
            memory_events_norm_x_2 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_2();  memory_events_norm_x_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_8: "f32[1, 8192, 1]" = norm_x_2 + 1e-05;  norm_x_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073255936 bytes allocated
            memory_events_add_8 = thunder_dev_utils_debug_memory_transform_memory_events_add_8();  memory_events_add_8 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_2: "f32[1, 8192, 1]" = torch.rsqrt(add_8);  add_8 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073255936 bytes allocated
            memory_events_rsqrt_2 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_2();  memory_events_rsqrt_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_3: "f32[1, 8192, 2]" = x_5 * rsqrt_2;  x_5 = rsqrt_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073321472 bytes allocated
            memory_events_x_normed_3 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_3();  memory_events_x_normed_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_2: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 35073321984 bytes allocated
            memory_events_weight_2 = thunder_dev_utils_debug_memory_transform_memory_events_weight_2();  memory_events_weight_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_6: "f32[2]" = weight_2.float();  weight_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 35073321984 bytes allocated
            memory_events_float_6 = thunder_dev_utils_debug_memory_transform_memory_events_float_6();  memory_events_float_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_14: "f32[1, 8192, 2]" = x_normed_3 * float_6;  x_normed_3 = float_6 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073387520 bytes allocated
            memory_events_mul_14 = thunder_dev_utils_debug_memory_transform_memory_events_mul_14();  memory_events_mul_14 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            to_5: "bf16[1, 8192, 2]" = mul_14.to(dtype = torch.bfloat16);  mul_14 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35073354752 bytes allocated
            memory_events_to_5 = thunder_dev_utils_debug_memory_transform_memory_events_to_5();  memory_events_to_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:406 in forward, code: x_fc_1 = self.fc_1(x)
            x_fc_1: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, None);  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 35073616896 bytes allocated
            memory_events_x_fc_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_fc_1();  memory_events_x_fc_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:407 in forward, code: x_fc_2 = self.fc_2(x)
            x_fc_2: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, None);  to_5 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 35073879040 bytes allocated
            memory_events_x_fc_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_fc_2();  memory_events_x_fc_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            gelu: "bf16[1, 8192, 16]" = torch._C._nn.gelu(x_fc_1, approximate = 'tanh');  x_fc_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 35074141184 bytes allocated
            memory_events_gelu = thunder_dev_utils_debug_memory_transform_memory_events_gelu();  memory_events_gelu = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            x_6: "bf16[1, 8192, 16]" = gelu * x_fc_2;  gelu = x_fc_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 35074403328 bytes allocated
            memory_events_x_6 = thunder_dev_utils_debug_memory_transform_memory_events_x_6();  memory_events_x_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:409 in forward, code: return self.proj(x)
            linear_4: "bf16[1, 8192, 2]" = torch._C._nn.linear(x_6, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, None);  x_6 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes], total 35074436096 bytes allocated
            memory_events_linear_4 = thunder_dev_utils_debug_memory_transform_memory_events_linear_4();  memory_events_linear_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_7: "f32[1, 8192, 2]" = linear_4.float();  linear_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35074468864 bytes allocated
            memory_events_x_7 = thunder_dev_utils_debug_memory_transform_memory_events_x_7();  memory_events_x_7 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_16: "f32[1, 8192, 2]" = x_7 * x_7

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35074534400 bytes allocated
            memory_events_mul_16 = thunder_dev_utils_debug_memory_transform_memory_events_mul_16();  memory_events_mul_16 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_3: "f32[1, 8192, 1]" = torch.mean(mul_16, dim = -1, keepdim = True);  mul_16 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35074501632 bytes allocated
            memory_events_norm_x_3 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_3();  memory_events_norm_x_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_10: "f32[1, 8192, 1]" = norm_x_3 + 1e-05;  norm_x_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35074501632 bytes allocated
            memory_events_add_10 = thunder_dev_utils_debug_memory_transform_memory_events_add_10();  memory_events_add_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_3: "f32[1, 8192, 1]" = torch.rsqrt(add_10);  add_10 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35074501632 bytes allocated
            memory_events_rsqrt_3 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_3();  memory_events_rsqrt_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_4: "f32[1, 8192, 2]" = x_7 * rsqrt_3;  x_7 = rsqrt_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35074567168 bytes allocated
            memory_events_x_normed_4 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_4();  memory_events_x_normed_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_3: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 35074567680 bytes allocated
            memory_events_weight_3 = thunder_dev_utils_debug_memory_transform_memory_events_weight_3();  memory_events_weight_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_8: "f32[2]" = weight_3.float();  weight_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 35074567680 bytes allocated
            memory_events_float_8 = thunder_dev_utils_debug_memory_transform_memory_events_float_8();  memory_events_float_8 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_18: "f32[1, 8192, 2]" = x_normed_4 * float_8;  x_normed_4 = float_8 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35074633216 bytes allocated
            memory_events_mul_18 = thunder_dev_utils_debug_memory_transform_memory_events_mul_18();  memory_events_mul_18 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            to_6: "bf16[1, 8192, 2]" = mul_18.to(dtype = torch.bfloat16);  mul_18 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35074600448 bytes allocated
            memory_events_to_6 = thunder_dev_utils_debug_memory_transform_memory_events_to_6();  memory_events_to_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:237 in forward, code: x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
            x_8: "bf16[1, 8192, 2]" = to_6 + x_4;  to_6 = x_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35074567680 bytes allocated
            memory_events_x_8 = thunder_dev_utils_debug_memory_transform_memory_events_x_8();  memory_events_x_8 = None
            return (x_8,)

            # No stacktrace found for following nodes
            memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

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

         # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 34940349952 bytes allocated
        memory_events_tag_activation_checkpoint_1 = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint_1();  memory_events_tag_activation_checkpoint_1 = None
        return tag_activation_checkpoint_1

        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

    class wrap_body_1(torch.nn.Module):
        def forward(self, x_2: "bf16[1, 8192, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_: "bf16[8192, 2]", cos: "bf16[8192, 128]", sin: "bf16[8192, 128]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_: "bf16[2, 4096]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_: "bf16[2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_: "bf16[16, 2]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_: "bf16[2, 16]", l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_: "bf16[2]"):
             # File: <debug>:0 in <debug>, code: memory_events = [], total 17490241024 bytes allocated
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

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x: "f32[1, 8192, 2]" = x_2.float()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17490306560 bytes allocated
            memory_events_x = thunder_dev_utils_debug_memory_transform_memory_events_x();  memory_events_x = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul: "f32[1, 8192, 2]" = x * x

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17490372096 bytes allocated
            memory_events_mul = thunder_dev_utils_debug_memory_transform_memory_events_mul();  memory_events_mul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x: "f32[1, 8192, 1]" = torch.mean(mul, dim = -1, keepdim = True);  mul = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17490339328 bytes allocated
            memory_events_norm_x = thunder_dev_utils_debug_memory_transform_memory_events_norm_x();  memory_events_norm_x = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add: "f32[1, 8192, 1]" = norm_x + 1e-05;  norm_x = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17490339328 bytes allocated
            memory_events_add = thunder_dev_utils_debug_memory_transform_memory_events_add();  memory_events_add = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt: "f32[1, 8192, 1]" = torch.rsqrt(add);  add = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 17490339328 bytes allocated
            memory_events_rsqrt = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt();  memory_events_rsqrt = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed: "f32[1, 8192, 2]" = x * rsqrt;  x = rsqrt = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17490404864 bytes allocated
            memory_events_x_normed = thunder_dev_utils_debug_memory_transform_memory_events_x_normed();  memory_events_x_normed = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_1_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17490405376 bytes allocated
            memory_events_weight = thunder_dev_utils_debug_memory_transform_memory_events_weight();  memory_events_weight = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_2: "f32[2]" = weight.float();  weight = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 17490405376 bytes allocated
            memory_events_float_2 = thunder_dev_utils_debug_memory_transform_memory_events_float_2();  memory_events_float_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_2: "f32[1, 8192, 2]" = x_normed * float_2;  x_normed = float_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 17490470912 bytes allocated
            memory_events_mul_2 = thunder_dev_utils_debug_memory_transform_memory_events_mul_2();  memory_events_mul_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            x_normed_1: "bf16[1, 8192, 2]" = mul_2.to(dtype = torch.bfloat16);  mul_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 17490438144 bytes allocated
            memory_events_x_normed_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_1();  memory_events_x_normed_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:269 in forward, code: qkv = self.attn(x)
            qkv: "bf16[1, 8192, 8192]" = torch._C._nn.linear(x_normed_1, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_, None);  x_normed_1 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_attn_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes], total 17624655872 bytes allocated
            memory_events_qkv = thunder_dev_utils_debug_memory_transform_memory_events_qkv();  memory_events_qkv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:274 in forward, code: qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
            qkv_1: "bf16[1, 8192, 16, 4, 128]" = qkv.view(1, 8192, 16, 4, 128);  qkv = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_qkv_1 = thunder_dev_utils_debug_memory_transform_memory_events_qkv_1();  memory_events_qkv_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:275 in forward, code: qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)
            qkv_2: "bf16[1, 16, 4, 8192, 128]" = qkv_1.permute(0, 2, 3, 1, 4);  qkv_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_qkv_2 = thunder_dev_utils_debug_memory_transform_memory_events_qkv_2();  memory_events_qkv_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            split = qkv_2.split((2, 1, 1), dim = 2);  qkv_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_split = thunder_dev_utils_debug_memory_transform_memory_events_split();  memory_events_split = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            q: "bf16[1, 16, 2, 8192, 128]" = split[0]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_q = thunder_dev_utils_debug_memory_transform_memory_events_q();  memory_events_q = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            k: "bf16[1, 16, 1, 8192, 128]" = split[1]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_k = thunder_dev_utils_debug_memory_transform_memory_events_k();  memory_events_k = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:278 in forward, code: q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)
            v: "bf16[1, 16, 1, 8192, 128]" = split[2];  split = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_v = thunder_dev_utils_debug_memory_transform_memory_events_v();  memory_events_v = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:284 in forward, code: k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            k_1: "bf16[1, 16, 2, 8192, 128]" = k.expand(1, 16, 2, 8192, 128);  k = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_k_1 = thunder_dev_utils_debug_memory_transform_memory_events_k_1();  memory_events_k_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:285 in forward, code: v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v_1: "bf16[1, 16, 2, 8192, 128]" = v.expand(1, 16, 2, 8192, 128);  v = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17624655872 bytes allocated
            memory_events_v_1 = thunder_dev_utils_debug_memory_transform_memory_events_v_1();  memory_events_v_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:287 in forward, code: q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
            q_1: "bf16[1, 32, 8192, 128]" = q.reshape(1, -1, 8192, 128);  q = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17691764736 bytes allocated
            memory_events_q_1 = thunder_dev_utils_debug_memory_transform_memory_events_q_1();  memory_events_q_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:288 in forward, code: k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
            k_2: "bf16[1, 32, 8192, 128]" = k_1.reshape(1, -1, 8192, 128);  k_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17758873600 bytes allocated
            memory_events_k_2 = thunder_dev_utils_debug_memory_transform_memory_events_k_2();  memory_events_k_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:289 in forward, code: v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)
            v_2: "bf16[1, 32, 8192, 128]" = v_1.reshape(1, -1, 8192, 128);  v_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 17691764736 bytes allocated
            memory_events_v_2 = thunder_dev_utils_debug_memory_transform_memory_events_v_2();  memory_events_v_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:291 in forward, code: q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
            getitem_3: "bf16[1, 32, 8192, 128]" = q_1[(Ellipsis, slice(None, 128, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17691764736 bytes allocated
            memory_events_getitem_3 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_3();  memory_events_getitem_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(None, 64, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17691764736 bytes allocated
            memory_events_x1 = thunder_dev_utils_debug_memory_transform_memory_events_x1();  memory_events_x1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2: "bf16[1, 32, 8192, 64]" = getitem_3[(Ellipsis, slice(64, None, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17691764736 bytes allocated
            memory_events_x2 = thunder_dev_utils_debug_memory_transform_memory_events_x2();  memory_events_x2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg: "bf16[1, 32, 8192, 64]" = -x2;  x2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 33554432 bytes], total 17725319168 bytes allocated
            memory_events_neg = thunder_dev_utils_debug_memory_transform_memory_events_neg();  memory_events_neg = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            rotated: "bf16[1, 32, 8192, 128]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes], total 17758873600 bytes allocated
            memory_events_rotated = thunder_dev_utils_debug_memory_transform_memory_events_rotated();  memory_events_rotated = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_1: "bf16[1, 8192, 128]" = cos.unsqueeze(-3)

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_cos_1 = thunder_dev_utils_debug_memory_transform_memory_events_cos_1();  memory_events_cos_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_1: "bf16[1, 8192, 128]" = sin.unsqueeze(-3)

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_sin_1 = thunder_dev_utils_debug_memory_transform_memory_events_sin_1();  memory_events_sin_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_3: "bf16[1, 32, 8192, 128]" = getitem_3 * cos_1;  getitem_3 = cos_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17825982464 bytes allocated
            memory_events_mul_3 = thunder_dev_utils_debug_memory_transform_memory_events_mul_3();  memory_events_mul_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_4: "bf16[1, 32, 8192, 128]" = rotated * sin_1;  rotated = sin_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17825982464 bytes allocated
            memory_events_mul_4 = thunder_dev_utils_debug_memory_transform_memory_events_mul_4();  memory_events_mul_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            roped: "bf16[1, 32, 8192, 128]" = mul_3 + mul_4;  mul_3 = mul_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17758873600 bytes allocated
            memory_events_roped = thunder_dev_utils_debug_memory_transform_memory_events_roped();  memory_events_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            q_roped: "bf16[1, 32, 8192, 128]" = roped.to(dtype = torch.bfloat16);  roped = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_q_roped = thunder_dev_utils_debug_memory_transform_memory_events_q_roped();  memory_events_q_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:292 in forward, code: k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
            getitem_6: "bf16[1, 32, 8192, 128]" = k_2[(Ellipsis, slice(None, 128, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_getitem_6 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_6();  memory_events_getitem_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:581 in apply_rope, code: x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x1_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(None, 64, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_x1_1 = thunder_dev_utils_debug_memory_transform_memory_events_x1_1();  memory_events_x1_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:582 in apply_rope, code: x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            x2_1: "bf16[1, 32, 8192, 64]" = getitem_6[(Ellipsis, slice(64, None, None))]

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_x2_1 = thunder_dev_utils_debug_memory_transform_memory_events_x2_1();  memory_events_x2_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            neg_1: "bf16[1, 32, 8192, 64]" = -x2_1;  x2_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 33554432 bytes], total 17792428032 bytes allocated
            memory_events_neg_1 = thunder_dev_utils_debug_memory_transform_memory_events_neg_1();  memory_events_neg_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:583 in apply_rope, code: rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            rotated_1: "bf16[1, 32, 8192, 128]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 33554432 bytes, free_completed - 33554432 bytes], total 17825982464 bytes allocated
            memory_events_rotated_1 = thunder_dev_utils_debug_memory_transform_memory_events_rotated_1();  memory_events_rotated_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:588 in apply_rope, code: cos = cos.unsqueeze(-3)
            cos_2: "bf16[1, 8192, 128]" = cos.unsqueeze(-3);  cos = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17825982464 bytes allocated
            memory_events_cos_2 = thunder_dev_utils_debug_memory_transform_memory_events_cos_2();  memory_events_cos_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:589 in apply_rope, code: sin = sin.unsqueeze(-3)
            sin_2: "bf16[1, 8192, 128]" = sin.unsqueeze(-3);  sin = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17825982464 bytes allocated
            memory_events_sin_2 = thunder_dev_utils_debug_memory_transform_memory_events_sin_2();  memory_events_sin_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_5: "bf16[1, 32, 8192, 128]" = getitem_6 * cos_2;  getitem_6 = cos_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 17893091328 bytes allocated
            memory_events_mul_5 = thunder_dev_utils_debug_memory_transform_memory_events_mul_5();  memory_events_mul_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            mul_6: "bf16[1, 32, 8192, 128]" = rotated_1 * sin_2;  rotated_1 = sin_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17893091328 bytes allocated
            memory_events_mul_6 = thunder_dev_utils_debug_memory_transform_memory_events_mul_6();  memory_events_mul_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:591 in apply_rope, code: roped = (x * cos) + (rotated * sin)
            roped_1: "bf16[1, 32, 8192, 128]" = mul_5 + mul_6;  mul_5 = mul_6 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17825982464 bytes allocated
            memory_events_roped_1 = thunder_dev_utils_debug_memory_transform_memory_events_roped_1();  memory_events_roped_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:592 in apply_rope, code: return roped.to(dtype=x.dtype)
            k_roped: "bf16[1, 32, 8192, 128]" = roped_1.to(dtype = torch.bfloat16);  roped_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17825982464 bytes allocated
            memory_events_k_roped = thunder_dev_utils_debug_memory_transform_memory_events_k_roped();  memory_events_k_roped = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            getitem_9: "bf16[1, 32, 8192, 0]" = q_1[(Ellipsis, slice(128, None, None))];  q_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17825982464 bytes allocated
            memory_events_getitem_9 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_9();  memory_events_getitem_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:293 in forward, code: q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
            q_2: "bf16[1, 32, 8192, 128]" = torch.cat((q_roped, getitem_9), dim = -1);  q_roped = getitem_9 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17758873600 bytes allocated
            memory_events_q_2 = thunder_dev_utils_debug_memory_transform_memory_events_q_2();  memory_events_q_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            getitem_10: "bf16[1, 32, 8192, 0]" = k_2[(Ellipsis, slice(128, None, None))];  k_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17758873600 bytes allocated
            memory_events_getitem_10 = thunder_dev_utils_debug_memory_transform_memory_events_getitem_10();  memory_events_getitem_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:294 in forward, code: k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)
            k_3: "bf16[1, 32, 8192, 128]" = torch.cat((k_roped, getitem_10), dim = -1);  k_roped = getitem_10 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 17691764736 bytes allocated
            memory_events_k_3 = thunder_dev_utils_debug_memory_transform_memory_events_k_3();  memory_events_k_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            getattr_1: "bf16[1, 32, 128, 8192]" = k_3.mT;  k_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 17691764736 bytes allocated
            memory_events_getattr_1 = thunder_dev_utils_debug_memory_transform_memory_events_getattr_1();  memory_events_getattr_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            matmul: "bf16[1, 32, 8192, 8192]" = q_2 @ getattr_1;  q_2 = getattr_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 21986732032 bytes allocated
            memory_events_matmul = thunder_dev_utils_debug_memory_transform_memory_events_matmul();  memory_events_matmul = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:334 in scaled_dot_product_attention, code: scores = q @ k.mT * scale
            scores: "bf16[1, 32, 8192, 8192]" = matmul * 0.08333333333333333;  matmul = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 21986732032 bytes allocated
            memory_events_scores = thunder_dev_utils_debug_memory_transform_memory_events_scores();  memory_events_scores = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            truediv: "bf16[1, 32, 8192, 8192]" = scores / 50.0;  scores = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 21986732032 bytes allocated
            memory_events_truediv = thunder_dev_utils_debug_memory_transform_memory_events_truediv();  memory_events_truediv = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            tanh: "bf16[1, 32, 8192, 8192]" = torch.tanh(truediv);  truediv = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 21986732032 bytes allocated
            memory_events_tanh = thunder_dev_utils_debug_memory_transform_memory_events_tanh();  memory_events_tanh = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:336 in scaled_dot_product_attention, code: torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            scores_1: "bf16[1, 32, 8192, 8192]" = tanh * 50.0;  tanh = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 26281699328 bytes allocated
            memory_events_scores_1 = thunder_dev_utils_debug_memory_transform_memory_events_scores_1();  memory_events_scores_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:339 in scaled_dot_product_attention, code: mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
            ones: "bf16[8192, 8192]" = torch.ones(8192, 8192, dtype = torch.bfloat16, device = device(type='cuda', index=0))

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes], total 26415917056 bytes allocated
            memory_events_ones = thunder_dev_utils_debug_memory_transform_memory_events_ones();  memory_events_ones = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:339 in scaled_dot_product_attention, code: mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
            mask: "bf16[8192, 8192]" = ones.triu(diagonal = 1);  ones = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 134217728 bytes, free_requested - 134217728 bytes, free_completed - 134217728 bytes], total 26415917056 bytes allocated
            memory_events_mask = thunder_dev_utils_debug_memory_transform_memory_events_mask();  memory_events_mask = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:340 in scaled_dot_product_attention, code: mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            bool_1: "b8[8192, 8192]" = mask.bool()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 26483025920 bytes allocated
            memory_events_bool_1 = thunder_dev_utils_debug_memory_transform_memory_events_bool_1();  memory_events_bool_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:340 in scaled_dot_product_attention, code: mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            masked_fill_: "bf16[8192, 8192]" = mask.masked_fill_(bool_1, -3.3895313892515355e+38);  bool_1 = masked_fill_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 26415917056 bytes allocated
            memory_events_masked_fill_ = thunder_dev_utils_debug_memory_transform_memory_events_masked_fill_();  memory_events_masked_fill_ = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:341 in scaled_dot_product_attention, code: scores = scores + mask
            scores_2: "bf16[1, 32, 8192, 8192]" = scores_1 + mask;  scores_1 = mask = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 4294967296 bytes, alloc - 4294967296 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 26415917056 bytes allocated
            memory_events_scores_2 = thunder_dev_utils_debug_memory_transform_memory_events_scores_2();  memory_events_scores_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            softmax: "f32[1, 32, 8192, 8192]" = torch.nn.functional.softmax(scores_2, dim = -1, dtype = torch.float32);  scores_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, segment_alloc - 8589934592 bytes, alloc - 8589934592 bytes, free_requested - 8589934592 bytes, free_completed - 8589934592 bytes, free_requested - 4294967296 bytes, free_completed - 4294967296 bytes], total 30710884352 bytes allocated
            memory_events_softmax = thunder_dev_utils_debug_memory_transform_memory_events_softmax();  memory_events_softmax = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:342 in scaled_dot_product_attention, code: scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            scores_3: "bf16[1, 32, 8192, 8192]" = softmax.to(dtype = torch.bfloat16);  softmax = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 4294967296 bytes], total 35005851648 bytes allocated
            memory_events_scores_3 = thunder_dev_utils_debug_memory_transform_memory_events_scores_3();  memory_events_scores_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:343 in scaled_dot_product_attention, code: y = scores @ v
            y: "bf16[1, 32, 8192, 128]" = scores_3 @ v_2;  scores_3 = v_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes], total 35072960512 bytes allocated
            memory_events_y = thunder_dev_utils_debug_memory_transform_memory_events_y();  memory_events_y = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:348 in scaled_dot_product_attention, code: return y.transpose(1, 2)
            y_1: "bf16[1, 8192, 32, 128]" = y.transpose(1, 2);  y = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 35072960512 bytes allocated
            memory_events_y_1 = thunder_dev_utils_debug_memory_transform_memory_events_y_1();  memory_events_y_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:321 in forward, code: y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side
            y_2: "bf16[1, 8192, 4096]" = y_1.reshape(1, 8192, 4096);  y_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 67108864 bytes, free_requested - 67108864 bytes, free_completed - 67108864 bytes], total 35072960512 bytes allocated
            memory_events_y_2 = thunder_dev_utils_debug_memory_transform_memory_events_y_2();  memory_events_y_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:324 in forward, code: return self.proj(y)
            attention_output: "bf16[1, 8192, 2]" = torch._C._nn.linear(y_2, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_, None);  y_2 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_attn_modules_proj_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes], total 35072993280 bytes allocated
            memory_events_attention_output = thunder_dev_utils_debug_memory_transform_memory_events_attention_output();  memory_events_attention_output = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_3: "f32[1, 8192, 2]" = attention_output.float();  attention_output = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073026048 bytes allocated
            memory_events_x_3 = thunder_dev_utils_debug_memory_transform_memory_events_x_3();  memory_events_x_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_9: "f32[1, 8192, 2]" = x_3 * x_3

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073091584 bytes allocated
            memory_events_mul_9 = thunder_dev_utils_debug_memory_transform_memory_events_mul_9();  memory_events_mul_9 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_1: "f32[1, 8192, 1]" = torch.mean(mul_9, dim = -1, keepdim = True);  mul_9 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35073058816 bytes allocated
            memory_events_norm_x_1 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_1();  memory_events_norm_x_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_5: "f32[1, 8192, 1]" = norm_x_1 + 1e-05;  norm_x_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073058816 bytes allocated
            memory_events_add_5 = thunder_dev_utils_debug_memory_transform_memory_events_add_5();  memory_events_add_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_1: "f32[1, 8192, 1]" = torch.rsqrt(add_5);  add_5 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073058816 bytes allocated
            memory_events_rsqrt_1 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_1();  memory_events_rsqrt_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_2: "f32[1, 8192, 2]" = x_3 * rsqrt_1;  x_3 = rsqrt_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073124352 bytes allocated
            memory_events_x_normed_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_2();  memory_events_x_normed_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_1: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_attention_norm_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 35073124864 bytes allocated
            memory_events_weight_1 = thunder_dev_utils_debug_memory_transform_memory_events_weight_1();  memory_events_weight_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_4: "f32[2]" = weight_1.float();  weight_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 35073124864 bytes allocated
            memory_events_float_4 = thunder_dev_utils_debug_memory_transform_memory_events_float_4();  memory_events_float_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_11: "f32[1, 8192, 2]" = x_normed_2 * float_4;  x_normed_2 = float_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073190400 bytes allocated
            memory_events_mul_11 = thunder_dev_utils_debug_memory_transform_memory_events_mul_11();  memory_events_mul_11 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            attention_output_1: "bf16[1, 8192, 2]" = mul_11.to(dtype = torch.bfloat16);  mul_11 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35073157632 bytes allocated
            memory_events_attention_output_1 = thunder_dev_utils_debug_memory_transform_memory_events_attention_output_1();  memory_events_attention_output_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:236 in forward, code: x = attention_output + x
            x_4: "bf16[1, 8192, 2]" = attention_output_1 + x_2;  attention_output_1 = x_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073157632 bytes allocated
            memory_events_x_4 = thunder_dev_utils_debug_memory_transform_memory_events_x_4();  memory_events_x_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_5: "f32[1, 8192, 2]" = x_4.float()

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073223168 bytes allocated
            memory_events_x_5 = thunder_dev_utils_debug_memory_transform_memory_events_x_5();  memory_events_x_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_12: "f32[1, 8192, 2]" = x_5 * x_5

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073288704 bytes allocated
            memory_events_mul_12 = thunder_dev_utils_debug_memory_transform_memory_events_mul_12();  memory_events_mul_12 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_2: "f32[1, 8192, 1]" = torch.mean(mul_12, dim = -1, keepdim = True);  mul_12 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35073255936 bytes allocated
            memory_events_norm_x_2 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_2();  memory_events_norm_x_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_8: "f32[1, 8192, 1]" = norm_x_2 + 1e-05;  norm_x_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073255936 bytes allocated
            memory_events_add_8 = thunder_dev_utils_debug_memory_transform_memory_events_add_8();  memory_events_add_8 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_2: "f32[1, 8192, 1]" = torch.rsqrt(add_8);  add_8 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35073255936 bytes allocated
            memory_events_rsqrt_2 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_2();  memory_events_rsqrt_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_3: "f32[1, 8192, 2]" = x_5 * rsqrt_2;  x_5 = rsqrt_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073321472 bytes allocated
            memory_events_x_normed_3 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_3();  memory_events_x_normed_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_2: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_norm_2_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 35073321984 bytes allocated
            memory_events_weight_2 = thunder_dev_utils_debug_memory_transform_memory_events_weight_2();  memory_events_weight_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_6: "f32[2]" = weight_2.float();  weight_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 35073321984 bytes allocated
            memory_events_float_6 = thunder_dev_utils_debug_memory_transform_memory_events_float_6();  memory_events_float_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_14: "f32[1, 8192, 2]" = x_normed_3 * float_6;  x_normed_3 = float_6 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35073387520 bytes allocated
            memory_events_mul_14 = thunder_dev_utils_debug_memory_transform_memory_events_mul_14();  memory_events_mul_14 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            to_5: "bf16[1, 8192, 2]" = mul_14.to(dtype = torch.bfloat16);  mul_14 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35073354752 bytes allocated
            memory_events_to_5 = thunder_dev_utils_debug_memory_transform_memory_events_to_5();  memory_events_to_5 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:406 in forward, code: x_fc_1 = self.fc_1(x)
            x_fc_1: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_, None);  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_1_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 35073616896 bytes allocated
            memory_events_x_fc_1 = thunder_dev_utils_debug_memory_transform_memory_events_x_fc_1();  memory_events_x_fc_1 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:407 in forward, code: x_fc_2 = self.fc_2(x)
            x_fc_2: "bf16[1, 8192, 16]" = torch._C._nn.linear(to_5, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_, None);  to_5 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_fc_2_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 35073879040 bytes allocated
            memory_events_x_fc_2 = thunder_dev_utils_debug_memory_transform_memory_events_x_fc_2();  memory_events_x_fc_2 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            gelu: "bf16[1, 8192, 16]" = torch._C._nn.gelu(x_fc_1, approximate = 'tanh');  x_fc_1 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 35074141184 bytes allocated
            memory_events_gelu = thunder_dev_utils_debug_memory_transform_memory_events_gelu();  memory_events_gelu = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:408 in forward, code: x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
            x_6: "bf16[1, 8192, 16]" = gelu * x_fc_2;  gelu = x_fc_2 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 262144 bytes], total 35074403328 bytes allocated
            memory_events_x_6 = thunder_dev_utils_debug_memory_transform_memory_events_x_6();  memory_events_x_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:409 in forward, code: return self.proj(x)
            linear_4: "bf16[1, 8192, 2]" = torch._C._nn.linear(x_6, l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_, None);  x_6 = l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_mlp_modules_proj_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes], total 35074436096 bytes allocated
            memory_events_linear_4 = thunder_dev_utils_debug_memory_transform_memory_events_linear_4();  memory_events_linear_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:643 in forward, code: x = x.float()
            x_7: "f32[1, 8192, 2]" = linear_4.float();  linear_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35074468864 bytes allocated
            memory_events_x_7 = thunder_dev_utils_debug_memory_transform_memory_events_x_7();  memory_events_x_7 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            mul_16: "f32[1, 8192, 2]" = x_7 * x_7

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35074534400 bytes allocated
            memory_events_mul_16 = thunder_dev_utils_debug_memory_transform_memory_events_mul_16();  memory_events_mul_16 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:645 in forward, code: norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
            norm_x_3: "f32[1, 8192, 1]" = torch.mean(mul_16, dim = -1, keepdim = True);  mul_16 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35074501632 bytes allocated
            memory_events_norm_x_3 = thunder_dev_utils_debug_memory_transform_memory_events_norm_x_3();  memory_events_norm_x_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            add_10: "f32[1, 8192, 1]" = norm_x_3 + 1e-05;  norm_x_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35074501632 bytes allocated
            memory_events_add_10 = thunder_dev_utils_debug_memory_transform_memory_events_add_10();  memory_events_add_10 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            rsqrt_3: "f32[1, 8192, 1]" = torch.rsqrt(add_10);  add_10 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35074501632 bytes allocated
            memory_events_rsqrt_3 = thunder_dev_utils_debug_memory_transform_memory_events_rsqrt_3();  memory_events_rsqrt_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:646 in forward, code: x_normed = x * torch.rsqrt(norm_x + self.eps)
            x_normed_4: "f32[1, 8192, 2]" = x_7 * rsqrt_3;  x_7 = rsqrt_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35074567168 bytes allocated
            memory_events_x_normed_4 = thunder_dev_utils_debug_memory_transform_memory_events_x_normed_4();  memory_events_x_normed_4 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:647 in forward, code: weight = (1 + self.weight) if self.add_unit_offset else self.weight
            weight_3: "bf16[2]" = 1 + l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_;  l_self_modules_transformer_modules_h_modules_1_modules_checkpoint_wrapped_module_modules_post_mlp_norm_parameters_weight_ = None

             # File: <debug>:0 in <debug>, code: memory_events = [], total 35074567680 bytes allocated
            memory_events_weight_3 = thunder_dev_utils_debug_memory_transform_memory_events_weight_3();  memory_events_weight_3 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            float_8: "f32[2]" = weight_3.float();  weight_3 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 8 bytes], total 35074567680 bytes allocated
            memory_events_float_8 = thunder_dev_utils_debug_memory_transform_memory_events_float_8();  memory_events_float_8 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            mul_18: "f32[1, 8192, 2]" = x_normed_4 * float_8;  x_normed_4 = float_8 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 35074633216 bytes allocated
            memory_events_mul_18 = thunder_dev_utils_debug_memory_transform_memory_events_mul_18();  memory_events_mul_18 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:648 in forward, code: return (x_normed * weight.float()).to(dtype=dtype)
            to_6: "bf16[1, 8192, 2]" = mul_18.to(dtype = torch.bfloat16);  mul_18 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 65536 bytes, free_completed - 65536 bytes], total 35074600448 bytes allocated
            memory_events_to_6 = thunder_dev_utils_debug_memory_transform_memory_events_to_6();  memory_events_to_6 = None

             # File: /usr/local/lib/python3.12/dist-packages/litgpt/model.py:237 in forward, code: x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
            x_8: "bf16[1, 8192, 2]" = to_6 + x_4;  to_6 = x_4 = None

             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes, free_requested - 32768 bytes, free_completed - 32768 bytes], total 35074567680 bytes allocated
            memory_events_x_8 = thunder_dev_utils_debug_memory_transform_memory_events_x_8();  memory_events_x_8 = None
            return (x_8,)

            # No stacktrace found for following nodes
            memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None

