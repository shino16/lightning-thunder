# Rank: 0, Graph 40

class GraphModule(torch.nn.Module):
    def forward(self, l_hidden_states_: "bf16[1, 7168]", l_self_modules_experts_parameters_w13_weight_: "f8e4m3fn[257, 1024, 7168]", l_self_modules_experts_parameters_w2_weight_: "f8e4m3fn[257, 7168, 512]", l_self_modules_experts_parameters_w13_weight_scale_inv_: "f32[257, 8, 56]", l_self_modules_experts_parameters_w2_weight_scale_inv_: "f32[257, 56, 4]", l_stack0_topk_weights: "f32[1, 9]", l_stack0_topk_ids: "i32[1, 9]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_hidden_states_, l_self_modules_experts_parameters_w13_weight_, l_self_modules_experts_parameters_w2_weight_, l_stack0_topk_weights, l_stack0_topk_ids, l_self_modules_experts_parameters_w13_weight_scale_inv_, l_self_modules_experts_parameters_w2_weight_scale_inv_);  l_self_modules_experts_parameters_w13_weight_ = l_self_modules_experts_parameters_w2_weight_ = l_stack0_topk_weights = l_stack0_topk_ids = l_self_modules_experts_parameters_w13_weight_scale_inv_ = l_self_modules_experts_parameters_w2_weight_scale_inv_ = inductor_0 = None
        thunder_1 = self.thunder_1(l_hidden_states_);  l_hidden_states_ = None
        inductor_2 = self.inductor_2(thunder_1);  inductor_2 = None
        return (thunder_1,)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_hidden_states_: "bf16[1, 7168]", l_self_modules_experts_parameters_w13_weight_: "f8e4m3fn[257, 1024, 7168]", l_self_modules_experts_parameters_w2_weight_: "f8e4m3fn[257, 7168, 512]", l_stack0_topk_weights: "f32[1, 9]", l_stack0_topk_ids: "i32[1, 9]", l_self_modules_experts_parameters_w13_weight_scale_inv_: "f32[257, 8, 56]", l_self_modules_experts_parameters_w2_weight_scale_inv_: "f32[257, 56, 4]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:268 in fused_experts, code: torch.ops.sglang.inplace_fused_experts(
            inplace_fused_experts = torch.ops.sglang.inplace_fused_experts(l_hidden_states_, l_self_modules_experts_parameters_w13_weight_, l_self_modules_experts_parameters_w2_weight_, l_stack0_topk_weights, l_stack0_topk_ids, None, None, 'silu', False, True, False, False, False, False, l_self_modules_experts_parameters_w13_weight_scale_inv_, l_self_modules_experts_parameters_w2_weight_scale_inv_, None, None, None, None, [128, 128], 2.5, None, None);  l_hidden_states_ = l_self_modules_experts_parameters_w13_weight_ = l_self_modules_experts_parameters_w2_weight_ = l_stack0_topk_weights = l_stack0_topk_ids = l_self_modules_experts_parameters_w13_weight_scale_inv_ = l_self_modules_experts_parameters_w2_weight_scale_inv_ = inplace_fused_experts = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_hidden_states_: "bf16[1, 7168]", l_self_modules_experts_parameters_w13_weight_: "f8e4m3fn[257, 1024, 7168]", l_self_modules_experts_parameters_w2_weight_: "f8e4m3fn[257, 7168, 512]", l_stack0_topk_weights: "f32[1, 9]", l_stack0_topk_ids: "i32[1, 9]", l_self_modules_experts_parameters_w13_weight_scale_inv_: "f32[257, 8, 56]", l_self_modules_experts_parameters_w2_weight_scale_inv_: "f32[257, 56, 4]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:268 in fused_experts, code: torch.ops.sglang.inplace_fused_experts(
                inplace_fused_experts = torch.ops.sglang.inplace_fused_experts(l_hidden_states_, l_self_modules_experts_parameters_w13_weight_, l_self_modules_experts_parameters_w2_weight_, l_stack0_topk_weights, l_stack0_topk_ids, None, None, 'silu', False, True, False, False, False, False, l_self_modules_experts_parameters_w13_weight_scale_inv_, l_self_modules_experts_parameters_w2_weight_scale_inv_, None, None, None, None, [128, 128], 2.5, None, None);  l_hidden_states_ = l_self_modules_experts_parameters_w13_weight_ = l_self_modules_experts_parameters_w2_weight_ = l_stack0_topk_weights = l_stack0_topk_ids = l_self_modules_experts_parameters_w13_weight_scale_inv_ = l_self_modules_experts_parameters_w2_weight_scale_inv_ = inplace_fused_experts = None
                return ()
                
    class thunder_1(torch.nn.Module):
        def forward(self, l_hidden_states_: "bf16[1, 7168]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:840 in forward, code: final_hidden_states = final_hidden_states[
            getitem: "bf16[1, 7168]" = l_hidden_states_[(Ellipsis, slice(None, 7168, None))];  l_hidden_states_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:842 in forward, code: ].contiguous()
            final_hidden_states: "bf16[1, 7168]" = getitem.contiguous();  getitem = None
            return final_hidden_states
            
        class _model(torch.nn.Module):
            def forward(self, l_hidden_states_: "bf16[1, 7168]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:840 in forward, code: final_hidden_states = final_hidden_states[
                getitem: "bf16[1, 7168]" = l_hidden_states_[(Ellipsis, slice(None, 7168, None))];  l_hidden_states_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:842 in forward, code: ].contiguous()
                final_hidden_states: "bf16[1, 7168]" = getitem.contiguous();  getitem = None
                return final_hidden_states
                
    class inductor_2(torch.nn.Module):
        def forward(self, final_hidden_states: "bf16[1, 7168]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_1 = final_hidden_states.is_cpu;  getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(final_hidden_states, group_name = 'tp:0');  final_hidden_states = inplace_all_reduce = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, final_hidden_states: "bf16[1, 7168]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_1 = final_hidden_states.is_cpu;  getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(final_hidden_states, group_name = 'tp:0');  final_hidden_states = inplace_all_reduce = None
                return ()
                