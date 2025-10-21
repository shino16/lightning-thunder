# Rank: 0, Graph 23

class GraphModule(torch.nn.Module):
    def forward(self, l_hidden_states_: "bf16[1, 2880]", l_residual_: "bf16[1, 2880]", l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2880]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_hidden_states_, l_residual_, l_self_layer_communicator_input_layernorm_parameters_weight_);  l_hidden_states_ = l_residual_ = l_self_layer_communicator_input_layernorm_parameters_weight_ = None
        thunder_1 = self.thunder_1(inductor_0);  inductor_0 = None
        getitem = thunder_1[0]
        getitem_1 = thunder_1[1];  thunder_1 = None
        return (getitem, getitem_1)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_hidden_states_: "bf16[1, 2880]", l_residual_: "bf16[1, 2880]", l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2880]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:236 in forward_with_allreduce_fusion, code: fused_result = fused_op(
            flashinfer_allreduce_residual_rmsnorm = torch.ops.sglang.flashinfer_allreduce_residual_rmsnorm(input_tensor = l_hidden_states_, residual = l_residual_, weight = l_self_layer_communicator_input_layernorm_parameters_weight_, eps = 1e-05);  l_hidden_states_ = l_residual_ = l_self_layer_communicator_input_layernorm_parameters_weight_ = None
            return flashinfer_allreduce_residual_rmsnorm
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_hidden_states_: "bf16[1, 2880]", l_residual_: "bf16[1, 2880]", l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2880]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:236 in forward_with_allreduce_fusion, code: fused_result = fused_op(
                flashinfer_allreduce_residual_rmsnorm = torch.ops.sglang.flashinfer_allreduce_residual_rmsnorm(input_tensor = l_hidden_states_, residual = l_residual_, weight = l_self_layer_communicator_input_layernorm_parameters_weight_, eps = 1e-05);  l_hidden_states_ = l_residual_ = l_self_layer_communicator_input_layernorm_parameters_weight_ = None
                return flashinfer_allreduce_residual_rmsnorm
                
    class thunder_1(torch.nn.Module):
        def forward(self, flashinfer_allreduce_residual_rmsnorm):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:236 in forward_with_allreduce_fusion, code: fused_result = fused_op(
            hidden_states: "bf16[1, 2880]" = flashinfer_allreduce_residual_rmsnorm[0]
            residual: "bf16[1, 2880]" = flashinfer_allreduce_residual_rmsnorm[1];  flashinfer_allreduce_residual_rmsnorm = None
            return (hidden_states, residual)
            
        class _model(torch.nn.Module):
            def forward(self, flashinfer_allreduce_residual_rmsnorm):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:236 in forward_with_allreduce_fusion, code: fused_result = fused_op(
                hidden_states: "bf16[1, 2880]" = flashinfer_allreduce_residual_rmsnorm[0]
                residual: "bf16[1, 2880]" = flashinfer_allreduce_residual_rmsnorm[1];  flashinfer_allreduce_residual_rmsnorm = None
                return (hidden_states, residual)
                