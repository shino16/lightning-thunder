# Rank: 2, Graph 21

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_hidden_states: "bf16[1, 3072]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_hidden_states);  l_stack0_hidden_states = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_hidden_states: "bf16[1, 3072]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:840 in torch_dynamo_resume_in_forward_at_833, code: final_hidden_states = final_hidden_states[
            getitem: "bf16[1, 2880]" = l_stack0_hidden_states[(Ellipsis, slice(None, 2880, None))];  l_stack0_hidden_states = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:842 in torch_dynamo_resume_in_forward_at_833, code: ].contiguous()
            final_hidden_states: "bf16[1, 2880]" = getitem.contiguous();  getitem = None
            return final_hidden_states
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_hidden_states: "bf16[1, 3072]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:840 in torch_dynamo_resume_in_forward_at_833, code: final_hidden_states = final_hidden_states[
                getitem: "bf16[1, 2880]" = l_stack0_hidden_states[(Ellipsis, slice(None, 2880, None))];  l_stack0_hidden_states = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:842 in torch_dynamo_resume_in_forward_at_833, code: ].contiguous()
                final_hidden_states: "bf16[1, 2880]" = getitem.contiguous();  getitem = None
                return final_hidden_states
                