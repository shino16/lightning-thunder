# Rank: 1, Graph 22

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 2880]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_);  l_stack0_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 2880]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/gpt_oss.py:192 in torch_dynamo_resume_in_forward_normal_at_187, code: ans = final_hidden_states.view(num_tokens, hidden_dim)
            ans: "bf16[1, 2880]" = l_stack0_.view(1, 2880);  l_stack0_ = None
            return ans
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 2880]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/gpt_oss.py:192 in torch_dynamo_resume_in_forward_normal_at_187, code: ans = final_hidden_states.view(num_tokens, hidden_dim)
                ans: "bf16[1, 2880]" = l_stack0_.view(1, 2880);  l_stack0_ = None
                return ans
                