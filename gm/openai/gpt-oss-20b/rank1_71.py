# Rank: 1, Graph 71

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 2880]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_stack0_);  inductor_0 = None
        thunder_1 = self.thunder_1(l_stack0_);  l_stack0_ = None
        return (thunder_1,)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 2880]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_1 = l_stack0_.is_cpu;  getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(l_stack0_, group_name = 'tp:0');  l_stack0_ = inplace_all_reduce = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 2880]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_1 = l_stack0_.is_cpu;  getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(l_stack0_, group_name = 'tp:0');  l_stack0_ = inplace_all_reduce = None
                return ()
                
    class thunder_1(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 2880]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/gpt_oss.py:192 in torch_dynamo_resume_in_forward_normal_at_187, code: ans = final_hidden_states.view(num_tokens, hidden_dim)
            ans: "bf16[1, 2880]" = l_stack0_.view(1, 2880);  l_stack0_ = None
            return ans
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 2880]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/gpt_oss.py:192 in torch_dynamo_resume_in_forward_normal_at_187, code: ans = final_hidden_states.view(num_tokens, hidden_dim)
                ans: "bf16[1, 2880]" = l_stack0_.view(1, 2880);  l_stack0_ = None
                return ans
                