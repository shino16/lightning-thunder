# Rank: 2, Graph 31

class GraphModule(torch.nn.Module):
    def forward(self, l_input_: "bf16[1, 7168]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_input_);  inductor_0 = None
        return (l_input_,)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_input_: "bf16[1, 7168]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_1 = l_input_.is_cpu;  getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(l_input_, group_name = 'tp:0');  l_input_ = inplace_all_reduce = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_input_: "bf16[1, 7168]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_1 = l_input_.is_cpu;  getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(l_input_, group_name = 'tp:0');  l_input_ = inplace_all_reduce = None
                return ()
                