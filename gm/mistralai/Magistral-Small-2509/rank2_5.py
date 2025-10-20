# Rank: 2, Graph 5

class GraphModule(torch.nn.Module):
    def forward(self, l_self_modules_o_proj_parameters_weight_: "f16[5120, 1024]", l_stack0_: "f16[1, 1024]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_, l_self_modules_o_proj_parameters_weight_);  l_stack0_ = l_self_modules_o_proj_parameters_weight_ = None
        inductor_1 = self.inductor_1(thunder_0);  inductor_1 = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "f16[1, 1024]", l_self_modules_o_proj_parameters_weight_: "f16[5120, 1024]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
            output_parallel: "f16[1, 5120]" = torch._C._nn.linear(l_stack0_, l_self_modules_o_proj_parameters_weight_, None);  l_stack0_ = l_self_modules_o_proj_parameters_weight_ = None
            return output_parallel
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "f16[1, 1024]", l_self_modules_o_proj_parameters_weight_: "f16[5120, 1024]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:133 in apply, code: return F.linear(x, layer.weight, bias)
                output_parallel: "f16[1, 5120]" = torch._C._nn.linear(l_stack0_, l_self_modules_o_proj_parameters_weight_, None);  l_stack0_ = l_self_modules_o_proj_parameters_weight_ = None
                return output_parallel
                
    class inductor_1(torch.nn.Module):
        def forward(self, output_parallel: "f16[1, 5120]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_1 = output_parallel.is_cpu;  getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(output_parallel, group_name = 'tp:0');  output_parallel = inplace_all_reduce = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, output_parallel: "f16[1, 5120]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_1 = output_parallel.is_cpu;  getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(output_parallel, group_name = 'tp:0');  output_parallel = inplace_all_reduce = None
                return ()
                