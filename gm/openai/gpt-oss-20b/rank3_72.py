# Rank: 3, Graph 72

class GraphModule(torch.nn.Module):
    def forward(self, l_self_parameters_weight_: "bf16[2880]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_self_parameters_weight_);  l_self_parameters_weight_ = inductor_0 = None
        return ()
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_self_parameters_weight_: "bf16[2880]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:95 in forward_cuda, code: fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            _get_data_attr: "bf16[2880]" = torch._C._autograd._get_data_attr(l_self_parameters_weight_);  l_self_parameters_weight_ = _get_data_attr = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_self_parameters_weight_: "bf16[2880]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:95 in forward_cuda, code: fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
                _get_data_attr: "bf16[2880]" = torch._C._autograd._get_data_attr(l_self_parameters_weight_);  l_self_parameters_weight_ = _get_data_attr = None
                return ()
                