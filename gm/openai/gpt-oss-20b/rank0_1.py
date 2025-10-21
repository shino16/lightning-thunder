# Rank: 0, Graph 1

class GraphModule(torch.nn.Module):
    def forward(self, l_hidden_states_: "bf16[1, 2880]", l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2880]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_self_layer_communicator_input_layernorm_parameters_weight_);  l_self_layer_communicator_input_layernorm_parameters_weight_ = None
        thunder_1 = self.thunder_1(l_hidden_states_)
        inductor_2 = self.inductor_2(thunder_1, l_hidden_states_, inductor_0);  l_hidden_states_ = inductor_0 = inductor_2 = None
        return (thunder_1,)
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2880]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:97 in forward_cuda, code: out = rmsnorm(x, self.weight.data, self.variance_epsilon)
            _get_data_attr: "bf16[2880]" = torch._C._autograd._get_data_attr(l_self_layer_communicator_input_layernorm_parameters_weight_);  l_self_layer_communicator_input_layernorm_parameters_weight_ = None
            return _get_data_attr
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_self_layer_communicator_input_layernorm_parameters_weight_: "bf16[2880]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/layernorm.py:97 in forward_cuda, code: out = rmsnorm(x, self.weight.data, self.variance_epsilon)
                _get_data_attr: "bf16[2880]" = torch._C._autograd._get_data_attr(l_self_layer_communicator_input_layernorm_parameters_weight_);  l_self_layer_communicator_input_layernorm_parameters_weight_ = None
                return _get_data_attr
                
    class thunder_1(torch.nn.Module):
        def forward(self, l_hidden_states_: "bf16[1, 2880]"):
             # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/elementwise.py:42 in rmsnorm, code: out = torch.empty_like(input)
            out: "bf16[1, 2880]" = torch.empty_like(l_hidden_states_);  l_hidden_states_ = None
            return out
            
        class _model(torch.nn.Module):
            def forward(self, l_hidden_states_: "bf16[1, 2880]"):
                 # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/elementwise.py:42 in rmsnorm, code: out = torch.empty_like(input)
                out: "bf16[1, 2880]" = torch.empty_like(l_hidden_states_);  l_hidden_states_ = None
                return out
                
    class inductor_2(torch.nn.Module):
        def forward(self, out: "bf16[1, 2880]", l_hidden_states_: "bf16[1, 2880]", _get_data_attr: "bf16[2880]"):
             # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/utils.py:49 in is_arch_support_pdl, code: major, minor = torch.cuda.get_device_capability(device)
            get_device_capability = torch.cuda.get_device_capability(0);  get_device_capability = None
            
             # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/elementwise.py:45 in rmsnorm, code: torch.ops.sgl_kernel.rmsnorm.default(out, input, weight, eps, enable_pdl)
            rmsnorm_default = torch.ops.sgl_kernel.rmsnorm.default(out, l_hidden_states_, _get_data_attr, 1e-05, True);  out = l_hidden_states_ = _get_data_attr = rmsnorm_default = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, out: "bf16[1, 2880]", l_hidden_states_: "bf16[1, 2880]", _get_data_attr: "bf16[2880]"):
                 # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/utils.py:49 in is_arch_support_pdl, code: major, minor = torch.cuda.get_device_capability(device)
                get_device_capability = torch.cuda.get_device_capability(0);  get_device_capability = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/elementwise.py:45 in rmsnorm, code: torch.ops.sgl_kernel.rmsnorm.default(out, input, weight, eps, enable_pdl)
                rmsnorm_default = torch.ops.sgl_kernel.rmsnorm.default(out, l_hidden_states_, _get_data_attr, 1e-05, True);  out = l_hidden_states_ = _get_data_attr = rmsnorm_default = None
                return ()
                