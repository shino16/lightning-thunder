# Rank: 2, Graph 38

class GraphModule(torch.nn.Module):
    def forward(self, l_hidden_states_: "bf16[1, 7168]", l_self_modules_gate_parameters_weight_: "bf16[256, 7168]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0()
        inductor_1 = self.inductor_1(thunder_0, l_hidden_states_, l_self_modules_gate_parameters_weight_);  l_hidden_states_ = l_self_modules_gate_parameters_weight_ = inductor_1 = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/gemm.py:274 in dsv3_router_gemm, code: output = torch.empty(
            output: "f32[1, 256]" = torch.empty(1, 256, device = device(type='cuda', index=2), dtype = torch.float32)
            return output
            
        class _model(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/gemm.py:274 in dsv3_router_gemm, code: output = torch.empty(
                output: "f32[1, 256]" = torch.empty(1, 256, device = device(type='cuda', index=2), dtype = torch.float32)
                return output
                
    class inductor_1(torch.nn.Module):
        def forward(self, output: "f32[1, 256]", l_hidden_states_: "bf16[1, 7168]", l_self_modules_gate_parameters_weight_: "bf16[256, 7168]"):
             # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/gemm.py:280 in dsv3_router_gemm, code: torch.ops.sgl_kernel.dsv3_router_gemm(
            dsv3_router_gemm = torch.ops.sgl_kernel.dsv3_router_gemm(output, l_hidden_states_, l_self_modules_gate_parameters_weight_);  output = l_hidden_states_ = l_self_modules_gate_parameters_weight_ = dsv3_router_gemm = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, output: "f32[1, 256]", l_hidden_states_: "bf16[1, 7168]", l_self_modules_gate_parameters_weight_: "bf16[256, 7168]"):
                 # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/gemm.py:280 in dsv3_router_gemm, code: torch.ops.sgl_kernel.dsv3_router_gemm(
                dsv3_router_gemm = torch.ops.sgl_kernel.dsv3_router_gemm(output, l_hidden_states_, l_self_modules_gate_parameters_weight_);  output = l_hidden_states_ = l_self_modules_gate_parameters_weight_ = dsv3_router_gemm = None
                return ()
                