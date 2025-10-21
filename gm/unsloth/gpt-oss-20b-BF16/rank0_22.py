# Rank: 0, Graph 22

class GraphModule(torch.nn.Module):
    def forward(self, l_expt_offs_: "i32[33]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_expt_offs_);  l_expt_offs_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_expt_offs_: "i32[33]"):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:208 in apply_postprocessing_features, code: num_rows = num_indx or (None if expt_offs is None else expt_offs[-1])
            num_rows: "i32[]" = l_expt_offs_[-1];  l_expt_offs_ = None
            return num_rows
            
        class _model(torch.nn.Module):
            def forward(self, l_expt_offs_: "i32[33]"):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/matmul_ogs.py:208 in apply_postprocessing_features, code: num_rows = num_indx or (None if expt_offs is None else expt_offs[-1])
                num_rows: "i32[]" = l_expt_offs_[-1];  l_expt_offs_ = None
                return num_rows
                