# Rank: 3, Graph 7

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0();  inductor_0 = None
        return ()
        
    class inductor_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/utils.py:23 in get_cuda_stream, code: return torch.cuda.current_stream().cuda_stream
            current_stream = torch.cuda.current_stream();  current_stream = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/utils.py:23 in get_cuda_stream, code: return torch.cuda.current_stream().cuda_stream
                current_stream = torch.cuda.current_stream();  current_stream = None
                return ()
                