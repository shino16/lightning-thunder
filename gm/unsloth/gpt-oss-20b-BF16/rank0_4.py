# Rank: 0, Graph 4

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0();  inductor_0 = None
        return ()
        
    class inductor_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/utils.py:49 in is_arch_support_pdl, code: major, minor = torch.cuda.get_device_capability(device)
            get_device_capability = torch.cuda.get_device_capability(0);  get_device_capability = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/utils.py:49 in is_arch_support_pdl, code: major, minor = torch.cuda.get_device_capability(device)
                get_device_capability = torch.cuda.get_device_capability(0);  get_device_capability = None
                return ()
                