# Rank: 1, Graph 80

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0();  inductor_0 = None
        return ()
        
    class inductor_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:236 in get_compute_capability, code: return torch.cuda.get_device_capability(device.index)
            get_device_capability = torch.cuda.get_device_capability(1);  get_device_capability = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/flashinfer/utils.py:236 in get_compute_capability, code: return torch.cuda.get_device_capability(device.index)
                get_device_capability = torch.cuda.get_device_capability(1);  get_device_capability = None
                return ()
                