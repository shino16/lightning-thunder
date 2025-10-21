# Rank: 0, Graph 8

class GraphModule(torch.nn.Module):
    def forward(self, l_x_: "bf16[20842304, 128]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_x_);  l_x_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_x_: "bf16[20842304, 128]"):
             # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/elementwise.py:321 in _view_3d, code: return x.view(x.shape[0], -1, head_size)
            view: "bf16[20842304, 2, 64]" = l_x_.view(20842304, -1, 64);  l_x_ = None
            return view
            
        class _model(torch.nn.Module):
            def forward(self, l_x_: "bf16[20842304, 128]"):
                 # File: /usr/local/lib/python3.12/dist-packages/sgl_kernel/elementwise.py:321 in _view_3d, code: return x.view(x.shape[0], -1, head_size)
                view: "bf16[20842304, 2, 64]" = l_x_.view(20842304, -1, 64);  l_x_ = None
                return view
                