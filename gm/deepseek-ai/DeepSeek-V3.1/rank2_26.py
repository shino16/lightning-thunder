# Rank: 2, Graph 26

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_0_: "bf16[1, 9216]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_0_);  l_stack0_0_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_0_: "bf16[1, 9216]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/activation.py:64 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
            getitem: "bf16[1, 4608]" = l_stack0_0_[(Ellipsis, slice(None, 4608, None))]
            silu: "bf16[1, 4608]" = torch.nn.functional.silu(getitem);  getitem = None
            getitem_1: "bf16[1, 4608]" = l_stack0_0_[(Ellipsis, slice(4608, None, None))];  l_stack0_0_ = None
            x: "bf16[1, 4608]" = silu * getitem_1;  silu = getitem_1 = None
            return x
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_0_: "bf16[1, 9216]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/activation.py:64 in forward_native, code: return F.silu(x[..., :d]) * x[..., d:]
                getitem: "bf16[1, 4608]" = l_stack0_0_[(Ellipsis, slice(None, 4608, None))]
                silu: "bf16[1, 4608]" = torch.nn.functional.silu(getitem);  getitem = None
                getitem_1: "bf16[1, 4608]" = l_stack0_0_[(Ellipsis, slice(4608, None, None))];  l_stack0_0_ = None
                x: "bf16[1, 4608]" = silu * getitem_1;  silu = getitem_1 = None
                return x
                