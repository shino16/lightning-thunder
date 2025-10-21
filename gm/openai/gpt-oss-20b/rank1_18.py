# Rank: 1, Graph 18

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_1_: "u8[96]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_1_);  l_stack0_1_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_1_: "u8[96]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/mxfp4.py:664 in torch_dynamo_resume_in_apply_at_663, code: x_scale = x_scale.view(torch.float8_e4m3fn).reshape(-1)
            view: "f8e4m3fn[96]" = l_stack0_1_.view(torch.float8_e4m3fn);  l_stack0_1_ = None
            x_scale: "f8e4m3fn[96]" = view.reshape(-1);  view = None
            return x_scale
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_1_: "u8[96]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/mxfp4.py:664 in torch_dynamo_resume_in_apply_at_663, code: x_scale = x_scale.view(torch.float8_e4m3fn).reshape(-1)
                view: "f8e4m3fn[96]" = l_stack0_1_.view(torch.float8_e4m3fn);  l_stack0_1_ = None
                x_scale: "f8e4m3fn[96]" = view.reshape(-1);  view = None
                return x_scale
                