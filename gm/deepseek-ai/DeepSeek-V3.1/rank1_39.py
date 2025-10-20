# Rank: 1, Graph 39

class GraphModule(torch.nn.Module):
    def forward(self, l_gating_output_: "f32[1, 256]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_gating_output_);  l_gating_output_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_gating_output_: "f32[1, 256]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:690 in biased_grouped_topk_gpu, code: gating_output.to(dtype=torch.float32),
            to: "f32[1, 256]" = l_gating_output_.to(dtype = torch.float32);  l_gating_output_ = None
            return to
            
        class _model(torch.nn.Module):
            def forward(self, l_gating_output_: "f32[1, 256]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/moe/topk.py:690 in biased_grouped_topk_gpu, code: gating_output.to(dtype=torch.float32),
                to: "f32[1, 256]" = l_gating_output_.to(dtype = torch.float32);  l_gating_output_ = None
                return to
                