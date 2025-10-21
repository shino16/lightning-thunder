# Rank: 3, Graph 19

class GraphModule(torch.nn.Module):
    def forward(self, l_topk_output_router_logits: "bf16[1, 32]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_topk_output_router_logits);  l_topk_output_router_logits = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_topk_output_router_logits: "bf16[1, 32]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/mxfp4.py:675 in torch_dynamo_resume_in_apply_at_669, code: router_logits.to(torch.bfloat16),
            to: "bf16[1, 32]" = l_topk_output_router_logits.to(torch.bfloat16);  l_topk_output_router_logits = None
            return to
            
        class _model(torch.nn.Module):
            def forward(self, l_topk_output_router_logits: "bf16[1, 32]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/mxfp4.py:675 in torch_dynamo_resume_in_apply_at_669, code: router_logits.to(torch.bfloat16),
                to: "bf16[1, 32]" = l_topk_output_router_logits.to(torch.bfloat16);  l_topk_output_router_logits = None
                return to
                