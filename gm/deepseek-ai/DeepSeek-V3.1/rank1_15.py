# Rank: 1, Graph 15

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 32, 512]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_);  l_stack0_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 32, 512]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:638 in torch_dynamo_resume_in_forward_decode_at_630, code: return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
            view: "bf16[1, 16384]" = l_stack0_.view(-1, 16384);  l_stack0_ = None
            return view
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 32, 512]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:638 in torch_dynamo_resume_in_forward_decode_at_630, code: return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
                view: "bf16[1, 16384]" = l_stack0_.view(-1, 16384);  l_stack0_ = None
                return view
                