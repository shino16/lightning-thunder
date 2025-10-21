# Rank: 0, Graph 12

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 16, 64]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_);  l_stack0_ = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 16, 64]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:574 in torch_dynamo_resume_in_forward_decode_at_560, code: return o.view(-1, layer.tp_q_head_num * layer.head_dim)
            view: "bf16[1, 1024]" = l_stack0_.view(-1, 1024);  l_stack0_ = None
            return view
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 16, 64]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/attention/trtllm_mha_backend.py:574 in torch_dynamo_resume_in_forward_decode_at_560, code: return o.view(-1, layer.tp_q_head_num * layer.head_dim)
                view: "bf16[1, 1024]" = l_stack0_.view(-1, 1024);  l_stack0_ = None
                return view
                