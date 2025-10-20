# Rank: 0, Graph 16

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_: "bf16[1, 16384]", l_self_w_vc: "bf16[32, 512, 128]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_);  l_stack0_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1]
        getitem_2 = thunder_0[2];  thunder_0 = None
        inductor_1 = self.inductor_1(getitem, l_self_w_vc, getitem_1);  getitem = l_self_w_vc = getitem_1 = inductor_1 = None
        return (getitem_2,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 16384]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1500 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
            attn_output: "bf16[1, 32, 512]" = l_stack0_.view(-1, 32, 512);  l_stack0_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1565 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: attn_bmm_output = torch.empty(
            attn_bmm_output: "bf16[1, 4096]" = torch.empty((1, 4096), dtype = torch.bfloat16, device = device(type='cuda', index=0))
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1571 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: attn_output.transpose(0, 1),
            transpose: "bf16[32, 1, 512]" = attn_output.transpose(0, 1);  attn_output = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1573 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: out=attn_bmm_output.view(
            view_1: "bf16[1, 32, 128]" = attn_bmm_output.view(-1, 32, 128)
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1575 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: ).transpose(0, 1),
            transpose_1: "bf16[32, 1, 128]" = view_1.transpose(0, 1);  view_1 = None
            return (transpose, transpose_1, attn_bmm_output)
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 16384]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1500 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
                attn_output: "bf16[1, 32, 512]" = l_stack0_.view(-1, 32, 512);  l_stack0_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1565 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: attn_bmm_output = torch.empty(
                attn_bmm_output: "bf16[1, 4096]" = torch.empty((1, 4096), dtype = torch.bfloat16, device = device(type='cuda', index=0))
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1571 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: attn_output.transpose(0, 1),
                transpose: "bf16[32, 1, 512]" = attn_output.transpose(0, 1);  attn_output = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1573 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: out=attn_bmm_output.view(
                view_1: "bf16[1, 32, 128]" = attn_bmm_output.view(-1, 32, 128)
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1575 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: ).transpose(0, 1),
                transpose_1: "bf16[32, 1, 128]" = view_1.transpose(0, 1);  view_1 = None
                return (transpose, transpose_1, attn_bmm_output)
                
    class inductor_1(torch.nn.Module):
        def forward(self, transpose: "bf16[32, 1, 512]", l_self_w_vc: "bf16[32, 512, 128]", transpose_1: "bf16[32, 1, 128]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1572 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: self.w_vc,
            bmm: "bf16[32, 1, 128]" = torch.bmm(transpose, l_self_w_vc, out = transpose_1);  transpose = l_self_w_vc = transpose_1 = bmm = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, transpose: "bf16[32, 1, 512]", l_self_w_vc: "bf16[32, 512, 128]", transpose_1: "bf16[32, 1, 128]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/models/deepseek_v2.py:1572 in torch_dynamo_resume_in_forward_absorb_core_at_1472, code: self.w_vc,
                bmm: "bf16[32, 1, 128]" = torch.bmm(transpose, l_self_w_vc, out = transpose_1);  transpose = l_self_w_vc = transpose_1 = bmm = None
                return ()
                