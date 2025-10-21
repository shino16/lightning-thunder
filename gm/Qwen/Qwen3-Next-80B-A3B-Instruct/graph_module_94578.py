class GraphModule(torch.nn.Module):
    def forward(self, L_input_: "i64[1]", L_self_parameters_weight_: "bf16[18992, 2048]"):
        l_input_ = L_input_
        l_self_parameters_weight_ = L_self_parameters_weight_
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:136 in get_masked_input_and_mask, code: org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
        ge: "b8[1]" = l_input_ >= 94960
        lt: "b8[1]" = l_input_ < 113952
        org_vocab_mask: "b8[1]" = ge & lt;  ge = lt = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:137 in get_masked_input_and_mask, code: added_vocab_mask = (input_ >= added_vocab_start_index) & (
        ge_1: "b8[1]" = l_input_ >= 151936
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:138 in get_masked_input_and_mask, code: input_ < added_vocab_end_index
        lt_1: "b8[1]" = l_input_ < 151936
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:137 in get_masked_input_and_mask, code: added_vocab_mask = (input_ >= added_vocab_start_index) & (
        added_vocab_mask: "b8[1]" = ge_1 & lt_1;  ge_1 = lt_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:145 in get_masked_input_and_mask, code: valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        mul: "i64[1]" = 94960 * org_vocab_mask
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:146 in get_masked_input_and_mask, code: added_offset * added_vocab_mask
        mul_1: "i64[1]" = 132944 * added_vocab_mask
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:145 in get_masked_input_and_mask, code: valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        valid_offset: "i64[1]" = mul + mul_1;  mul = mul_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:148 in get_masked_input_and_mask, code: vocab_mask = org_vocab_mask | added_vocab_mask
        vocab_mask: "b8[1]" = org_vocab_mask | added_vocab_mask;  org_vocab_mask = added_vocab_mask = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:149 in get_masked_input_and_mask, code: input_ = vocab_mask * (input_ - valid_offset)
        sub: "i64[1]" = l_input_ - valid_offset;  l_input_ = valid_offset = None
        input_: "i64[1]" = vocab_mask * sub;  sub = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:150 in get_masked_input_and_mask, code: return input_, ~vocab_mask
        input_mask: "b8[1]" = ~vocab_mask;  vocab_mask = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:477 in forward, code: output_parallel = self.quant_method.embedding(self, masked_input.long())
        long: "i64[1]" = input_.long();  input_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:83 in embedding, code: return F.embedding(input_, layer.weight)
        output_parallel: "bf16[1, 2048]" = torch.nn.functional.embedding(long, l_self_parameters_weight_);  long = l_self_parameters_weight_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:481 in forward, code: output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        unsqueeze: "b8[1, 1]" = input_mask.unsqueeze(-1);  input_mask = None
        masked_fill_: "bf16[1, 2048]" = output_parallel.masked_fill_(unsqueeze, 0);  unsqueeze = masked_fill_ = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
        getattr_1 = output_parallel.is_cpu;  getattr_1 = None
        
         # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
        inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(output_parallel, group_name = 'tp:0');  inplace_all_reduce = None
        return (output_parallel,)
        