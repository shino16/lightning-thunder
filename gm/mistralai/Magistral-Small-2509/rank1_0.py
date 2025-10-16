# Rank: 1, Graph 0

class GraphModule(torch.nn.Module):
    def forward(self, l_input_ids_: "i64[1]", l_language_model_modules_model_modules_embed_tokens_parameters_weight_: "f16[32768, 5120]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_input_ids_, l_language_model_modules_model_modules_embed_tokens_parameters_weight_);  l_input_ids_ = l_language_model_modules_model_modules_embed_tokens_parameters_weight_ = None
        inductor_1 = self.inductor_1(thunder_0);  inductor_1 = None
        return (thunder_0,)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_input_ids_: "i64[1]", l_language_model_modules_model_modules_embed_tokens_parameters_weight_: "f16[32768, 5120]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:136 in get_masked_input_and_mask, code: org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
            ge: "b8[1]" = l_input_ids_ >= 32768
            lt: "b8[1]" = l_input_ids_ < 65536
            org_vocab_mask: "b8[1]" = ge & lt;  ge = lt = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:137 in get_masked_input_and_mask, code: added_vocab_mask = (input_ >= added_vocab_start_index) & (
            ge_1: "b8[1]" = l_input_ids_ >= 131072
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:138 in get_masked_input_and_mask, code: input_ < added_vocab_end_index
            lt_1: "b8[1]" = l_input_ids_ < 131072
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:137 in get_masked_input_and_mask, code: added_vocab_mask = (input_ >= added_vocab_start_index) & (
            added_vocab_mask: "b8[1]" = ge_1 & lt_1;  ge_1 = lt_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:145 in get_masked_input_and_mask, code: valid_offset = (org_vocab_start_index * org_vocab_mask) + (
            mul: "i64[1]" = 32768 * org_vocab_mask
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:146 in get_masked_input_and_mask, code: added_offset * added_vocab_mask
            mul_1: "i64[1]" = 98304 * added_vocab_mask
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:145 in get_masked_input_and_mask, code: valid_offset = (org_vocab_start_index * org_vocab_mask) + (
            valid_offset: "i64[1]" = mul + mul_1;  mul = mul_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:148 in get_masked_input_and_mask, code: vocab_mask = org_vocab_mask | added_vocab_mask
            vocab_mask: "b8[1]" = org_vocab_mask | added_vocab_mask;  org_vocab_mask = added_vocab_mask = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:149 in get_masked_input_and_mask, code: input_ = vocab_mask * (input_ - valid_offset)
            sub: "i64[1]" = l_input_ids_ - valid_offset;  l_input_ids_ = valid_offset = None
            input_: "i64[1]" = vocab_mask * sub;  sub = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:150 in get_masked_input_and_mask, code: return input_, ~vocab_mask
            input_mask: "b8[1]" = ~vocab_mask;  vocab_mask = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:477 in forward, code: output_parallel = self.quant_method.embedding(self, masked_input.long())
            long: "i64[1]" = input_.long();  input_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:83 in embedding, code: return F.embedding(input_, layer.weight)
            output_parallel: "f16[1, 5120]" = torch.nn.functional.embedding(long, l_language_model_modules_model_modules_embed_tokens_parameters_weight_);  long = l_language_model_modules_model_modules_embed_tokens_parameters_weight_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:481 in forward, code: output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            unsqueeze: "b8[1, 1]" = input_mask.unsqueeze(-1);  input_mask = None
            masked_fill_: "f16[1, 5120]" = output_parallel.masked_fill_(unsqueeze, 0);  unsqueeze = masked_fill_ = None
            return output_parallel
            
        class _model(torch.nn.Module):
            def forward(self, l_input_ids_: "i64[1]", l_language_model_modules_model_modules_embed_tokens_parameters_weight_: "f16[32768, 5120]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:136 in get_masked_input_and_mask, code: org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
                ge: "b8[1]" = l_input_ids_ >= 32768
                lt: "b8[1]" = l_input_ids_ < 65536
                org_vocab_mask: "b8[1]" = ge & lt;  ge = lt = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:137 in get_masked_input_and_mask, code: added_vocab_mask = (input_ >= added_vocab_start_index) & (
                ge_1: "b8[1]" = l_input_ids_ >= 131072
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:138 in get_masked_input_and_mask, code: input_ < added_vocab_end_index
                lt_1: "b8[1]" = l_input_ids_ < 131072
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:137 in get_masked_input_and_mask, code: added_vocab_mask = (input_ >= added_vocab_start_index) & (
                added_vocab_mask: "b8[1]" = ge_1 & lt_1;  ge_1 = lt_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:145 in get_masked_input_and_mask, code: valid_offset = (org_vocab_start_index * org_vocab_mask) + (
                mul: "i64[1]" = 32768 * org_vocab_mask
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:146 in get_masked_input_and_mask, code: added_offset * added_vocab_mask
                mul_1: "i64[1]" = 98304 * added_vocab_mask
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:145 in get_masked_input_and_mask, code: valid_offset = (org_vocab_start_index * org_vocab_mask) + (
                valid_offset: "i64[1]" = mul + mul_1;  mul = mul_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:148 in get_masked_input_and_mask, code: vocab_mask = org_vocab_mask | added_vocab_mask
                vocab_mask: "b8[1]" = org_vocab_mask | added_vocab_mask;  org_vocab_mask = added_vocab_mask = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:149 in get_masked_input_and_mask, code: input_ = vocab_mask * (input_ - valid_offset)
                sub: "i64[1]" = l_input_ids_ - valid_offset;  l_input_ids_ = valid_offset = None
                input_: "i64[1]" = vocab_mask * sub;  sub = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:150 in get_masked_input_and_mask, code: return input_, ~vocab_mask
                input_mask: "b8[1]" = ~vocab_mask;  vocab_mask = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:477 in forward, code: output_parallel = self.quant_method.embedding(self, masked_input.long())
                long: "i64[1]" = input_.long();  input_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/quantization/unquant.py:83 in embedding, code: return F.embedding(input_, layer.weight)
                output_parallel: "f16[1, 5120]" = torch.nn.functional.embedding(long, l_language_model_modules_model_modules_embed_tokens_parameters_weight_);  long = l_language_model_modules_model_modules_embed_tokens_parameters_weight_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/vocab_parallel_embedding.py:481 in forward, code: output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
                unsqueeze: "b8[1, 1]" = input_mask.unsqueeze(-1);  input_mask = None
                masked_fill_: "f16[1, 5120]" = output_parallel.masked_fill_(unsqueeze, 0);  unsqueeze = masked_fill_ = None
                return output_parallel
                
    class inductor_1(torch.nn.Module):
        def forward(self, output_parallel: "f16[1, 5120]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
            getattr_1 = output_parallel.is_cpu;  getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(output_parallel, group_name = 'tp:0');  output_parallel = inplace_all_reduce = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, output_parallel: "f16[1, 5120]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:493 in all_reduce, code: if input_.is_cpu:
                getattr_1 = output_parallel.is_cpu;  getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:550 in all_reduce, code: torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
                inplace_all_reduce = torch.ops.sglang.inplace_all_reduce(output_parallel, group_name = 'tp:0');  output_parallel = inplace_all_reduce = None
                return ()
                