# Rank: 0, Graph 10

class GraphModule(torch.nn.Module):
    def forward(self, l_forward_batch_next_token_logits_buffer: "f32[1, 131072]", l_self_modules_lm_head_parameters_weight_: "bf16[32768, 4096]", l_stack0_: "bf16[1, 4096]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_, l_self_modules_lm_head_parameters_weight_);  l_stack0_ = l_self_modules_lm_head_parameters_weight_ = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        inductor_1 = self.inductor_1(getitem, getitem_1);  getitem = inductor_1 = None
        thunder_2 = self.thunder_2(getitem_1, l_forward_batch_next_token_logits_buffer);  getitem_1 = l_forward_batch_next_token_logits_buffer = thunder_2 = None
        return ()
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_: "bf16[1, 4096]", l_self_modules_lm_head_parameters_weight_: "bf16[32768, 4096]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/logits_processor.py:473 in _get_logits, code: hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
            to: "bf16[1, 4096]" = l_stack0_.to(torch.bfloat16);  l_stack0_ = None
            getattr_1: "bf16[4096, 32768]" = l_self_modules_lm_head_parameters_weight_.T;  l_self_modules_lm_head_parameters_weight_ = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/logits_processor.py:472 in _get_logits, code: logits = torch.matmul(
            logits: "bf16[1, 32768]" = torch.matmul(to, getattr_1);  to = getattr_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:695 in all_gather, code: output_tensor = torch.empty(
            output_tensor: "bf16[4, 32768]" = torch.empty((4, 32768), dtype = torch.bfloat16, device = device(type='cuda', index=0))
            return (logits, output_tensor)
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_: "bf16[1, 4096]", l_self_modules_lm_head_parameters_weight_: "bf16[32768, 4096]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/logits_processor.py:473 in _get_logits, code: hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                to: "bf16[1, 4096]" = l_stack0_.to(torch.bfloat16);  l_stack0_ = None
                getattr_1: "bf16[4096, 32768]" = l_self_modules_lm_head_parameters_weight_.T;  l_self_modules_lm_head_parameters_weight_ = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/logits_processor.py:472 in _get_logits, code: logits = torch.matmul(
                logits: "bf16[1, 32768]" = torch.matmul(to, getattr_1);  to = getattr_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:695 in all_gather, code: output_tensor = torch.empty(
                output_tensor: "bf16[4, 32768]" = torch.empty((4, 32768), dtype = torch.bfloat16, device = device(type='cuda', index=0))
                return (logits, output_tensor)
                
    class inductor_1(torch.nn.Module):
        def forward(self, logits: "bf16[1, 32768]", output_tensor: "bf16[4, 32768]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:700 in all_gather, code: if input_.is_cpu and is_shm_available(
            getattr_2 = logits.is_cpu;  getattr_2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:705 in all_gather, code: if input_.is_cpu:
            getattr_3 = logits.is_cpu;  getattr_3 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:643 in all_gather_into_tensor, code: torch.ops.sglang.reg_all_gather_into_tensor(
            reg_all_gather_into_tensor = torch.ops.sglang.reg_all_gather_into_tensor(output_tensor, logits, group_name = 'tp:0');  output_tensor = logits = reg_all_gather_into_tensor = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, logits: "bf16[1, 32768]", output_tensor: "bf16[4, 32768]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:700 in all_gather, code: if input_.is_cpu and is_shm_available(
                getattr_2 = logits.is_cpu;  getattr_2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:705 in all_gather, code: if input_.is_cpu:
                getattr_3 = logits.is_cpu;  getattr_3 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:643 in all_gather_into_tensor, code: torch.ops.sglang.reg_all_gather_into_tensor(
                reg_all_gather_into_tensor = torch.ops.sglang.reg_all_gather_into_tensor(output_tensor, logits, group_name = 'tp:0');  output_tensor = logits = reg_all_gather_into_tensor = None
                return ()
                
    class thunder_2(torch.nn.Module):
        def forward(self, output_tensor: "bf16[4, 32768]", l_forward_batch_next_token_logits_buffer: "f32[1, 131072]"):
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:713 in all_gather, code: output_tensor = output_tensor.reshape((world_size,) + input_size)
            output_tensor_1: "bf16[4, 1, 32768]" = output_tensor.reshape((4, 1, 32768));  output_tensor = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:714 in all_gather, code: output_tensor = output_tensor.movedim(0, dim)
            output_tensor_2: "bf16[1, 4, 32768]" = output_tensor_1.movedim(0, 1);  output_tensor_1 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:715 in all_gather, code: output_tensor = output_tensor.reshape(
            output_tensor_3: "bf16[1, 131072]" = output_tensor_2.reshape((1, 131072));  output_tensor_2 = None
            
             # File: /opt/sglang/sglang-src/python/sglang/srt/layers/logits_processor.py:528 in _get_logits, code: logits_buffer.copy_(logits[:, : self.config.vocab_size])
            getitem: "bf16[1, 131072]" = output_tensor_3[(slice(None, None, None), slice(None, 131072, None))];  output_tensor_3 = None
            copy_: "f32[1, 131072]" = l_forward_batch_next_token_logits_buffer.copy_(getitem);  l_forward_batch_next_token_logits_buffer = getitem = copy_ = None
            return ()
            
        class _model(torch.nn.Module):
            def forward(self, output_tensor: "bf16[4, 32768]", l_forward_batch_next_token_logits_buffer: "f32[1, 131072]"):
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:713 in all_gather, code: output_tensor = output_tensor.reshape((world_size,) + input_size)
                output_tensor_1: "bf16[4, 1, 32768]" = output_tensor.reshape((4, 1, 32768));  output_tensor = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:714 in all_gather, code: output_tensor = output_tensor.movedim(0, dim)
                output_tensor_2: "bf16[1, 4, 32768]" = output_tensor_1.movedim(0, 1);  output_tensor_1 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/distributed/parallel_state.py:715 in all_gather, code: output_tensor = output_tensor.reshape(
                output_tensor_3: "bf16[1, 131072]" = output_tensor_2.reshape((1, 131072));  output_tensor_2 = None
                
                 # File: /opt/sglang/sglang-src/python/sglang/srt/layers/logits_processor.py:528 in _get_logits, code: logits_buffer.copy_(logits[:, : self.config.vocab_size])
                getitem: "bf16[1, 131072]" = output_tensor_3[(slice(None, None, None), slice(None, 131072, None))];  output_tensor_3 = None
                copy_: "f32[1, 131072]" = l_forward_batch_next_token_logits_buffer.copy_(getitem);  l_forward_batch_next_token_logits_buffer = getitem = copy_ = None
                return ()
                