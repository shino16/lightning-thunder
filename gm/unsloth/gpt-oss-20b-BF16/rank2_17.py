# Rank: 2, Graph 17

class GraphModule(torch.nn.Module):
    def forward(self, l_stack0_0_: "bf16[1, 4]", l_stack0_1_: "i16[1, 4]", l_stack0_2_storage_data: "u32[1, 1]", l_stack0_2_scratchpad: "i32[128]"):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0(l_stack0_2_scratchpad);  l_stack0_2_scratchpad = None
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        inductor_1 = self.inductor_1(l_stack0_2_storage_data, getitem, getitem_1);  l_stack0_2_storage_data = inductor_1 = None
        thunder_2 = self.thunder_2(getitem_1, getitem);  getitem_1 = getitem = None
        getitem_2 = thunder_2[0]
        getitem_3 = thunder_2[1]
        getitem_4 = thunder_2[2]
        getitem_5 = thunder_2[3]
        getitem_6 = thunder_2[4]
        getitem_7 = thunder_2[5]
        getitem_8 = thunder_2[6]
        getitem_9 = thunder_2[7]
        getitem_10 = thunder_2[8]
        getitem_11 = thunder_2[9]
        getitem_12 = thunder_2[10];  thunder_2 = None
        inductor_3 = self.inductor_3(getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, l_stack0_0_, l_stack0_1_, getitem_11);  getitem_2 = getitem_4 = getitem_5 = getitem_6 = l_stack0_0_ = l_stack0_1_ = inductor_3 = None
        thunder_4 = self.thunder_4(getitem_11, getitem_7);  getitem_11 = getitem_7 = None
        getitem_13 = thunder_4[0]
        getitem_14 = thunder_4[1]
        getitem_15 = thunder_4[2]
        getitem_16 = thunder_4[3]
        getitem_17 = thunder_4[4]
        getitem_18 = thunder_4[5]
        getitem_19 = thunder_4[6]
        getitem_20 = thunder_4[7];  thunder_4 = None
        return (getitem_3, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_10, getitem_9, getitem_8)
        
    class thunder_0(torch.nn.Module):
        def forward(self, l_stack0_2_scratchpad: "i32[128]"):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/tensor.py:183 in sum, code: out_ret = self.scratchpad[:n_cols]
            out_ret: "i32[32]" = l_stack0_2_scratchpad[slice(None, 32, None)];  l_stack0_2_scratchpad = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/reduction_details/reduce_bitmatrix.py:97 in sum_bitmatrix_rows, code: out_partials = torch.empty((pids_y * 32, pids_x * TILE_SIZE), device=out_ret.device, dtype=torch.int32)
            out_partials: "i32[32, 4]" = torch.empty((32, 4), device = device(type='cuda', index=2), dtype = torch.int32)
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/reduction_details/reduce_bitmatrix.py:98 in sum_bitmatrix_rows, code: out_partials = torch.transpose(out_partials, 0, 1)
            out_partials_1: "i32[4, 32]" = torch.transpose(out_partials, 0, 1);  out_partials = None
            return (out_ret, out_partials_1)
            
        class _model(torch.nn.Module):
            def forward(self, l_stack0_2_scratchpad: "i32[128]"):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/tensor.py:183 in sum, code: out_ret = self.scratchpad[:n_cols]
                out_ret: "i32[32]" = l_stack0_2_scratchpad[slice(None, 32, None)];  l_stack0_2_scratchpad = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/reduction_details/reduce_bitmatrix.py:97 in sum_bitmatrix_rows, code: out_partials = torch.empty((pids_y * 32, pids_x * TILE_SIZE), device=out_ret.device, dtype=torch.int32)
                out_partials: "i32[32, 4]" = torch.empty((32, 4), device = device(type='cuda', index=2), dtype = torch.int32)
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/reduction_details/reduce_bitmatrix.py:98 in sum_bitmatrix_rows, code: out_partials = torch.transpose(out_partials, 0, 1)
                out_partials_1: "i32[4, 32]" = torch.transpose(out_partials, 0, 1);  out_partials = None
                return (out_ret, out_partials_1)
                
    class inductor_1(torch.nn.Module):
        def forward(self, l_stack0_2_storage_data: "u32[1, 1]", out_ret: "i32[32]", out_partials_1: "i32[4, 32]"):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/reduction_details/reduce_bitmatrix.py:101 in sum_bitmatrix_rows, code: _sum_bitmatrix_rows[(pids_x, pids_y)](
            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 0, grid = [(1, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'B': l_stack0_2_storage_data, 'Ret': out_ret, 'Partials': out_partials_1});  l_stack0_2_storage_data = out_ret = out_partials_1 = triton_kernel_wrapper_mutation = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_stack0_2_storage_data: "u32[1, 1]", out_ret: "i32[32]", out_partials_1: "i32[4, 32]"):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/reduction_details/reduce_bitmatrix.py:101 in sum_bitmatrix_rows, code: _sum_bitmatrix_rows[(pids_x, pids_y)](
                triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 2, constant_args_idx = 0, grid = [(1, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'B': l_stack0_2_storage_data, 'Ret': out_ret, 'Partials': out_partials_1});  l_stack0_2_storage_data = out_ret = out_partials_1 = triton_kernel_wrapper_mutation = None
                return ()
                
    class thunder_2(torch.nn.Module):
        def forward(self, out_partials_1: "i32[4, 32]", out_ret: "i32[32]"):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/reduction_details/reduce_bitmatrix.py:109 in sum_bitmatrix_rows, code: out_partials = out_partials[:cdiv(n_rows_max, PARTIALS_BLOCK_M), :]
            out_partials_2: "i32[1, 32]" = out_partials_1[(slice(None, 1, None), slice(None, None, None))];  out_partials_1 = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:109 in forward, code: hist = hist[:n_expts_tot]
            hist: "i32[32]" = out_ret[slice(None, 32, None)];  out_ret = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:112 in forward, code: expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
            expt_offs: "i32[32]" = torch.empty(32, dtype = torch.int32, device = device(type='cuda', index=2))
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:113 in forward, code: combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32, device=device)
            combined_indx: "i32[8]" = torch.empty(8, dtype = torch.int32, device = device(type='cuda', index=2))
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:115 in forward, code: topk_indx = combined_indx[:n_gates_pad]
            topk_indx: "i32[4]" = combined_indx[slice(None, 4, None)]
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:116 in forward, code: gate_indx = combined_indx[n_gates_pad:]
            gate_indx: "i32[4]" = combined_indx[slice(4, None, None)]
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:117 in forward, code: gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)
            gate_scal: "bf16[4]" = torch.empty(4, dtype = torch.bfloat16, device = device(type='cuda', index=2))
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:226 in _compute_expt_data_internal, code: token_offs_combined = torch.empty((block_m_num + 1, pad(n_expts_tot + 1)), dtype=dtype, device=device)
            token_offs_combined: "i32[5, 512]" = torch.empty((5, 512), dtype = torch.int32, device = device(type='cuda', index=2))
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:228 in _compute_expt_data_internal, code: token_offs_raw = token_offs_combined[0][:n_expts_tot + 1]
            getitem_5: "i32[512]" = token_offs_combined[0]
            token_offs_raw: "i32[33]" = getitem_5[slice(None, 33, None)];  getitem_5 = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:229 in _compute_expt_data_internal, code: token_offs_pad = token_offs_combined[1:]
            token_offs_pad: "i32[4, 512]" = token_offs_combined[slice(1, None, None)]
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:231 in _compute_expt_data_internal, code: block_pid_map = torch.empty((block_m_num, pad(max_n_tiles)), dtype=dtype, device=device)
            block_pid_map: "i32[4, 512]" = torch.empty((4, 512), dtype = torch.int32, device = device(type='cuda', index=2))
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:234 in _compute_expt_data_internal, code: token_offs_pad = token_offs_pad[:, :n_expts_tot + 1]
            token_offs_pad_1: "i32[4, 33]" = token_offs_pad[(slice(None, None, None), slice(None, 33, None))];  token_offs_pad = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:235 in _compute_expt_data_internal, code: block_pid_map = block_pid_map[:, :max_n_tiles]
            block_pid_map_1: "i32[4, 4]" = block_pid_map[(slice(None, None, None), slice(None, 4, None))];  block_pid_map = None
            return (combined_indx, hist, expt_offs, out_partials_2, token_offs_combined, block_pid_map_1, topk_indx, gate_indx, gate_scal, token_offs_pad_1, token_offs_raw)
            
        class _model(torch.nn.Module):
            def forward(self, out_partials_1: "i32[4, 32]", out_ret: "i32[32]"):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/reduction_details/reduce_bitmatrix.py:109 in sum_bitmatrix_rows, code: out_partials = out_partials[:cdiv(n_rows_max, PARTIALS_BLOCK_M), :]
                out_partials_2: "i32[1, 32]" = out_partials_1[(slice(None, 1, None), slice(None, None, None))];  out_partials_1 = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:109 in forward, code: hist = hist[:n_expts_tot]
                hist: "i32[32]" = out_ret[slice(None, 32, None)];  out_ret = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:112 in forward, code: expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
                expt_offs: "i32[32]" = torch.empty(32, dtype = torch.int32, device = device(type='cuda', index=2))
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:113 in forward, code: combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32, device=device)
                combined_indx: "i32[8]" = torch.empty(8, dtype = torch.int32, device = device(type='cuda', index=2))
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:115 in forward, code: topk_indx = combined_indx[:n_gates_pad]
                topk_indx: "i32[4]" = combined_indx[slice(None, 4, None)]
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:116 in forward, code: gate_indx = combined_indx[n_gates_pad:]
                gate_indx: "i32[4]" = combined_indx[slice(4, None, None)]
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:117 in forward, code: gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)
                gate_scal: "bf16[4]" = torch.empty(4, dtype = torch.bfloat16, device = device(type='cuda', index=2))
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:226 in _compute_expt_data_internal, code: token_offs_combined = torch.empty((block_m_num + 1, pad(n_expts_tot + 1)), dtype=dtype, device=device)
                token_offs_combined: "i32[5, 512]" = torch.empty((5, 512), dtype = torch.int32, device = device(type='cuda', index=2))
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:228 in _compute_expt_data_internal, code: token_offs_raw = token_offs_combined[0][:n_expts_tot + 1]
                getitem_5: "i32[512]" = token_offs_combined[0]
                token_offs_raw: "i32[33]" = getitem_5[slice(None, 33, None)];  getitem_5 = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:229 in _compute_expt_data_internal, code: token_offs_pad = token_offs_combined[1:]
                token_offs_pad: "i32[4, 512]" = token_offs_combined[slice(1, None, None)]
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:231 in _compute_expt_data_internal, code: block_pid_map = torch.empty((block_m_num, pad(max_n_tiles)), dtype=dtype, device=device)
                block_pid_map: "i32[4, 512]" = torch.empty((4, 512), dtype = torch.int32, device = device(type='cuda', index=2))
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:234 in _compute_expt_data_internal, code: token_offs_pad = token_offs_pad[:, :n_expts_tot + 1]
                token_offs_pad_1: "i32[4, 33]" = token_offs_pad[(slice(None, None, None), slice(None, 33, None))];  token_offs_pad = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:235 in _compute_expt_data_internal, code: block_pid_map = block_pid_map[:, :max_n_tiles]
                block_pid_map_1: "i32[4, 4]" = block_pid_map[(slice(None, None, None), slice(None, 4, None))];  block_pid_map = None
                return (combined_indx, hist, expt_offs, out_partials_2, token_offs_combined, block_pid_map_1, topk_indx, gate_indx, gate_scal, token_offs_pad_1, token_offs_raw)
                
    class inductor_3(torch.nn.Module):
        def forward(self, combined_indx: "i32[8]", hist: "i32[32]", expt_offs: "i32[32]", out_partials_2: "i32[1, 32]", token_offs_combined: "i32[5, 512]", block_pid_map_1: "i32[4, 4]", topk_indx: "i32[4]", gate_indx: "i32[4]", gate_scal: "bf16[4]", l_stack0_0_: "bf16[1, 4]", l_stack0_1_: "i16[1, 4]", token_offs_pad_1: "i32[4, 33]"):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:125 in forward, code: _combined_routing_memset[(blocks1a + blocks1b, )](
            triton_kernel_wrapper_mutation_1 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 3, constant_args_idx = 1, grid = [(43, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'Indx': combined_indx, 'ExpertHist': hist, 'FinalExpertOffs': expt_offs, 'PartialHist': out_partials_2, 'MDStarts': token_offs_combined, 'MDTileInfo': block_pid_map_1});  combined_indx = token_offs_combined = triton_kernel_wrapper_mutation_1 = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:137 in forward, code: _combined_routing_compute[(blocks2a + blocks2b, )](
            triton_kernel_wrapper_mutation_2 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 4, constant_args_idx = 2, grid = [(129, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'GatherIndx': topk_indx, 'ScatterIndx': gate_indx, 'GateScal': gate_scal, 'ExptScal': l_stack0_0_, 'ExptIndx': l_stack0_1_, 'PartialOffs': out_partials_2, 'TokensStart': expt_offs, 'Hist': hist, 'MDTileStarts': token_offs_pad_1, 'MDTileInfo': block_pid_map_1});  topk_indx = gate_indx = gate_scal = l_stack0_0_ = l_stack0_1_ = out_partials_2 = expt_offs = hist = token_offs_pad_1 = block_pid_map_1 = triton_kernel_wrapper_mutation_2 = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, combined_indx: "i32[8]", hist: "i32[32]", expt_offs: "i32[32]", out_partials_2: "i32[1, 32]", token_offs_combined: "i32[5, 512]", block_pid_map_1: "i32[4, 4]", topk_indx: "i32[4]", gate_indx: "i32[4]", gate_scal: "bf16[4]", l_stack0_0_: "bf16[1, 4]", l_stack0_1_: "i16[1, 4]", token_offs_pad_1: "i32[4, 33]"):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:125 in forward, code: _combined_routing_memset[(blocks1a + blocks1b, )](
                triton_kernel_wrapper_mutation_1 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 3, constant_args_idx = 1, grid = [(43, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'Indx': combined_indx, 'ExpertHist': hist, 'FinalExpertOffs': expt_offs, 'PartialHist': out_partials_2, 'MDStarts': token_offs_combined, 'MDTileInfo': block_pid_map_1});  combined_indx = token_offs_combined = triton_kernel_wrapper_mutation_1 = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:137 in forward, code: _combined_routing_compute[(blocks2a + blocks2b, )](
                triton_kernel_wrapper_mutation_2 = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 4, constant_args_idx = 2, grid = [(129, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'GatherIndx': topk_indx, 'ScatterIndx': gate_indx, 'GateScal': gate_scal, 'ExptScal': l_stack0_0_, 'ExptIndx': l_stack0_1_, 'PartialOffs': out_partials_2, 'TokensStart': expt_offs, 'Hist': hist, 'MDTileStarts': token_offs_pad_1, 'MDTileInfo': block_pid_map_1});  topk_indx = gate_indx = gate_scal = l_stack0_0_ = l_stack0_1_ = out_partials_2 = expt_offs = hist = token_offs_pad_1 = block_pid_map_1 = triton_kernel_wrapper_mutation_2 = None
                return ()
                
    class thunder_4(torch.nn.Module):
        def forward(self, token_offs_pad_1: "i32[4, 33]", block_pid_map_1: "i32[4, 4]"):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:245 in _unpack_into_dict, code: x = {2**j: x[i, :] for i, j in enumerate(range(block_m_log2_start, block_m_log2_end))}
            v: "i32[33]" = token_offs_pad_1[(0, slice(None, None, None))]
            v_1: "i32[33]" = token_offs_pad_1[(1, slice(None, None, None))]
            v_2: "i32[33]" = token_offs_pad_1[(2, slice(None, None, None))]
            v_3: "i32[33]" = token_offs_pad_1[(3, slice(None, None, None))];  token_offs_pad_1 = None
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:245 in _unpack_into_dict, code: x = {2**j: x[i, :] for i, j in enumerate(range(block_m_log2_start, block_m_log2_end))}
            v_4: "i32[4]" = block_pid_map_1[(0, slice(None, None, None))]
            v_5: "i32[4]" = block_pid_map_1[(1, slice(None, None, None))]
            v_6: "i32[4]" = block_pid_map_1[(2, slice(None, None, None))]
            v_7: "i32[4]" = block_pid_map_1[(3, slice(None, None, None))];  block_pid_map_1 = None
            return (v, v_1, v_2, v_3, v_4, v_5, v_6, v_7)
            
        class _model(torch.nn.Module):
            def forward(self, token_offs_pad_1: "i32[4, 33]", block_pid_map_1: "i32[4, 4]"):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:245 in _unpack_into_dict, code: x = {2**j: x[i, :] for i, j in enumerate(range(block_m_log2_start, block_m_log2_end))}
                v: "i32[33]" = token_offs_pad_1[(0, slice(None, None, None))]
                v_1: "i32[33]" = token_offs_pad_1[(1, slice(None, None, None))]
                v_2: "i32[33]" = token_offs_pad_1[(2, slice(None, None, None))]
                v_3: "i32[33]" = token_offs_pad_1[(3, slice(None, None, None))];  token_offs_pad_1 = None
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/routing.py:245 in _unpack_into_dict, code: x = {2**j: x[i, :] for i, j in enumerate(range(block_m_log2_start, block_m_log2_end))}
                v_4: "i32[4]" = block_pid_map_1[(0, slice(None, None, None))]
                v_5: "i32[4]" = block_pid_map_1[(1, slice(None, None, None))]
                v_6: "i32[4]" = block_pid_map_1[(2, slice(None, None, None))]
                v_7: "i32[4]" = block_pid_map_1[(3, slice(None, None, None))];  block_pid_map_1 = None
                return (v, v_1, v_2, v_3, v_4, v_5, v_6, v_7)
                