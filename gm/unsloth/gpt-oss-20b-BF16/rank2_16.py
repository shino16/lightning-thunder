# Rank: 2, Graph 16

class GraphModule(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        thunder_0 = self.thunder_0()
        getitem = thunder_0[0]
        getitem_1 = thunder_0[1];  thunder_0 = None
        inductor_1 = self.inductor_1()
        thunder_2 = self.thunder_2()
        return (getitem, getitem_1, inductor_1, thunder_2)
        
    class thunder_0(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:26 in topk_forward, code: y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
            y_vals: "bf16[1, 4]" = torch.empty((1, 4), dtype = torch.bfloat16, device = device(type='cuda', index=2))
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:30 in topk_forward, code: y_indx = torch.empty((n_rows_max, k), dtype=torch.int16, device=dev)
            y_indx: "i16[1, 4]" = torch.empty((1, 4), dtype = torch.int16, device = device(type='cuda', index=2))
            return (y_indx, y_vals)
            
        class _model(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:26 in topk_forward, code: y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
                y_vals: "bf16[1, 4]" = torch.empty((1, 4), dtype = torch.bfloat16, device = device(type='cuda', index=2))
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:30 in topk_forward, code: y_indx = torch.empty((n_rows_max, k), dtype=torch.int16, device=dev)
                y_indx: "i16[1, 4]" = torch.empty((1, 4), dtype = torch.int16, device = device(type='cuda', index=2))
                return (y_indx, y_vals)
                
    class inductor_1(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:35 in topk_forward, code: bitmatrix = torch.empty((n_cols_words, cdiv(n_rows_max, 32) * 32), dtype=torch.uint32, device=dev)
            bitmatrix: "u32[1, 32]" = torch.empty((1, 32), dtype = torch.uint32, device = device(type='cuda', index=2))
            
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:36 in topk_forward, code: bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows_max]
            transpose: "u32[32, 1]" = torch.transpose(bitmatrix, 0, 1);  bitmatrix = None
            bitmatrix_1: "u32[1, 1]" = transpose[slice(None, 1, None)];  transpose = None
            return bitmatrix_1
            
        class _orig_mod(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:35 in topk_forward, code: bitmatrix = torch.empty((n_cols_words, cdiv(n_rows_max, 32) * 32), dtype=torch.uint32, device=dev)
                bitmatrix: "u32[1, 32]" = torch.empty((1, 32), dtype = torch.uint32, device = device(type='cuda', index=2))
                
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:36 in topk_forward, code: bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows_max]
                transpose: "u32[32, 1]" = torch.transpose(bitmatrix, 0, 1);  bitmatrix = None
                bitmatrix_1: "u32[1, 1]" = transpose[slice(None, 1, None)];  transpose = None
                return bitmatrix_1
                
    class thunder_2(torch.nn.Module):
        def forward(self):
             # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:39 in topk_forward, code: scratchpad = torch.empty((s_cols, ), dtype=torch.int32, device=dev)
            scratchpad: "i32[128]" = torch.empty((128,), dtype = torch.int32, device = device(type='cuda', index=2))
            return scratchpad
            
        class _model(torch.nn.Module):
            def forward(self):
                 # File: /usr/local/lib/python3.12/dist-packages/triton_kernels/topk.py:39 in topk_forward, code: scratchpad = torch.empty((s_cols, ), dtype=torch.int32, device=dev)
                scratchpad: "i32[128]" = torch.empty((128,), dtype = torch.int32, device = device(type='cuda', index=2))
                return scratchpad
                