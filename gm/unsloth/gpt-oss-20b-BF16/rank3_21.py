# Rank: 3, Graph 21

class GraphModule(torch.nn.Module):
    def forward(self, l_args_0_: "bf16[1, 1, 4, 720]", l_args_1_: "f32[11, 1, 4, 1440]", l_args_12_: "bf16[1, 2880]", l_args_22_: "bf16[32, 2880, 1440]", l_args_32_: "f32[32, 1440]", l_args_40_: "i32[4]", l_args_45_: "i32[32]", l_args_46_: "i32[33]", l_args_47_: "i32[]", l_args_48_: "i32[4]"):
        # No stacktrace found for following nodes
        inductor_0 = self.inductor_0(l_args_0_, l_args_1_, l_args_12_, l_args_22_, l_args_32_, l_args_40_, l_args_45_, l_args_46_, l_args_47_, l_args_48_);  l_args_0_ = l_args_1_ = l_args_12_ = l_args_22_ = l_args_32_ = l_args_40_ = l_args_45_ = l_args_46_ = l_args_47_ = l_args_48_ = inductor_0 = None
        return ()
        
    class inductor_0(torch.nn.Module):
        def forward(self, l_args_0_: "bf16[1, 1, 4, 720]", l_args_1_: "f32[11, 1, 4, 1440]", l_args_12_: "bf16[1, 2880]", l_args_22_: "bf16[32, 2880, 1440]", l_args_32_: "f32[32, 1440]", l_args_40_: "i32[4]", l_args_45_: "i32[32]", l_args_46_: "i32[33]", l_args_47_: "i32[]", l_args_48_: "i32[4]"):
             # File: /usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py:390 in <lambda>, code: return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 6, grid = [(264, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'Y': l_args_0_, 'Out': l_args_1_, 'X': l_args_12_, 'XPtr': l_args_12_, 'W': l_args_22_, 'B': l_args_32_, 'GatherIndx': l_args_40_, 'ExptHist': l_args_45_, 'ExptOffs': l_args_46_, 'ExptOffsSum': l_args_47_, 'ExptData': l_args_48_});  l_args_0_ = l_args_1_ = l_args_12_ = l_args_22_ = l_args_32_ = l_args_40_ = l_args_45_ = l_args_46_ = l_args_47_ = l_args_48_ = triton_kernel_wrapper_mutation = None
            return ()
            
        class _orig_mod(torch.nn.Module):
            def forward(self, l_args_0_: "bf16[1, 1, 4, 720]", l_args_1_: "f32[11, 1, 4, 1440]", l_args_12_: "bf16[1, 2880]", l_args_22_: "bf16[32, 2880, 1440]", l_args_32_: "f32[32, 1440]", l_args_40_: "i32[4]", l_args_45_: "i32[32]", l_args_46_: "i32[33]", l_args_47_: "i32[]", l_args_48_: "i32[4]"):
                 # File: /usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py:390 in <lambda>, code: return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 7, constant_args_idx = 6, grid = [(264, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'Y': l_args_0_, 'Out': l_args_1_, 'X': l_args_12_, 'XPtr': l_args_12_, 'W': l_args_22_, 'B': l_args_32_, 'GatherIndx': l_args_40_, 'ExptHist': l_args_45_, 'ExptOffs': l_args_46_, 'ExptOffsSum': l_args_47_, 'ExptData': l_args_48_});  l_args_0_ = l_args_1_ = l_args_12_ = l_args_22_ = l_args_32_ = l_args_40_ = l_args_45_ = l_args_46_ = l_args_47_ = l_args_48_ = triton_kernel_wrapper_mutation = None
                return ()
                