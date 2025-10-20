output_parallel.is_cpu
torch.cuda.get_device_capability(0)
torch.cuda.get_device_capability(1)
torch.cuda.get_device_capability(2)
torch.cuda.get_device_capability(3)
torch.ops.sglang.inplace_all_reduce(output_parallel, group_name = 'tp:0')
torch.ops.sglang.reg_all_gather_into_tensor(output_tensor, logits, group_name = 'tp:0')
