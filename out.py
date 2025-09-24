class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[128, 128]"):
        l_x_ = L_x_
        
         # File: /opt/pytorch/lightning-thunder/main.py:19 in checkpoint_fn, code: return checkpoint(fn, x, use_reentrant=False)
        wrap_body_0 = self.wrap_body_0
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
        getitem: "f32[128, 128]" = tag_activation_checkpoint[0];  tag_activation_checkpoint = None
        return (getitem,)
        
    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[128, 128]"):
             # File: /opt/pytorch/lightning-thunder/main.py:15 in fn, code: w = x.sin()
            w: "f32[128, 128]" = l_x_.sin();  l_x_ = None
            
             # File: /opt/pytorch/lightning-thunder/main.py:16 in fn, code: return (w @ w).sin()
            matmul: "f32[128, 128]" = w @ w;  w = None
            sin_1: "f32[128, 128]" = matmul.sin();  matmul = None
            return (sin_1,)
            
# Constructed by Debug trace (took 0.13 milliseconds)
import torch
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def computation(l_x_):
  # l_x_: "cuda:0 f32[128, 128]"

  # <eval_with_key>.3:6: 	    tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
  t37 = torch.sin(l_x_)  # t37: "cuda:0 f32[128, 128]"
    # t37 = ltorch.sin(l_x_)  # t37: "cuda:0 f32[128, 128]"
      # t37 = prims.sin(l_x_)  # t37: "cuda:0 f32[128, 128]"
  # alloc - 65536 bytes
  # total 131072 bytes allocated
  debug_post_sin1(t37, l_x_)

  # <eval_with_key>.3:6: 	    tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
  t38 = torch.matmul(t37, t37)  # t38: "cuda:0 f32[128, 128]"
    # t38 = ltorch.matmul(t37, t37)  # t38: "cuda:0 f32[128, 128]"
      # t38 = prims.matmul(t37, t37)  # t38: "cuda:0 f32[128, 128]"
  # alloc - 65536 bytes
  # segment_alloc - 33554432 bytes
  # alloc - 33554432 bytes
  # total 33751040 bytes allocated
  debug_post_matmul2(t38, t37, t37)
  del t37
  # free_requested - 65536 bytes
  # free_completed - 65536 bytes
  # total 33685504 bytes allocated
  debug_post_python_del3(None)

  # <eval_with_key>.3:6: 	    tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
  t39 = torch.sin(t38)  # t39: "cuda:0 f32[128, 128]"
    # t39 = ltorch.sin(t38)  # t39: "cuda:0 f32[128, 128]"
      # t39 = prims.sin(t38)  # t39: "cuda:0 f32[128, 128]"
  # alloc - 65536 bytes
  # total 33751040 bytes allocated
  debug_post_sin4(t39, t38)
  del t38
  # free_requested - 65536 bytes
  # free_completed - 65536 bytes
  # total 33685504 bytes allocated
  debug_post_python_del5(None)
  t40 = shallow_copy(t39)  # t40: "cuda:0 f32[128, 128]"
  # 
  # total 33685504 bytes allocated
  debug_post_shallow_copy6(t40, t39)
  del t39
  # 
  # total 33685504 bytes allocated
  debug_post_python_del7(None)
  return {'output': (t40,), 'flat_args': [l_x_], 'flat_output': (t40,)}, ((l_x_,), ())
32.125 MB
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[128, 128]"):
        l_x_ = L_x_
        
         # File: /opt/pytorch/lightning-thunder/main.py:19 in checkpoint_fn, code: return checkpoint(fn, x, use_reentrant=False)
        wrap_body_0 = self.wrap_body_0
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
        getitem: "f32[128, 128]" = tag_activation_checkpoint[0];  tag_activation_checkpoint = None
        return (getitem,)
        
    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[128, 128]"):
             # File: /opt/pytorch/lightning-thunder/main.py:15 in fn, code: w = x.sin()
            w: "f32[128, 128]" = l_x_.sin();  l_x_ = None
            
             # File: /opt/pytorch/lightning-thunder/main.py:16 in fn, code: return (w.mT @ w).sin()
            getattr_1: "f32[128, 128]" = w.mT
            matmul: "f32[128, 128]" = getattr_1 @ w;  getattr_1 = w = None
            sin_1: "f32[128, 128]" = matmul.sin();  matmul = None
            return (sin_1,)
            
# Constructed by Unwrap the actual return value
import torch
from thunder.executors.torchex import no_autocast

@torch.no_grad()
@no_autocast
def computation(getitem):
  # getitem: "cuda:0 f32[128, 128]"
  return (getitem,)
class GraphModule(torch.nn.Module):
    def forward(self, l_x_: "f32[128, 128]"):
         # File: /opt/pytorch/lightning-thunder/main.py:19 in checkpoint_fn, code: return checkpoint(fn, x, use_reentrant=False)
        wrap_body_0 = self.wrap_body_0
        
         # File: <debug>:0 in <debug>, code: memory_events = [], total 65536 bytes allocated
        memory_events_wrap_body_0 = thunder_dev_utils_debug_memory_transform_memory_events_wrap_body_0();  memory_events_wrap_body_0 = None
        memory_events_l_x_ = thunder_dev_utils_debug_memory_transform_memory_events_l_x_();  memory_events_l_x_ = None
        
         # File: /opt/pytorch/lightning-thunder/main.py:19 in checkpoint_fn, code: return checkpoint(fn, x, use_reentrant=False)
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
        
         # File: <debug>:0 in <debug>, code: memory_events = [], total 33816576 bytes allocated
        memory_events_tag_activation_checkpoint = thunder_dev_utils_debug_memory_transform_memory_events_tag_activation_checkpoint();  memory_events_tag_activation_checkpoint = None
        return tag_activation_checkpoint
        
        # No stacktrace found for following nodes
        memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None
        
    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[128, 128]"):
             # File: <debug>:0 in <debug>, code: memory_events = [], total 65536 bytes allocated
            memory_events_l_x_ = thunder_dev_utils_debug_memory_transform_memory_events_l_x_();  memory_events_l_x_ = None
            
             # File: /opt/pytorch/lightning-thunder/main.py:15 in fn, code: w = x.sin()
            w: "f32[128, 128]" = l_x_.sin();  l_x_ = None
            
             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 131072 bytes allocated
            memory_events_w = thunder_dev_utils_debug_memory_transform_memory_events_w();  memory_events_w = None
            
             # File: /opt/pytorch/lightning-thunder/main.py:16 in fn, code: return (w.mT @ w).sin()
            getattr_1: "f32[128, 128]" = w.mT
            
             # File: <debug>:0 in <debug>, code: memory_events = [], total 131072 bytes allocated
            memory_events_getattr_1 = thunder_dev_utils_debug_memory_transform_memory_events_getattr_1();  memory_events_getattr_1 = None
            
             # File: /opt/pytorch/lightning-thunder/main.py:16 in fn, code: return (w.mT @ w).sin()
            matmul: "f32[128, 128]" = getattr_1 @ w;  getattr_1 = w = None
            
             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes, segment_alloc - 33554432 bytes, alloc - 33554432 bytes], total 33751040 bytes allocated
            memory_events_matmul = thunder_dev_utils_debug_memory_transform_memory_events_matmul();  memory_events_matmul = None
            
             # File: /opt/pytorch/lightning-thunder/main.py:16 in fn, code: return (w.mT @ w).sin()
            sin_1: "f32[128, 128]" = matmul.sin();  matmul = None
            
             # File: <debug>:0 in <debug>, code: memory_events = [alloc - 65536 bytes], total 33816576 bytes allocated
            memory_events_sin_1 = thunder_dev_utils_debug_memory_transform_memory_events_sin_1();  memory_events_sin_1 = None
            return (sin_1,)
            
            # No stacktrace found for following nodes
            memory_events_output = thunder_dev_utils_debug_memory_transform_memory_events_output();  memory_events_output = None
            
32.1875 MB
