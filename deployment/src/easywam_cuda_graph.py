"""Opt-in CUDA graph execution for fixed-shape EasyWAM joint denoising.

This changes attention/compiler kernels and therefore floating-point results.
The baseline deployment path remains the default.
"""
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, register_pytree_node
from model.component.attention import KeyPaddingMask

register_pytree_node(KeyPaddingMask,
 lambda x: ([x.valid, x.indices, x.cu_seqlens], x.max_seqlen),
 lambda values, context: KeyPaddingMask(*values, context))

class GraphDenoise:
 def __init__(self,fn):
  self.fn=fn;self.graph=None;self.replays=0
 def __call__(self,**kwargs):
  flat,spec=tree_flatten(kwargs)
  if self.graph is None:
   self.spec=spec
   self.static=[v.clone() if isinstance(v,torch.Tensor) else v for v in flat]
   static_kw=tree_unflatten(self.static,spec)
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):self.fn(**static_kw)
   torch.cuda.current_stream().wait_stream(stream)
   self.graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph):
    self.output=self.fn(**static_kw)
  else:
   if spec!=self.spec:raise ValueError("CUDA Graph input structure changed")
   for dst,src in zip(self.static,flat):
    if isinstance(dst,torch.Tensor):
     if (dst.shape,dst.dtype,dst.device)!=(src.shape,src.dtype,src.device):raise ValueError("CUDA Graph tensor signature changed")
     dst.copy_(src)
    elif dst!=src:raise ValueError("CUDA Graph scalar changed")
  self.graph.replay()
  self.replays+=1
  return tuple(x.clone() for x in self.output)
def install_cuda_graph_execution(model, device):
    from model.helpers.inference import configure_inference_compile
    torch._dynamo.config.cache_size_limit = 64
    for module in model.modules():
        if hasattr(module, "attention_backend"):
            module.attention_backend = "fa2"
    model.action_expert.freqs = model.action_expert.freqs.to(device)
    configure_inference_compile(model, enabled=True, mode="default",
                                fullgraph=False, dynamic=False)
    graph = GraphDenoise(model._predict_joint_noise)
    model._predict_joint_noise = graph
    return graph
