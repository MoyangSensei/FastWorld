"""Common score-graph execution: deployment engineering, no new weights."""
import torch
from torch import nn

class TensorGraphCallFastWorld(nn.Module):
    def __init__(self,core):super().__init__();self.core=core;self.graph=None
    @torch.inference_mode()
    def forward(self,*args):
        if not all(isinstance(value,torch.Tensor) for value in args):raise TypeError('Tensor-only graph inputs')
        signature=tuple((tuple(value.shape),value.dtype,value.device) for value in args)
        if self.graph is None:
            self.signature=signature;self.inputs=[value.clone() for value in args]
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.core(*self.inputs)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.outputs=self.core(*self.inputs)
        if signature!=self.signature:raise ValueError('Graph input shape changed')
        for destination,value in zip(self.inputs,args):destination.copy_(value)
        self.graph.replay();return self.outputs

class FeatureGraphCore(nn.Module):
    def __init__(self, core, keys):
        super().__init__()
        self.core = core
        self.keys = keys

    def forward(self, *values):
        return self.core(dict(zip(self.keys, values)))


class FeatureGraphPlanner(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.executor = None

    def forward(self, features):
        keys = tuple(sorted(features))
        if self.executor is None:
            self.keys = keys
            self.executor = TensorGraphCallFastWorld(FeatureGraphCore(self.core, keys))
        if keys != self.keys:
            raise ValueError("FastWorld's input features changed after CUDA Graph capture.")
        return self.executor(*(features[key] for key in keys))
