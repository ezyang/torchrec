from torchrec.distributed.comm_ops import PropagatingAsyncCollectiveTensor
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

fake_mode = FakeTensorMode()
result = torch.zeros(20, 20)

def thunk():
    print("evaled")
    result.fill_(2)
    return result

x = PropagatingAsyncCollectiveTensor(thunk, fake_mode.from_tensor(result))
y = x + 2
print(y)
print("ready")
print(y.tolist())
