import random
import numpy as np
import torch
import os
from utils import get_score_func
from cifar10loader import trainloader

from XAutoDL.xautodl.models import get_cell_based_tiny_net
from nats_bench import create

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
BATCHSIZE = 64

def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
api = create(None, 'tss', fast_mode=True, verbose=True)
arch_idx = [random.randint(1, len(api)) for _ in range(2)]
scores = np.zeros(len(arch_idx))
for i, idx in enumerate(arch_idx):
    try:
        arch = api.get_net_config(idx, 'cifar10')
        arch['num_classes'] = 1
        network = get_cell_based_tiny_net(arch)
        print(network)
        network.K = np.zeros((BATCHSIZE, BATCHSIZE))
        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1.-x) @ (1.-x.t())
                network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
            except:
                pass

            
        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

            
        for name, module in network.named_modules():
            if 'ReLU' in str(type(module)):
                #hooks[name] = module.register_forward_hook(counting_hook)
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)

        network = network.to(device)
        s = []
        maxofn = 1
        for j in range(maxofn):
            data_iterator = iter(trainloader)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, device)
            network(x2.to(device))
            s.append(get_score_func('hook_logdet')(network.K, target))
        scores[i] = np.mean(s)
    except Exception as e:
        print(e)
        scores[i] = np.nan
print(scores)