import random
import os
import torch
import numpy as np
import pandas as pd

from naswot import return_score
from cifar10loader import trainloader
from utils import add_dropout, init_network
from XAutoDL.xautodl.models import get_cell_based_tiny_net
from nats_bench import create



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
sigma = 0.05 # default from NASWOT
init = '' # default from NASWOT

api = create(None, 'tss', fast_mode=True, verbose=True)
searchspace = [random.randint(1, len(api)) for _ in range(1000)]
scores = np.zeros(len(searchspace))
accuracy = np.zeros(len(searchspace))
for i, idx in enumerate(searchspace):
    arch = api.get_net_config(idx, 'cifar10')
    arch['num_classes'] = 1
    network = get_cell_based_tiny_net(arch)
    add_dropout(network, sigma)
    init_network(network, init)

    score = return_score(network, trainloader)
    # print(score)
    info = api.get_more_info(idx, 'cifar10')
    test_acc = info['test-accuracy']
    # print(info['test-accuracy'])
    scores[i] = score
    accuracy[i] = test_acc
data = {
    'scores': scores,
    'accuracy': accuracy
}
df = pd.DataFrame(data)
df.to_csv('output.csv', encoding='utf-8', index=False)