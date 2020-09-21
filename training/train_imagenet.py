import torchvision.transforms as transforms
import torchbearer
from torchbearer import Trial, callbacks
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from imagenet_hdf5 import ImageNetHDF5
# from torchvision.datasets import ImageNet
from model import RetinalBottleneckModel

from sklearn.model_selection import ParameterGrid

import argparse
import os
import re

parser = argparse.ArgumentParser(description='Imagenet Training')
parser.add_argument('--arr', default=0, type=int, help='point in job array')
# parser.add_argument('--d-vvs', default=2, type=int, help='ventral depth')
# parser.add_argument('--cache', default=250, type=int, help='cache size')
parser.add_argument('--root', type=str, help='root')
args = parser.parse_args()

bottlenecks = [1, 2, 4, 8, 16, 32]

n_trials = 5

param_grid = ParameterGrid({
    'n_bn': bottlenecks,
    'a': list(range(n_trials))
})

params = param_grid[args.arr]
n_bn = params['n_bn']
rep = params['a']

# n_bn = bns[args.arr % 6]
# rep = args.arr // 6

dir = '/scratch/ewah1g13/models/'
model_file = f'resnet50_{n_bn}_{rep}'
# log_file = f'./logs/imagenet/resnet50_{n_bn}_{rep}.csv'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    # transforms.Grayscale(),
    # transforms.RandomAffine(0, translate=(0.1, 0.1)),
    # transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(128),
    # transforms.ToTensor()  # convert to tensor
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
    #transforms.Grayscale(),
    # transforms.CenterCrop(224),
    # transforms.Resize(128),
    # transforms.ToTensor()  # convert to tensor
])

# load data
trainset = ImageNetHDF5(f'{args.root}/train', transform=train_transform, cache_size=1000)
testset = ImageNetHDF5(f'{args.root}/val', transform=test_transform, cache_size=1000)

# create data loaders
trainloader = DataLoader(trainset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=16)
testloader = DataLoader(testset, batch_size=1024, shuffle=False,  pin_memory=True, num_workers=16)

# model = ImageNetModel(n_bn, args.d_vvs, n_inch=3)
model = RetinalBottleneckModel(n_bn, 'resnet50', n_out=1000, n_inch=3, retina_kernel_size=7)
model = nn.DataParallel(model)
# print(model)

optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)

# optimiser = optim.RMSprop(model.parameters(), alpha=0.9, lr=0.0001, weight_decay=1e-6)
loss_function = nn.CrossEntropyLoss()

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimizer, loss_function, metrics=['loss', 'acc', 'top_5_acc'],
              callbacks=[callbacks.TensorBoard(write_graph=False, comment=f'resnet50_{n_bn}_{rep}'), callbacks.MultiStepLR([30, 60]), callbacks.MostRecent(dir + model_file + '_{epoch:02d}.pt')]).to('cuda')
trial.with_generators(trainloader, test_generator=testloader)

pattern = re.compile(model_file + '_\d+.pt')
for filepath in os.listdir(dir):
    if pattern.match(filepath):
        trial.load_state_dict(torch.load(dir + filepath))

trial.run(epochs=90)
trial.evaluate(data_key=torchbearer.TEST_DATA)

torch.save(model.module.state_dict(), dir + model_file + '.pt')
