# !pip install foolbox
import torch
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np
import json

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from training.imagenet_hdf5 import ImageNetHDF5, ImageNetHDF5Subset
from training.model import RetinalBottleneckModel

from sklearn.model_selection import ParameterGrid

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Imagenet Training')
parser.add_argument('--arr', default=0, type=int, help='point in job array')
# parser.add_argument('--d-vvs', default=2, type=int, help='ventral depth')
# parser.add_argument('--cache', default=250, type=int, help='cache size')
parser.add_argument('--root', type=str, help='root')
parser.add_argument('--subset', type=bool, default=False)
args = parser.parse_args()

bottlenecks = [1, 2, 4, 8, 16, 32]

n_trials = 5

attacks = [
    fa.FGSM(),
    fa.LinfPGD(),
    fa.LinfBasicIterativeAttack(),
    fa.LinfAdditiveUniformNoiseAttack(),
    fa.LinfDeepFoolAttack(),
]

param_grid = ParameterGrid({
    'n_bn': bottlenecks,
    'a': list(range(n_trials)),
    'method': list(range(len(attacks)))
})

params = param_grid[args.arr]
n_bn = params['n_bn']
rep = params['a']
attack_index = params['method']
attack = attacks[attack_index]

# n_bn = bns[args.arr % 6]
# rep = args.arr // 6

dir = '/scratch/ewah1g13/models/'
model_file = f'resnet50_{n_bn}_{rep}'


def normalize_with(mean, std):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return lambda x: (x - mean.to(x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)) / std.to(x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


normalize = normalize_with(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
#     # transforms.Grayscale(),
#     # transforms.RandomAffine(0, translate=(0.1, 0.1)),
#     # transforms.CenterCrop(224),
#     # transforms.RandomHorizontalFlip(),
#     # transforms.Resize(128),
#     # transforms.ToTensor()  # convert to tensor
# ])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normalize
    # transforms.Grayscale(),
    # transforms.CenterCrop(224),
    # transforms.Resize(128),
    # transforms.ToTensor()  # convert to tensor
])

if args.subset:
    testset = ImageNetHDF5Subset(f'{args.root}/val', 10, transform=test_transform, cache_size=1000)
else:
    testset = ImageNetHDF5(f'{args.root}/val', transform=test_transform, cache_size=1000)
testloader = DataLoader(testset, batch_size=250, shuffle=False,  pin_memory=True, num_workers=4)

# test_transform = transforms.Compose([
#     transforms.ToTensor()  # convert to tensor
# ])

# load data
# testset = CIFAR10(".", train=False, download=True, transform=test_transform)
# testloader = DataLoader(testset, batch_size=1000, shuffle=False)
#
# bottlenecks = [1,2,4,8,16,32]
# runs = range(0,10)
# ventraldepths = [0,1,2,3,4]

epsilons = [
    0.0,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.1,
    0.3,
    0.5,
    1.0,
]

results = dict()

# for vdepth in ventraldepths:
#     results[vdepth] = dict()
#     for bn in bottlenecks:
#         results[vdepth][bn] = dict()
results[n_bn] = dict()
results[n_bn][rep] = dict()

model = RetinalBottleneckModel(n_bn, 'resnet50', n_out=1000, n_inch=3, retina_kernel_size=7, transform=normalize)
model.load_state_dict(torch.load(dir + model_file + '.pt'))

# model = torch.hub.load('ecs-vlc/opponency:master', 'colour_full', n_bn=bn, d_vvs=vdepth, rep=run)
model.eval()
fmodel = PyTorchModel(model, bounds=(0, 1))

# images, labels = samples(fmodel, dataset="cifar10", batchsize=20)
# images = images.contiguous()


# results[vdepth][bn][run]["accuracy"] = accuracy(fmodel, images, labels)

attack_success = np.zeros((len(attacks), len(epsilons), len(testset)), dtype=np.bool)
# for i, attack in enumerate(attacks):
print(attack)
idx = 0
for images, labels in tqdm(testloader):
    images = images.to(fmodel.device)
    labels = labels.to(fmodel.device)

    _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
    success_ = success.cpu().numpy()
    attack_success[attack_index][:, idx:idx+len(labels)] = success_
    idx = idx + len(labels)
# print("")
# for i, attack in enumerate(attacks):
results[n_bn][rep][str(attack)] = (1.0 - attack_success[attack_index].mean(axis=-1)).tolist()

robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
results[n_bn][rep]['robust_accuracy'] = robust_accuracy.tolist()

with open(f'results-imagenet-linf-{n_bn}-{rep}-{attack_index}.json', 'w') as fp:
    json.dump(results, fp)
