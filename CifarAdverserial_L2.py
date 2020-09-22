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

test_transform = transforms.Compose([
    transforms.ToTensor()  # convert to tensor
])

# load data
testset = CIFAR10(".", train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

bottlenecks = [1,2,4,8,16,32]
runs = range(0,10)
ventraldepths = [0,1,2,3,4]

attacks = [
    fa.L2FastGradientAttack(),
    fa.L2PGD(),
    fa.L2BasicIterativeAttack(),
    fa.L2AdditiveUniformNoiseAttack(),
    fa.L2DeepFoolAttack(),
]

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

for vdepth in ventraldepths:
    results[vdepth] = dict()
    for bn in bottlenecks:
        results[vdepth][bn] = dict()
        for run in runs:
            print(vdepth, bn, run)
            
            results[vdepth][bn][run] = dict()
            
            model = torch.hub.load('ecs-vlc/opponency:master', 'colour_full', n_bn=bn, d_vvs=vdepth, rep=run)
            model.eval()
            fmodel = PyTorchModel(model, bounds=(0, 1))

            # images, labels = samples(fmodel, dataset="cifar10", batchsize=20)
            # images = images.contiguous()


            # results[vdepth][bn][run]["accuracy"] = accuracy(fmodel, images, labels)

            attack_success = np.zeros((len(attacks), len(epsilons), len(testset)), dtype=np.bool)
            for i, attack in enumerate(attacks):
                print(attack)
                idx=0
                for images, labels in testloader:
                    print('.', end='')
                    images = images.to(fmodel.device)
                    labels = labels.to(fmodel.device)

                    _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
                    success_ = success.cpu().numpy()
                    attack_success[i][:,idx:idx+len(labels)] = success_
                    idx = idx + len(labels)
                print("")
            for i, attack in enumerate(attacks):
                results[vdepth][bn][run][str(attack)] = (1.0 - attack_success[i].mean(axis=-1)).tolist()

            robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
            results[vdepth][bn][run]['robust_accuracy'] = robust_accuracy.tolist()

            with open('adv-results-cifar-l2.json', 'w') as fp:
                json.dump(results, fp)
