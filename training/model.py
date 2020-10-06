import torch
import torch.nn as nn
import torchvision.models as models


class Flatten(nn.Module):
    """Flatten incoming tensors
    """
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view(x.size()[0], -1)


def init_weights(m):
    """Inplace xavier uniform initialisation with zero bias for conv and linear layers
    :param m: The layer to initialise
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class RetinalBottleneckModel(nn.Module):
    """
    Parameterised implemention of the retina-net - ventral stream architecture.
    This version allows different architectures to be used for the ventral stream (e.g. a ResNet)
    Note that the last layer does not have an explicit softmax and will thus output the logits

    :param n_bn: Number of filters in the bottleneck layer
    :param ventral: This specifies the ventral stream model; it can be:
        - an integer specifying the depth (using Lindsay et al's original ventral model)
        - an nn.Module implementing a ventral stream (in which case the model will be modified as necessary to match dimensionality)
        - a string ('resnet18', 'resnet50', etc to specify a resnet ventral stream. Value must match one of the models in torchvision.models.)
    :param n_inch: Number of input channels
    :param n_out: Number of classifier outputs
    :param init: if true apply Xavier initialisation to linear and conv layers and zero biases

    Limitations: the ventral model rewriting code needs the first layer of the model to be in either 
    ventral._modules[0][1] or ventral._modules[0][1]._modules[0][1] in the case of a nn.Sequential wrapper. 
    The first layer must be an nn.Conv2d for the rewriting to work. Similar limitations apply to the final layer,
    which needs to be last in the _module dict (or last in the last _module dict entry if there is an nn.Sequential wrapper).
    """
    def __init__(self, n_bn, ventral, n_inch=1, n_out=10, init=True, retina_kernel_size=9, retina_ch=32, transform=None):
        super(RetinalBottleneckModel, self).__init__()

        self.transform = transform

        self.retina = nn.Sequential()
        self.retina.add_module("retina_conv1", nn.Conv2d(n_inch, retina_ch, (retina_kernel_size, retina_kernel_size), padding=retina_kernel_size // 2))
        self.retina.add_module("retina_relu1", nn.ReLU())
        self.retina.add_module("retina_conv2", nn.Conv2d(retina_ch, n_bn, (retina_kernel_size, retina_kernel_size), padding=retina_kernel_size // 2))
        self.retina.add_module("retina_relu2", nn.ReLU())
        
        if isinstance(ventral, int):
            self.__build_default_ventral(n_bn, n_out, ventral)
        elif isinstance(ventral, nn.Module):
            self.__build_module_ventral(n_bn, n_out, ventral)
        elif isinstance(ventral, str):
            mdl = getattr(models, ventral)
            self.__build_module_ventral(n_bn, n_out, mdl())
        else:
            raise TypeError("ventral argument must be an integer, string or nn.Module")

        if init:
            self.retina.apply(init_weights)

    def __build_default_ventral(self, last_size, n_out, d_vvs):
        self.ventral = nn.Sequential()
        for i in range(d_vvs):
            self.ventral.add_module("ventral_conv"+str(i), nn.Conv2d(last_size, 32, (9, 9), padding=4))
            self.ventral.add_module("ventral_relu"+str(i), nn.ReLU())
            last_size = 32

        self.ventral.add_module("ventral_flatten", Flatten())
        self.ventral.add_module("ventral_fc1", nn.Linear(last_size*32*32, 1024))
        self.ventral.add_module("ventral_fc1_relu", nn.ReLU())
        self.ventral.add_module("ventral_fc2", nn.Linear(1024, n_out))

    def __build_module_ventral(self, last_size, n_out, mdl):
        mods = list(mdl._modules.items())

        # fix up first layer to have the correct dimensionality
        old_name, old_mod = mods[0]
        modules = mdl._modules
        if isinstance(old_mod, nn.Sequential):
            #vgg has sequential wrappers
            old_name, old_mod = list(mods[0][1]._modules.items())[0]
            modules = mdl._modules[mods[0][0]]._modules
        if isinstance(old_mod, nn.Conv2d):
            if old_mod.in_channels != last_size:
                modules[old_name] = nn.Conv2d(
                    in_channels=last_size, 
                    out_channels=old_mod.out_channels, 
                    kernel_size=old_mod.kernel_size, 
                    stride=old_mod.stride, 
                    padding=old_mod.padding, 
                    dilation=old_mod.dilation, 
                    groups=old_mod.groups, 
                    bias=old_mod.bias is not None, 
                    padding_mode=old_mod.padding_mode)
        else:
            raise NotImplementedError('Unsupported first layer type')

        # and the last layer
        old_name, old_mod = mods[-1]
        modules = mdl._modules
        if isinstance(old_mod, nn.Sequential):
            #vgg has sequential wrappers
            old_name, old_mod = list(mods[-1][1]._modules.items())[-1]
            modules = mdl._modules[mods[-1][0]]._modules
        if isinstance(old_mod, nn.Linear):
            if old_mod.out_features != n_out:
                modules[old_name] = nn.Linear(
                    in_features=old_mod.in_features, 
                    out_features=n_out, 
                    bias=old_mod.bias is not None)
        else:
            raise NotImplementedError('Unsupported last layer type')

        self.ventral = mdl

    def has_layer(self, layer_name):
        return True
        # for name, module in self.retina:
        #     if layer_name == name:
        #         return True
        # for name, module in self.ventral:
        #     if layer_name == name:
        #         return True
        # return False

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        x = self.retina(x)
        # for name, module in self.retina:
        #     x = module(x)
        # for name, module in self.ventral:
        #     x = module(x)
        return self.ventral(x)

    def forward_to_layer(self, x, layer_name):
        if self.transform is not None:
            x = self.transform(x)

        loc, index = layer_name.split('_')
        index = int(index)

        if loc == 'retina':
            for i, module in enumerate(self.retina):
                x = module(x)
                if i == index:
                    return x
        else:
            x = self.retina(x)

        if loc == 'ventral':
            for i, module in enumerate(list(self.ventral._modules.items())):
                _, module = module
                x = module(x)
                if i == index:
                    return x
        else:
            x = self.ventral(x)
        return x


if __name__ == '__main__':
    # """Train a single mpodel to test
    # """
    # import torchvision.transforms as transforms
    # from torchbearer import Trial
    # import torchbearer
    # from torch import optim
    # from torch.utils.data import DataLoader
    # from torchvision.datasets import CIFAR10

    # train_transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.RandomAffine(0, translate=(0.1, 0.1)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()  # convert to tensor
    # ])
    # test_transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.ToTensor()  # convert to tensor
    # ])

    # # load data
    # trainset = CIFAR10(".", train=True, download=True, transform=train_transform)
    # testset = CIFAR10(".", train=False, download=True, transform=test_transform)

    # # create data loaders
    # trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    # testloader = DataLoader(testset, batch_size=32, shuffle=True)

    # model = BaselineModel(4, 1)
    # print(model)

    # optimiser = optim.RMSprop(model.parameters(), alpha=0.9, lr=0.0001, weight_decay=1e-6)
    # loss_function = nn.CrossEntropyLoss()

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
    # trial.with_generators(trainloader, test_generator=testloader)
    # trial.run(epochs=20)
    # results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    # print(results)
    print(RetinalBottleneckModel(8, 4, n_out=10))
    print(RetinalBottleneckModel(8, 'resnet50', n_out=10, n_inch=3))
    print(RetinalBottleneckModel(8, models.vgg16(), n_out=101, n_inch=3))
