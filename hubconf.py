dependencies = ['torch', 'torchvision']

from torch.hub import load_state_dict_from_url


def _imagenet_resnet50_bottleneck(n_bn=32, rep=0):
    from training import RetinalBottleneckModel
    model = RetinalBottleneckModel(n_bn, 'resnet50', n_out=1000, n_inch=3, retina_kernel_size=7)

    state = load_state_dict_from_url(
        f'http://marc.ecs.soton.ac.uk/pytorch-models/opponency/resnet50/resnet50_{n_bn}_{rep}.pt',
        progress=True,
        map_location='cpu'
    )
    model.load_state_dict(state)
    return model


def imagenet_resnet50(n_bn=32, rep=0):
    return _imagenet_resnet50_bottleneck(n_bn=n_bn, rep=rep)
