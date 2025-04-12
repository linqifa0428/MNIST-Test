import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)


def print_parameters(model):
    cnt = 0
    for name, layer in model.named_children():
        print(f"layer({name}) parameters:")
        for p in layer.parameters():
            print(f'\t {p.shape} has {p.numel()} parameter')
            cnt += p.numel()
    print('The model has %d trainable parameter\n' % (cnt))


def print_forward(model, x):
    print(f"x: {x.shape}")
    x = x.view(-1, 28 * 28)
    print(f"after view: {x.shape}")
    x = model.layer1(x)
    print(f"after layer1: {x.shape}")
    x = torch.relu(x)
    print(f"after relu: {x.shape}")
    x = model.layer2(x)
    print(f"after layer2: {x.shape}")


if __name__ == '__main__':
    model = Network()
    print(model)
    print("")

    print_parameters(model)
    x = torch.zeros([5, 28, 28])
    print_forward(model, x)
