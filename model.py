import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self): # 输入大小（5，1，28，28）
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential( # 将多个层组合在一起
            nn.Conv2d(         # 2d一般用于图像，3d用于视频数据（多一个时间维度），1d一般用于结构化的序列数据
                in_channels=1, # 图像通道个数，1表示灰度图（确定卷积核 组中的个数）
                out_channels=16, # 要得到多少特征图，卷积核的个数
                kernel_size=5,  # 卷积核大小
                stride=1,   # 步长
                padding=2,   # 边界填充大小
                bias=True
            ), # 输出的特征图为（16，28，28）-->16个大小28*28的图像
            nn.ReLU(), # relu层，不会改变特征图的大小
            nn.MaxPool2d(kernel_size=2) # 进行池化操作（2*2区域），输出结果为（16，14，14）
        )
        self.conv2 = nn.Sequential( # 输入（16，14，14）
            nn.Conv2d(16,32,5,1,2,bias=True), # 输出（32*14*14）
            nn.ReLU(),
            nn.Conv2d(32,64,5,1,2,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2) # 输出（64，7，7）
        )
        self.out = nn.Linear(64*7*7,10) # 修改输入维度为64*7*7

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1) # 取消注释，需要进行flatten操作
        output = self.out(x)
        return output


def print_parameters(model):
    cnt = 0
    for name, layer in model.named_children():
        print(f"layer({name}) parameters:")
        for p in layer.parameters():
            print(f'\t {p.shape} has {p.numel()} parameter')
            cnt += p.numel()
    print('The model has %d trainable parameter\n' % (cnt))


def print_forward(model, x):
    # print(f"x: {x.shape}")
    # x = x.view(-1, 28 * 28)
    # print(f"after view: {x.shape}")
    x = model.conv1(x)
    print(f"after conv1: {x.shape}")
    x = model.conv2(x)
    print(f"after conv2: {x.shape}")


if __name__ == '__main__':
    model = CNN()
    print(model)
    print("")

    print_parameters(model)
    x = torch.zeros([5, 1, 28, 28])
    print_forward(model, x)
