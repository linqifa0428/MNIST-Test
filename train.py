import torch
from torch import nn,optim
from model import CNN
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # 转换为单通道灰度图
        transforms.ToTensor()  # 转换为张量
    ])

    train_dataset = datasets.ImageFolder(root='../MNIST-Test/mnist_images/train',transform=transform)
    print("train_dataset length: ",len(train_dataset))

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    print("train loader length: ",len(train_loader))

    model = CNN()
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    all_losses = []

    for epoch in range(10):
        for batch_idx, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_losses.append(loss.item())  # 新增：记录每个batch的loss

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/10 "
                      f"| Batch {batch_idx}/{len(train_loader)}"
                      f"| Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'mnist.pth')
    # 🔥 绘制loss曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")  # 保存为图片
    plt.show()  # 显示图形窗口