import torch
from torch import nn
from torch import optim
from model import Network
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # 转换为单通道灰度图
        transforms.ToTensor()  # 转换为张量
    ])

    train_dataset = datasets.ImageFolder(root='../MNIST-Test/mnist_images/train',transform=transform)
    print("train_dataset length: ",len(train_dataset))

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    print("train loader length: ",len(train_loader))

    model = Network()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch_idx, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/10 "
                      f"| Batch {batch_idx}/{len(train_loader)}"
                      f"| Loss: {loss.item():.4f}")

    torch.save(model.state_dict(),'mnist.pth')
