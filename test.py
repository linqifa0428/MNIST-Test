from model import CNN
from torchvision import transforms
from torchvision import datasets
import torch

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # 转换为单通道灰度图
        transforms.ToTensor()  # 转换为张量
    ])

    test_dataset = datasets.ImageFolder(root='../MNIST-Test/mnist_images/train',transform=transform)
    print("test_dataset length: ", len(test_dataset))

    model = CNN()
    model.eval()
    model.load_state_dict(torch.load('mnist.pth'))

    right = 0
    for i, (x, y) in enumerate(test_dataset):
        x = x.unsqueeze(0)  # 添加batch维度,从[1,28,28]变为[1,1,28,28]
        output = model(x)
        predict = output.argmax(1).item()
        if predict == y:
            right += 1
        else:
            img_path = test_dataset.samples[i][0]
            print(f"wrong case: predict = {predict} y = {y} img_path = {img_path}")

    sample_num = len(test_dataset)
    acc = right * 1.0 / sample_num
    print("test accuracy = %d / %d = %.3lf" %(right,sample_num,acc))
