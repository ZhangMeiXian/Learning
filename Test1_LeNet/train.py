import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn as nn
from model import LeNet


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 50000张训练图片
train_set = torchvision.datasets.CIFAR10(root="./data",
                                         train=True,
                                         download=False,
                                         transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                          shuffle=True, num_workers=0)

# 10000张测试图片
test_set = torchvision.datasets.CIFAR10(root="./data",
                                        train=False,
                                        download=False,
                                        transform=transform)

testloader = torch.utils.data.DataLoader(test_set, batch_size=10000,
                                         shuffle=False, num_workers=0)

test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

# # 查看图像
# def imshow(img):
#     # 还原归一化
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     # 还原通道，返回原始格式
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


classes = ("plane", "car", "bird", 'cat',
           "deer", "dog", "horse", "ship", "truck")

# # print labels
# print(" ".join("%5s" % classes[test_label[j]] for j in range(4)))
# # show images
# imshow(torchvision.utils.make_grid(test_image))  # test_set batch_size设置成4

net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = net(test_image)
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)
                print("[%d, %5d] train_loss: %.3f, test_accuracy: %.3f" %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print("Finished Training")

save_path = "./Lenet.pth"
torch.save(net.state_dict(), save_path)