import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose(
    [transforms.Resize(32, 32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

classes = ("plane", "car", "bird", 'cat',
           "deer", "dog", "horse", "ship", "truck")

net = LeNet()
net.load_state_dict(torch.load("Lenet.pth"))

img = Image.open("1.jpg")
img = transform(img)  # [C, H, W]
img = torch.unsqueeze(img, dim=0)  # [N, C, H, W]


with torch.no_grad():
    outputs = net(img)
    predict = torch.max(outputs, dim=1)[1].data.numpy()

print(classes[int(predict)])

