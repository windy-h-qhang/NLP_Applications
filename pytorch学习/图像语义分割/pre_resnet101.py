## 导入本章所需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

import torch
from torchvision import transforms
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.segmentation.fcn_resnet101(pretrained=True).to(device)
model.eval()
## 读取照片
image = PIL.Image.open(r"D:\PycharmObject\pytorch学习\data\chap10\照片1.jpg")
## 照片预处理，转化到0-1之间，标准化处理
image_transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])
image_tensor = image_transf(image).unsqueeze(0).to(device)
output = model(image_tensor)["out"].to(device)

## 将输出转化为2维图像
outputarg = torch.argmax(output.squeeze(), dim=0).numpy()

## 对得到的输出结果进行编码
def decode_segmaps(image,label_colors, nc=21):
    """函数将输出的2D图像会将不同的类编码为不同的颜色"""
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for cla in range(0, nc):
        idx = image == cla
        r[idx] = label_colors[cla, 0]
        g[idx] = label_colors[cla, 1]
        b[idx] = label_colors[cla, 2]
    rgbimage = np.stack([r, g, b], axis=2)
    return rgbimage

label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

outputrgb = decode_segmaps(outputarg,label_colors)
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.imshow(image)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(outputrgb)
plt.axis("off")
plt.subplots_adjust(wspace=0.05)
plt.show()

## 读取照片
image = PIL.Image.open("data/chap10/2012_004308.jpg")
image_tensor = image_transf(image).unsqueeze(0)
output = model(image_tensor)["out"]
## 将输出转化为2维图像
outputarg = torch.argmax(output.squeeze(), dim=0).numpy()
outputrgb = decode_segmaps(outputarg,label_colors)
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
plt.imshow(image)
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(outputrgb)
plt.axis("off")
plt.subplots_adjust(wspace=0.05)
plt.show()