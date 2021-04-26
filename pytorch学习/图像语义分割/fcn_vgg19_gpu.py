## 导入本章所需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import glob
from time import time
import os
from skimage.io import imread
import copy
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
from torchvision.models import vgg19
from torchsummary import summary

# 定义计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取训练数据
# 定义一个读取图像的函数

def read_image(root="VOC2012/ImageSets/Segmentation/train.txt"):
    """读取指定路径下的所指定的图像文件"""
    image = np.loadtxt(root, dtype=str)
    n = len(image)
    data, label = [None] * n, [None] * n
    for i, fname in enumerate(image):
        data[i] = imread("VOC2012/JPEGImages/%s.jpg" % (fname))
        label[i] = imread("VOC2012/SegmentationClass/%s.png" % (fname))
    return data, label


traindata, trainlabel = read_image(root="VOC2012/ImageSets/Segmentation/train.txt")
# 读取验证数据集
# valdata,vallabel = read_image(root = "data/VOC2012/ImageSets/Segmentation/val.txt")
# len(traindata),len(valdata)

# 查看训练集和验证集的一些图像
# plt.figure(figsize=(12,8))
# plt.subplot(2,2,1)
# plt.imshow(traindata[0])
# plt.subplot(2,2,2)
# plt.imshow(trainlabel[0])
# plt.subplot(2,2,3)
# plt.imshow(traindata[10])
# plt.subplot(2,2,4)
# plt.imshow(trainlabel[10])
# plt.show()


# 列出每个物体对应背景的RGB值
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# 每个类的RGB值
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


# 给定一个标号图片，将像素值对应的物体找出来
def image2label(image, colormap):
    # 将标签转化为没个像素值为1类数据
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1] * 256 + cm[2])] = i
    # 对一张图像准换
    image = np.array(image, dtype="int64")
    ix = (image[:, :, 0] * 256 + image[:, :, 1] * 256 + image[:, :, 2])
    image2 = cm2lbl[ix]
    return image2


def center_crop(data, label, height, width):
    """data, label都是PIL.Image读取的图像"""
    ##使用中心裁剪（因为图像大小是一样的）
    data = transforms.CenterCrop((height, width))(data)
    label = transforms.CenterCrop((height, width))(label)
    return data, label


# 随机裁剪图像数据
def rand_crop(data, label, high, width):
    im_width, im_high = data.size
    ## 生成图像随机点的位置
    left = np.random.randint(0, im_width - width)
    top = np.random.randint(0, im_high - high)
    right = left + width
    bottom = top + high
    data = data.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))
    return data, label


# 单个图像的转换操作
def img_transforms(data, label, high, width, colormap):
    data, label = rand_crop(data, label, high, width)
    data_tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    data = data_tfs(data)
    label = torch.from_numpy(image2label(label, colormap))
    return data, label


# 定义一列出需要读取的数据路径的函数
def read_image_path(root="VOC2012/ImageSets/Segmentation/train.txt"):
    """保存指定路径下的所有需要读取的图像文件路径"""
    image = np.loadtxt(root, dtype=str)
    n = len(image)
    data, label = [None] * n, [None] * n
    for i, fname in enumerate(image):
        data[i] = "VOC2012/JPEGImages/%s.jpg" % (fname)
        label[i] = "VOC2012/SegmentationClass/%s.png" % (fname)
    return data, label


## 最后我们定义一个 MyDataset 继承于torch.utils.data.Dataset构成我们自定的训练集
class MyDataset(Data.Dataset):
    """用于读取图像，并进行相应的裁剪等"""

    def __init__(self, data_root, high, width, imtransform, colormap):
        ## data_root:数据所对应的文件名,high,width:图像裁剪后的尺寸,
        ## imtransform:预处理操作,colormap:颜色
        self.data_root = data_root
        self.high = high
        self.width = width
        self.imtransform = imtransform
        self.colormap = colormap
        data_list, label_list = read_image_path(root=data_root)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)

    def _filter(self, images):  # 过滤掉图片大小小于指定high,width的图片
        return [im for im in images if (Image.open(im).size[1] > high and
                                        Image.open(im).size[0] > width)]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.imtransform(img, label, self.high,
                                      self.width, self.colormap)
        return img, label

    def __len__(self):
        return len(self.data_list)

## 读取数据
high,width = 320,480
voc_train = MyDataset("VOC2012/ImageSets/Segmentation/train.txt",
                      high,width, img_transforms,colormap)
voc_val = MyDataset("VOC2012/ImageSets/Segmentation/val.txt",
                    high,width, img_transforms,colormap)
# 创建数据加载器每个batch使用4张图像
train_loader = Data.DataLoader(voc_train, batch_size=4,shuffle=True,
                               num_workers=8,pin_memory=True)
val_loader = Data.DataLoader(voc_val, batch_size=4,shuffle=True,
                             num_workers=8,pin_memory=True)

##  检查训练数据集的一个batch的样本的维度是否正确
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
## 输出训练图像的尺寸和标签的尺寸，和数据类型
print("b_x.shape:",b_x.shape)
print("b_y.shape:",b_y.shape)
print("b_x.dtype:",b_x.dtype)
print("b_y.dtype:",b_y.dtype)

## 将标准化后的图像转化为0～1的区间
def inv_normalize_image(data):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * rgb_std + rgb_mean
    return data.clip(0,1)

## 从预测的标签转化为图像的操作
def label2image(prelabel,colormap):
    ## 预测的到的标签转化为图像,针对一个标签图
    h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype="int32")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)
        image[index,:] = colormap[ii]
    return image.reshape(h,w,3)


## 可视化一个batch的图像，检查数据预处理 是否正确
b_x_numpy = b_x.data.numpy()
b_x_numpy = b_x_numpy.transpose(0,2,3,1)
b_y_numpy = b_y.data.numpy()
plt.figure(figsize=(16,6))
for ii in range(4):
    plt.subplot(2,4,ii+1)
    plt.imshow(inv_normalize_image(b_x_numpy[ii]))
    plt.axis("off")
    plt.subplot(2,4,ii+5)
    plt.imshow(label2image(b_y_numpy[ii],colormap))
    plt.axis("off")
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
