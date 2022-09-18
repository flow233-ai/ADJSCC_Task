import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
from torch.utils import data
from encoder import Encoder
from deepjscc import DeepJSCC
from torchvision.utils import save_image
import numpy as np
from torchvision import transforms
from dwt import run

# 数据集的预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

data_path = r'D:\paper coding\ADJSCC\data2'
# 获取数据集
train_data = CIFAR10(data_path,train=True,transform=transform,download=False)
test_data = CIFAR10(data_path,train=False,transform=transform,download=False)
#train_data = train_data[0] #只取图片，不需要label
#test_data = test_data[0]

#迭代器生成
train_loader = data.DataLoader(train_data,batch_size=1,shuffle=True)

for epoch in range(1):
    for i,(x,y) in enumerate(train_loader):
        if(i > 4000):
            break
        if(i % 50 != 0):
            continue
        batch_x = Variable(x) # torch.Size([1, 1, 32, 32])
        # 获取最后输出
        batch_x = batch_x.squeeze(0)
        save_image(batch_x,'./picture/'+str(i)+'a.jpg')
        image = transforms.ToPILImage()(batch_x) # 自动转换为0-255
        print(image)
        run(i,image)

