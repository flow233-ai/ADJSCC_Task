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

# 数据集的预处理
transform = transforms.Compose([transforms.ToTensor()])

data_path = r'D:\paper coding\ADJSCC_Task\data2'
# 获取数据集
train_data = CIFAR10(data_path,train=True,transform=transform,download=False)
test_data = CIFAR10(data_path,train=False,transform=transform,download=False)
#train_data = train_data[0] #只取图片，不需要label
#test_data = test_data[0]

#迭代器生成
train_loader = data.DataLoader(train_data,batch_size=1,shuffle=True)

#导入训练好的模型参数
model = torch.load("./model/ADJSCC_Task.pth")

#样本可视化
from torchvision import transforms

#定义损失
loss_func = nn.MSELoss()

for epoch in range(1):
    for i,(x,y) in enumerate(train_loader):
        if(i > 4000):
            break
        batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
        #batch_y = Variable(y) # torch.Size([128])
        # 获取最后输出
        out = model.forward1(batch_x[:,0,:,:].unsqueeze(1)) # torch.Size([128,10])
        out2 = model.forward1(batch_x[:,1,:,:].unsqueeze(1)) # torch.Size([128,10])
        out3 = model.forward1(batch_x[:,2,:,:].unsqueeze(1)) # torch.Size([128,10])
        out = torch.cat([out,out2,out3],dim = 1)

        #print(out.shape)
        if i % 50 == 0:
            #save_image(batch_x,'./picture/'+str(i)+'a.jpg')
            #save_image(out,'./picture/'+str(i)+'b.jpg')
            loss = loss_func(out,batch_x)
            print('{}:\t'.format(i), loss.item())