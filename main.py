'''
目前已完成进度：
1. 论文主体模型，包括编译码器和AWGN信道
2. 主体模型可以以MINST为数据集进行训练
3. 将ReLU改为PReLU，完成复原图片生成模块
4. 加入信道SNR，加入PSNR指标计算
5. 考虑功率约束P，加入norm模块进行能量限制
6. 开始使用CIFAR10数据集，并将epoch设置为10
7. 使用GDN和IGDN代替Batchnorm模块
8. 增加FL和AF模块，引入注意力机制
9. 完成JPEG和JPEG2000模块,使用GPU进行加速
10. 更改网络结构，增加epoch和学习率衰减
11. 考虑下游图像分类任务(语用网络预先训练）
12. Bug修复（PSNR\MSE计算错误，Decoder无Power恢复）
13. 添加复数combiner，考虑慢瑞利衰落信道
待完成部分：
14. 将ADJSCC、JPEG和JPEG2000进行绘图比较等
'''

import torch
import torchvision
import torch.nn as nn
import math
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
from torch.utils import data
from encoder import Encoder
from deepjscc import DeepJSCC
import numpy as np
from torchvision import transforms
from trainCNN import CNN_NET

#计算PSNR指标
def PSNR(loss):
    return 10 * math.log10(1/loss)

def EachImg(img):
    img=img/2+0.5   #将图像数据转换为0.0->1.0之间，才能正常对比度显示（以前-1.0->1.0色调对比度过大）
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()

#加速计算
torch.manual_seed(3407)
#CPU\GPU转换
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集的预处理
transform = transforms.Compose([transforms.ToTensor()])

data_path = r'D:\paper coding\ADJSCC_Task\data2'
# 获取数据集
train_data = CIFAR10(data_path,train=True,transform=transform,download=False)
test_data = CIFAR10(data_path,train=False,transform=transform,download=False)

#样本可视化
image = train_data[0][0]
#print(image)
#EachImg(image)

#迭代器生成
train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True)
# train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True,num_workers=8)
test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True)
# test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True,num_workers=8)

#定义损失和优化器
model = DeepJSCC(5) #设置信道SNR为20dB
#print(model.ratio) #查看信噪权值ratio
loss_func = nn.MSELoss()
loss_func2 = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

if __name__ == '__main__':
    #训练网络
    loss_count = []
    for epoch in range(15):
        for i,(x,y) in enumerate(train_loader):
            batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
            batch_y = Variable(y) # torch.Size([128])
            # 获取最后输出
            #print(batch_x[:,0,:,:].unsqueeze(1).shape)

            out1 = model.forward1(batch_x[:,0,:,:].unsqueeze(1)) # torch.Size([128,10])
            loss1 = loss_func(out1,batch_x[:,0,:,:].unsqueeze(1))
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss1.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            # 使用优化器优化损失

            out2 = model.forward1(batch_x[:,1,:,:].unsqueeze(1)) # torch.Size([128,10])
            loss2 = loss_func(out2,batch_x[:,1,:,:].unsqueeze(1))
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss2.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            # 使用优化器优化损失

            out3 = model.forward1(batch_x[:,2,:,:].unsqueeze(1)) # torch.Size([128,10])
            loss3 = loss_func(out3,batch_x[:,2,:,:].unsqueeze(1))
            #opt.zero_grad()  # 清空上一步残余更新参数值
            #loss3.backward() # 误差反向传播，计算参数更新值
            #opt.step() # 将参数更新值施加到net的parmeters上
            # 使用优化器优化损失

            #temp1 = torch.concat([out1,out2,out3],dim=1)
            #out4 = model.forward2(temp1)
            #loss4 = 0.2 * loss_func2(out4,batch_y)
            #loss = loss1 + loss2 + loss3 + loss4
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss3.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上


            if i % 20 == 0:
                loss_count.append(loss1.detach().numpy())
                print('{}:\t'.format(i), loss1.item(),'  ',loss2.item(),'  ',loss3.item())

        scheduler.step()

    print('PSNR:',PSNR((loss1+loss2+loss3)/3),'dB')
    #torch.save(model,r'D:\paper coding\ADJSCC_Task\model\ADJSCC.pth')
    plt.figure('PyTorch_CNN_Loss')
    plt.plot(loss_count,label='Loss')
    plt.legend()
    plt.show()