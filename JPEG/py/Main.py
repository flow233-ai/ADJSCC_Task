import RGB2YUV
import DCT
import Quantization
import AC
import DC
import Compress
import cv2
import math
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

def printBlock(block):
    for row in block:
        print(row)

DCT = DCT.DCT()
Quantization = Quantization.Quantization()
AC = AC.AC()
DC = DC.DC()
Compress = Compress.Compress()

def PSNR(loss):
    return 10 * math.log10(1/loss)

def compress(img):
    height = img.shape[0]
    width = img.shape[1]
    Y, U, V = RGB2YUV.rgb2yuv(img, img.shape[1], img.shape[0])
    Y = DCT.fill(Y)
    U = DCT.fill(U)
    V = DCT.fill(V)
    blocksY = DCT.split(Y)
    blocksU = DCT.split(U)
    blocksV = DCT.split(V)
    FDCT = []
    Quan = []
    Z = []
    ACnum = []
    for block in blocksY:
        FDCT.append(DCT.FDCT(block))
        Quan.append(Quantization.quanY(FDCT[-1]))
        Z.append(AC.ZScan(Quan[-1]))
        ACnum.append(AC.RLC(Z[-1]))
    DCnum = DC.DPCM(Quan)
    #print('Y: ')
    Bstr0 = ''
    for i in range(len(ACnum)):
        Bstr0 += Compress.AllCompressY(DCnum[i], ACnum[i])
    #print(Bstr0)
    #print(len(Bstr0))

    FDCT = []
    Quan = []
    Z = []
    ACnum = []
    for block in blocksU:
        FDCT.append(DCT.FDCT(block))
        Quan.append(Quantization.quanUV(FDCT[-1]))
        Z.append(AC.ZScan(Quan[-1]))
        ACnum.append(AC.RLC(Z[-1]))
    DCnum = DC.DPCM(Quan)
    #print('U: ')
    Bstr1 = ''
    for i in range(len(ACnum)):
        Bstr1 += Compress.AllCompressUV(DCnum[i], ACnum[i])
    #print(Bstr1)
    #print(len(Bstr1))

    FDCT = []
    Quan = []
    Z = []
    ACnum = []
    for block in blocksV:
        FDCT.append(DCT.FDCT(block))
        Quan.append(Quantization.quanUV(FDCT[-1]))
        Z.append(AC.ZScan(Quan[-1]))
        ACnum.append(AC.RLC(Z[-1]))
    DCnum = DC.DPCM(Quan)
    #print('V: ')
    Bstr2 = ''
    for i in range(len(ACnum)):
        Bstr2 += Compress.AllCompressUV(DCnum[i], ACnum[i])
    #print(Bstr2)
    #print(len(Bstr2))
    s = Bstr0 + Bstr1 + Bstr2
    #print(len(s))
    return height, width, s

def encoding(bs, width, heigth,i):
    DCY, DCU, DCV, ACY, ACU, ACV = Compress.encoding(bs, height, width)
    YBlocks = DC.DPCM2(DCY)
    UBlocks = DC.DPCM2(DCU)
    VBlocks = DC.DPCM2(DCV)
    for i in range(len(YBlocks)):
        AC.Z2Tab(ACY[i], YBlocks[i])
        YBlocks[i] = Quantization.reY(YBlocks[i])
        YBlocks[i] = DCT.IDCT(YBlocks[i])
    for i in range(len(UBlocks)):
        AC.Z2Tab(ACU[i], UBlocks[i])
        UBlocks[i] = Quantization.reUV(UBlocks[i])
        UBlocks[i] = DCT.IDCT(UBlocks[i])
    for i in range(len(VBlocks)):
        AC.Z2Tab(ACV[i], VBlocks[i])
        VBlocks[i] = Quantization.reUV(VBlocks[i])
        VBlocks[i] = DCT.IDCT(VBlocks[i])

    Y, U, V = DCT.merge(YBlocks, UBlocks, VBlocks, height, width)
    img = RGB2YUV.yuv2rgb(Y, U, V, width, height)
    img = img.astype('float32')
    img = (img / 255 - 0.5) / 0.5
    return img

# 数据集的预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

data_path = r'D:\paper coding\ADJSCC_Task\data2'
# 获取数据集
train_data = CIFAR10(data_path,train=True,transform=transform,download=False)
test_data = CIFAR10(data_path,train=False,transform=transform,download=False)
#train_data = train_data[0] #只取图片，不需要label
#test_data = test_data[0]

#迭代器生成
train_loader = data.DataLoader(train_data,batch_size=1,shuffle=True)
#定义损失
loss_func = nn.MSELoss()

for epoch in range(1):
    for i,(x,y) in enumerate(train_loader):
        if(i > 4000):
            break
        if(i % 50 != 0):
            continue
        batch_x = Variable(x) # torch.Size([1, 1, 32, 32])
        # 获取最后输出
        batch_x = batch_x.squeeze(0)
        save_image(batch_x,'../picture/'+str(i)+'a.jpg')
        image = batch_x.numpy()
        image = image.swapaxes(0,2)
        image = np.round((image*0.5+0.5)*255).astype(int)
        #print(image)
        #压缩图像
        height, width, s = compress(image)
        s_temp = [x for x in s]
        s_temp = list(map(int,s_temp))
        s_temp = np.array(s_temp,dtype='float32')
        noise = np.random.randn(len(s))

        #加入噪声
        SNR = 0
        ratio = math.sqrt(1 / (math.pow(10,SNR/10)))
        s_temp += ratio * noise
        s_temp = s_temp.tolist()
        s2 = [(0 if x <= 0.5 else 1) for x in s_temp]
        s2 = list(map(str,s2))
        s2 = ''.join(s2)
        #if(s2 != s):
        #    print('a')

        f = open(r'../txt.txt', 'w', encoding='utf-8')
        f.write(s2)
        f.close()


        #恢复图像
        f = open(r'../txt.txt', 'r', encoding='utf-8')
        s = f.read()

        temp = encoding(s, width, height,i)
        f.close()
        temp = torch.tensor(temp.swapaxes(0, 2))
        loss = loss_func(batch_x,temp)
        print(len(s)/(32*32*6),'    ',PSNR(loss.item()),'dB')
        save_image(temp, '../picture/' + str(i) + 'b.jpg')





