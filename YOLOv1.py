import torch
import torch.nn as nn

class conv_bn_lkrelu(nn.Module):
    def __init__(self,inplanes,planes,mkernel_size=3,mstride=1,mpadding=1):
        super(conv_bn_lkrelu,self).__init__()
        self.conv = nn.Conv2d(inplanes,planes,kernel_size=mkernel_size,
                             stride=mstride,padding=mpadding)
        self.bn = nn.BatchNorm2d(planes)
        self.lkrelu = nn.LeakyReLU(0.1)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lkrelu(x)
        return x
        
class conv_pack(nn.Module):
    expansion = 2
    def __init__(self,inplanes,planes):
        super(conv_pack,self).__init__()
        self.conv1x1 = nn.Conv2d(inplanes,planes,1,1,0)
        self.conv3x3 = nn.Conv2d(planes,planes*self.expansion,3,1,1)
    def forward(self,x):
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        return x

class YOLOv1(nn.Module):
    def __init__(self,class_num):
        super(YOLOv1,self).__init__()
        self.class_num = class_num
        self.layer1 = nn.Sequential(
            conv_bn_lkrelu(3,64,7,2,3),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.layer2 = nn.Sequential(
            conv_bn_lkrelu(64,192,3,1,1),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.layer3 = nn.Sequential(
            conv_pack(192,128),
            conv_pack(256,256),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2,stride=2),         
        )
        self.layer4 = nn.Sequential(
            conv_pack(512,256),
            conv_pack(512,256),
            conv_pack(512,256),
            conv_pack(512,256),
            conv_pack(512,512),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.layer5 = nn.Sequential(
            conv_pack(1024,512),
            conv_pack(1024,512),
            nn.Conv2d(1024,1024,3,1,1),
            nn.Conv2d(1024,1024,3,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024,1024,3,1,1),
            nn.Conv2d(1024,1024,3,1,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.conn1 = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
        )
        self.conn2 = nn.Linear(4096,7*7*(5*2+class_num))
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0),-1)
        x = self.conn1(x)
        x = self.conn2(x)
        # We Actually Got [Batchsize,7*7*30] Tensor Try to Make it [BatchSize,7,7,30] Tensor
        #x = x.view(-1,7,7,30)
        return x