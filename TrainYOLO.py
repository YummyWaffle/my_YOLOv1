import torch
from DataLoader import my_Loader
from YOLOv1 import YOLOv1
import random
import numpy as np
from YOLOLoss import YOLOLoss
import torch.optim as optim
import time
import math
import datetime

# Read Data
data_loader = my_Loader('D:/0426DIOR/DIOR/VOCdevkit')
train_num = len(data_loader)

# Read Model
model = YOLOv1(20).cuda()

# Set Training Params.
Epochs = 10
batch_size = 16
index_list = np.arange(train_num)
random.shuffle(index_list)
optimizer = optim.SGD(model.parameters(),lr=0.001)

info_interval = 5

# Start Training
for Epoch in range(Epochs):
    print('-----------------------EPOCH:# '+str(Epoch+1)+' #------------------------------')
    counter = 0
    for i in range(0,train_num,batch_size):
        total_batch = math.ceil(train_num / batch_size)
        start = time.time()
        batch_indxs = index_list[i:min(i+batch_size,train_num)] 
        real_batch_size = len(batch_indxs)
        in_tsr,in_target = data_loader[batch_indxs[0]]
        for batch_indx in range(1,real_batch_size):
            temp_img,temp_target = data_loader[batch_indx]
            in_tsr = torch.cat((in_tsr,temp_img),0)
            in_target = torch.cat((in_target,temp_target),0)
        #torch.cuda.empty_cache()
        out_tsr = model(in_tsr)
        loss_func = YOLOLoss(real_batch_size)
        Loss = loss_func(out_tsr,in_target)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        end = time.time()
        counter += 1
        if(counter%info_interval==0):
            print('Time / Batch: %.2f  Eta: %s  Epoch:(%d/%d)  Batch: (%d/%d)  Loss: %.4f'%(end-start,datetime.timedelta(seconds=(end-start)*((total_batch-counter)+(Epochs-Epoch)*total_batch)),Epoch+1,Epochs,counter,total_batch,Loss.item()))
        
        