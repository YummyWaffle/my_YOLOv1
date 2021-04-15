import os
import cv2
import math
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np
#import torchvision.transforms as transforms


class my_Loader(data.Dataset):

    CLASSES = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
               'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 
               'harbor', 'overpass','ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 
               'vehicle','windmill']
    mean = (104,117,123)           
    def __init__(self,dataset_folder):
        # Give The Path of VOCdevkit
        self.image_folder = dataset_folder+'/VOC2007/JPEGImages'
        self.annotation_folder = dataset_folder+'/VOC2007/Annotations'
        self.trainset = dataset_folder+'/VOC2007/ImageSets/Main/train.txt'
        # Read Trainset
        self.anno_list = []
        self.img_list = []
        f = open(self.trainset,'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            self.img_list.append(self.image_folder + '/' + line[0] + '.jpg')
            self.anno_list.append(self.annotation_folder + '/' + line[0] + '.xml')
        self.image_num = len(self.img_list)
        
    def __getitem__(self,index):
        # Fetch Image
        image = cv2.imread(self.img_list[index])
        # Fetch Meta-Infomations 
        root = ET.parse(self.anno_list[index])
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        # Get Resize Para.
        para_w = float(img_width) / 448.
        para_h = float(img_height) / 448.
        image = cv2.resize(image,(448,448))
        objs = root.findall('object')
        labels = []
        boxes = []
        for obj in objs:
            labels.append(obj.find('name').text)
            xmin = float(obj.find('bndbox').find('xmin').text)/para_w
            xmax = float(obj.find('bndbox').find('xmax').text)/para_w
            ymin = float(obj.find('bndbox').find('ymin').text)/para_h
            ymax = float(obj.find('bndbox').find('ymax').text)/para_h
            boxes.append(np.array([xmin,ymin,xmax,ymax]))
        # Encode Label
        e_label = torch.zeros(size=(len(boxes),len(self.CLASSES)))
        for midx in range(len(labels)):
            e_label[midx,self.CLASSES.index(labels[midx])] = 1.
        #print(e_label)
        # Encode Regression Param.
        n_element = 5 * 2 + len(self.CLASSES)
        target = torch.zeros(7,7,n_element)
        for box_idx,box in enumerate(boxes):
            w = (box[2] - box[0])/448.
            h = (box[3] - box[1])/448.
            xc = box[0] + (box[2] - box[0])/2.
            yc = box[1] + (box[3] - box[1])/2.
            norm_para = 448. / 7.
            i = math.ceil(xc/norm_para) - 1
            j = math.ceil(yc/norm_para) - 1
            x0 = i * norm_para
            y0 = j * norm_para
            nxc = (xc-x0) /norm_para
            nyc = (yc-y0) / norm_para
            block_unit = torch.tensor([nxc,nyc,w,h,1.,nxc,nyc,w,h,1.])
            #print(block_unit)
            block = torch.cat((block_unit,e_label[box_idx]))
            target[i,j] = block
        # Subtracting Mean & Normalize Image, then Trans to Tensor
        image_out = np.array(image,dtype=np.float32)
        image_out -= self.mean
        image_out /= 255.
        image_out = torch.tensor(image_out).unsqueeze(0).permute(0,3,1,2)
        target = target.unsqueeze(0)
        return image_out.cuda(),target.cuda()
        
    def __len__(self):
        return self.image_num
