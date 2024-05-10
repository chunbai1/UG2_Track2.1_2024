from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
from csv import reader


class TextDataset(Dataset): 
    def __init__(self, label_file, img_path, transform = None): 
        self.transform = transform
        
        self.labels = []
        with open(label_file, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for i, row in enumerate(csv_reader):
                if i == 0: 
                    self.phase = row[0]
                    img_num = int(row[1])
                    self.len = int(row[2])
                else: 
                    self.labels.append(row)
                    
        self.imgs = []
        for i in range(img_num): 
            # self.imgs.append(Image.open(os.path.join(img_path,'out_scene_{}_out.png'.format(i+1))).convert('L'))
            self.imgs.append(Image.open(os.path.join(img_path,'scene_{}_out.png'.format(i+1))).convert('L'))
            # self.imgs.append(Image.open(os.path.join(img_path,'scene_{}_out_qf_90.png'.format(i+1))).convert('L'))
            # self.imgs.append(Image.open(os.path.join(img_path,'out_scene_{}_out.png'.format(i+1))).convert('L'))
            # self.imgs.append(Image.open(os.path.join(img_path,'scene_{}/data_0001.png'.format(i+1))).convert('L'))
            
        
    def __len__(self): 
        print(self.len)
        return self.len
    
    def __getitem__(self, idx): 
        label = self.labels[idx]
        coordinates = [int(float(i)) for i in label[2:]]
        # print(int(label[0])-1)
        patch = self.imgs[int(label[0])-1].crop(coordinates)
        patch = self.transform(patch)
        
        if self.phase == 'dry run': 
            return (patch, label[1])
        else: 
            return patch
    
    
class transform_CRNN():
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img  
    
    
class transform_DAN():
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        
        return img
    
    
class transform_ASTER(): 
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        img = img.repeat(3,1,1)
        return img  
