from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import torch
import os
import numpy as np
import struct
class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype,nnx,nny,batchsize,nxx,nyy,overlap, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
    
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.dataroot=dataroot
        self.batchsize=batchsize
        self.nxx=nxx
        self.nyy=nyy
        self.nnx=nnx
        self.nny=nny
        self.overlap=overlap     
 
        
        
        self.sr_path = [ '{}/sr_{}_{}/'.format(dataroot, l_resolution, r_resolution)+x for x in sorted(os.listdir('{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution)),key=lambda x:int(x.split('.')[0]))]
 
        self.hr_path = ['{}/hr_{}/'.format(dataroot, r_resolution)+x for x in sorted(os.listdir('{}/hr_{}'.format(dataroot, r_resolution)),key=lambda x:int(x.split('.')[0]))]
     
        
     
        if self.need_LR:
            self.lr_path =['{}/lr_{}/'.format(dataroot, l_resolution)+x for x in sorted(os.listdir('{}/lr_{}'.format(dataroot, l_resolution)),key=lambda x:int(x.split('.')[0]))]   
        
        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)                
        

    def __len__(self):
        if self.split=='train':
        
          return self.data_len*4
        else:
          print(self.data_len*(self.nxx//128)*(self.nyy//128)*self.overlap)
          
          return self.data_len*(self.nxx//128)*(self.nyy//128)*self.overlap
        

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
      
        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        elif self.datatype == 'img':
        
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        else:
            
            if self.split=='train':
              
              index1=index//4
              
              x=random.randint(32, 128*4-128-32)
              y=random.randint(32, 128*4-128-32)
              
            
              img_HR2 = np.transpose((np.fromfile(self.hr_path[index1], dtype=np.float32).reshape(1,128*4, 128*4)),axes=(0,2,1))
              img_SR2 = np.transpose((np.fromfile(self.sr_path[index1], dtype=np.float32).reshape(1,128*4, 128*4)),axes=(0,2,1))
            
              img_HR=img_HR2[:,y:y+128,x:x+128]
              img_SR=img_SR2[:,y:y+128,x:x+128]
   
              if np.max(np.abs(img_HR))!=0:
                img_HR=np.copy(img_HR/np.max(np.abs(img_HR)))
              if np.max(np.abs(img_SR))!=0:
                img_SR=np.copy(img_SR/np.max(np.abs(img_SR)))
                
              if self.need_LR:
                img_LR2 = np.transpose((np.fromfile(self.lr_path[index1], dtype=np.float32).reshape(1,128*4, 128*4)),axes=(0,2,1))
               
                img_LR=img_LR2[:,y:y+128,x:x+128]
                if np.max(np.abs(img_LR))!=0:
                  img_LR=np.copy(img_LR/np.max(np.abs(img_LR)))
            else:
             
              ec=index//((self.nyy//128)*(self.nxx//128))
              ec1=(index//((self.nyy//128)*(self.nxx//128)))% self.overlap
              ec2=(index//((self.nyy//128)*(self.nxx//128)))// self.overlap
              x=(index//(self.nyy//128))%(self.nxx//128)
              
              y=index%(self.nyy//128)
              ec=ec+1
              ec1=ec1+1
             
            
              img_HR1 = np.transpose((np.fromfile(self.hr_path[ec2], dtype=np.float32).reshape(1,self.nnx,self.nny)),axes=(0,2,1))
              img_SR1 = np.transpose((np.fromfile(self.sr_path[ec2], dtype=np.float32).reshape(1,self.nnx,self.nny)),axes=(0,2,1))
              siz=8
              img_HR=img_HR1[:,siz*ec1+128*y:siz*ec1+128*(y)+128,siz*ec1+128*x:siz*ec1+128*(x)+128]
              img_SR=img_SR1[:,siz*ec1+128*y:siz*ec1+128*(y)+128,siz*ec1+128*x:siz*ec1+128*(x)+128]
              #print("in the dataset,ec1=",ec1, "index=",index)
               
              file = open('./coe_{}.txt'.format(ec), 'a')          
              file.write(str(np.max(np.abs(img_SR)))+'\n')
              
              if np.max(np.abs(img_HR))!=0:
                img_HR=np.copy(img_HR/np.max(np.abs(img_HR)))
              if np.max(np.abs(img_SR))!=0:
                img_SR=np.copy(img_SR/np.max(np.abs(img_SR)))
       
              #index1=index%16
              #index2=index//16
              #img_HR = np.transpose((np.fromfile('{}/hr_{}/slices_3patch_{}_{}.dat'.format(self.dataroot, 128,7*64+index1*128,index2*1*128), dtype=np.float32).reshape(1,128, 128)),axes=(0,2,1))
              #img_HR2 = np.transpose((np.fromfile('{}/hr_{}/whole.dat'.format(self.dataroot, 128), dtype=np.float32).reshape(1,301,3001)),axes=(0,2,1))
              #img_HR3 = np.pad(img_HR2, pad_width=((0,0),(0, 1), (0, 1)), mode='constant')
              #img_HR=img_HR3[:,0:2048,0:256]
              #img_SR = np.transpose((np.fromfile('{}/sr_{}_{}/slices_3patch_{}_{}.dat'.format(self.dataroot, 16, 128,7*64+index1*128,index2*1*128), dtype=np.float32).reshape(1,128, 128)),axes=(0,2,1))
              #img_SR2 = np.transpose((np.fromfile('{}/sr_{}_{}/whole.dat'.format(self.dataroot, 16, 128), dtype=np.float32).reshape(1,301,3001)),axes=(0,2,1)) 
              #img_SR3 = np.pad(img_SR2, pad_width=((0,0),(0, 1), (0, 1)), mode='constant')
              #img_SR=img_SR3[:,0:2048,0:256]

              if self.need_LR:
                img_LR1 = np.transpose((np.fromfile(self.lr_path[ec2], dtype=np.float32).reshape(1,self.nnx,self.nny)),axes=(0,2,1))
                
                img_LR=img_LR1[:,siz*ec1+128*y:siz*ec1+128*(y)+128,siz*ec1+128*x:siz*ec1+128*(x)+128]
                if np.max(np.abs(img_LR))!=0:
                  img_LR=np.copy(img_LR/np.max(np.abs(img_LR)))
                
                #img_LR = np.transpose((np.fromfile('{}/lr_{}/slices_3patch_{}_{}.dat'.format(self.dataroot, 16,7*64+index1*128,index2*1*128), dtype=np.float32).reshape(1,128, 128)),axes=(0,2,1))
                #img_LR2 = np.transpose((np.fromfile('{}/lr_{}/whole.dat'.format(self.dataroot, 16), dtype=np.float32).reshape(1,301, 3001)),axes=(0,2,1))
                #img_LR3 = np.pad(img_LR2, pad_width=((0,0),(0, 1), (0, 1)), mode='constant')
                #img_LR=img_LR3[:,0:2048,0:256]
                
        if self.need_LR:
           
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:

            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
        
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
