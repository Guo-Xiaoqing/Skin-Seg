import numpy as np 
from glob import glob 
import os
import random

a = 'val'
file_list = os.listdir(r'../skin/ISIC2017/dataset/aug_'+a+'/image/') 
img_path = '../skin/ISIC2017/dataset/aug_'+a+'/image/'
seg_path = '../skin/ISIC2017/dataset/aug_'+a+'/seg/'

counter = 0
image = []
seg = []
for i in range(len(file_list)):
    str_ = file_list[i].find('.')
    seg_file_name = file_list[i][0:str_]+'_segmentation.png'
    counter += 1
    image_path1 = img_path+'/'+file_list[i]
    seg_path1 = seg_path+'/'+seg_file_name
    image.append(image_path1)
    seg.append(seg_path1)
print('sum:',counter)

train_path = open('./sls'+a+'_txt.txt','w')

index = [i for i in range(len(image))]
random.shuffle(index)
    
counter = 0  
for i in index[0:int(len(index))]:    
    train_path.write(file_list[i]+' \n')
    counter += 1
print('train data:',counter)
train_path.close()
