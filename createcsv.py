import os,shutil
import re
import cv2
import json
import pandas as pd
rootdir='E:\\xjh\\train_data_cut\\'
dstdir='E:\\xjh\\train_data_final\\'
list = os.listdir(rootdir)
datalist=[]
data={}
csv_data=[]
if not os.path.exists(dstdir):
    os.mkdir(dstdir)
for i in range(0,len(list)):
    if(re.match('[0-9][0-9][0-9][0-9][0-9].json',list[i])!=None):
        data={}
        data['im_path']=list[i].split('.')[0]+'.jpg'
        shutil.copy(rootdir+data['im_path'],dstdir+data['im_path'])
        print(list[i].split('.')[0]+'.jpg->copy')
        img=cv2.imread(rootdir+list[i].split('.')[0]+'.jpg')
        im_h=img.shape[0]
        im_w=img.shape[1]
        im_c=img.shape[2]
        data['im_size'] = [im_c,im_w,im_h]
        data['im_net_mask']={'size': [0, 0], 'counts': b''}
        joints=[]
        with open(rootdir+re.match('[0-9][0-9][0-9][0-9][0-9].json',list[i]).group()) as load_f:
            tempjoints=[]
            tempdata=json.load(load_f)
            shapes=tempdata['shapes']
            n = [0, 0, 0]
            for node in shapes:
                if(node['label']=='0' or node['label']=='1' or node['label']=='2'):
                    n[int(node['label'])]=n[int(node['label'])]+1
                    position=node['points'][0]
                    x=position[0]
                    y=position[1]
                    tempjoint=[int(node['label']),x,y]
                    tempjoints.append(tempjoint)
                if(n[0]==n[1] and n[1]==n[2]):
                    joints.append(tempjoints)
                    tempjoints=[]
                    n = [0, 0, 0]
        data['joints']=[joints]
        csv_data.append(data)
        print(re.match('[0-9][0-9][0-9][0-9][0-9].json',list[i]).group()+'->csv')
df=pd.DataFrame(csv_data,columns=['im_path','im_size','im_net_mask','joints'])
df.to_csv("mul_fish_labels.csv",index=False)
