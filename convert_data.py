import cv2
import pandas as pd
import numpy as np
import pickle
import json
import codecs

dataset = {}
def _to_one_hot(name):
    one_hot_vector = [0] * 3
    if name == 0:
        one_hot_vector[0] = 1
    elif name == 1:
        one_hot_vector[1] = 1
    elif name == 2:
        one_hot_vector[2] = 1
    return one_hot_vector

df=pd.read_csv('/home/xjh/mul_fish_labels.csv')
EXPAND=6
name=df['im_path'].str[-9:]
joint=df['joints'].str[:]
size=df['im_size'].str[:]
name=name.tolist()
size=size.tolist()
joint=joint.tolist()
myjointsint=[[] for i in range(len(joint))]
mysize=[]


for onesize in size:
    newsize=onesize.replace('[','').replace(']','').split(',')
    mysize.append(newsize)
#make the images for coco
images = []
for myname in name:
    temp = {}
    temp['width'] = int(mysize[name.index(myname)][1])
    temp['height'] = int(mysize[name.index(myname)][2])
    temp['file_name'] =myname
    temp['id'] = myname[:-4]
    images.append(temp)

dataset['images'] = images

#make the categories for coco
cates_new = []
temp ={}
temp['id']=1
temp['name']='fish'
temp['keypoints']=['top','body','tail']
temp['skeleton']=[[1,2],[2,3]]
cates_new.append(temp)
dataset['categories'] = cates_new

#make the anotations for coco
Annotations=[]
sum_n=0
for myjoints in joint:
    myjoint=myjoints.replace('[','').replace(']','').replace(' ','').split(',')
    myjointintalone=list(map(eval, myjoint))
    n=0
    points=[]
    l3 = {}
    maxx = -1
    minx = 10000
    maxy = -1
    miny = 10000
    for num in range(0,len(myjointintalone),3):
        temp=myjointintalone[num:num + 3]
        x=temp[1]
        y=temp[2]
        v=2
        points.append(x)
        points.append(y)
        points.append(v)
        if (x < minx): minx = x
        if (x > maxx): maxx = x
        if (y < miny): miny = y
        if (y > maxy): maxy = y
        if(n==2):
            n=0
            sum_n=sum_n+1
            l3['keypoints']=points
            l3['image_id'] = name[joint.index(myjoints)][:-4]
            l3['category_id'] = 1
            l3['id'] = sum_n
            l3['segmentation'] = [[minx, miny, 2, 2, 3, 3]]
            l3['bbox']=[minx,miny,maxx-minx,maxy-miny]
            l3['num_keypoints'] = 3
            l3['iscrowd'] = 0
            l3['area']=1000
            Annotations.append(l3)
            l3 = {}
            points=[]
            maxx = -1
            minx = 10000
            maxy = -1
            miny = 10000
        else:
            n=n+1

dataset['annotations']=Annotations

with codecs.open('my_person_keypoints.json', 'w', 'utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)