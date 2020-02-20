import math
from os import getcwd

import pandas as pd

dataset = {}

df = pd.read_csv('real.csv')
name = df['im_path'].str[-9:]
joint = df['joints'].str[:]
size = df['im_size'].str[:]
name = name.tolist()
size = size.tolist()
joint = joint.tolist()
myjointsint = [[] for i in range(len(joint))]
mysize = []

for onesize in size:
    newsize = onesize.replace('[', '').replace(']', '').split(',')
    mysize.append(newsize)
# make the images for coco
images = []
for myname in name:
    temp = {}
    temp['width'] = int(mysize[name.index(myname)][1])
    temp['height'] = int(mysize[name.index(myname)][2])
    temp['file_name'] = myname
    temp['id'] = myname[:-4]
    images.append(temp)

dataset['images'] = images

# make the categories for coco
cates_new = []
temp = {}
temp['id'] = 1
temp['name'] = 'fish'
temp['keypoints'] = ['top', 'body', 'tail']
temp['skeleton'] = [[1, 2], [2, 3]]
cates_new.append(temp)
dataset['categories'] = cates_new

# make the anotations for coco
Annotations = []
sum_n = 0
for myjoints in joint:
    myjoint = myjoints.replace('[', '').replace(']', '').replace(' ', '').split(',')
    myjointintalone = list(map(eval, myjoint))
    n = 0
    points = []
    l3 = {}
    maxx = -1
    minx = 10000
    maxy = -1
    miny = 10000
    for num in range(0, len(myjointintalone), 3):
        temp = myjointintalone[num:num + 3]
        x = temp[1]
        y = temp[2]
        v = 2
        points.append(x)
        points.append(y)
        points.append(v)
        if (x < minx): minx = x
        if (x > maxx): maxx = x
        if (y < miny): miny = y
        if (y > maxy): maxy = y
        if (n == 2):
            MINXY = 0
            MAXX = 752
            MAXY = 480
            EXPANDY = 12
            EXPANDX = 12
            if minx - EXPANDX <= MINXY:
                minx = 0
            else:
                minx = minx - EXPANDX
            if miny - EXPANDY < MINXY:
                miny = 0
            else:
                miny = miny - EXPANDY
            if maxx + EXPANDX >= MAXX:
                maxx = MAXX
            else:
                maxx = maxx + EXPANDX
            if maxy + EXPANDY >= MAXY:
                maxy = MAXY
            else:
                maxy = maxy + EXPANDY
            h = maxy - miny
            w = maxx - minx
            if h > 160:
                miny += math.ceil((h - 160) / 2)
                maxy -= math.ceil((h - 160) / 2)
            if w > 160:
                minx += math.ceil((w - 160) / 2)
                maxx -= math.ceil((w - 160) / 2)
            n = 0
            sum_n = sum_n + 1
            l3['keypoints'] = points
            l3['image_id'] = name[joint.index(myjoints)][:-4]
            l3['category_id'] = 1
            l3['id'] = sum_n
            l3['segmentation'] = [[minx, miny, 2, 2, 3, 3]]
            l3['bbox'] = [minx, miny, maxx, maxy]
            l3['num_keypoints'] = 3
            l3['iscrowd'] = 0
            l3['area'] = 1000
            Annotations.append(l3)
            l3 = {}
            points = []
            maxx = -1
            minx = 10000
            maxy = -1
            miny = 10000
        else:
            n = n + 1

dataset['annotations'] = Annotations
wd = getcwd()
with open("gt.txt", "w") as f:
    tempname = ""
    tempstr = ""
    first = False
    for annotation in dataset['annotations']:
        if annotation['image_id'] != tempname:
            if first == True:
                tempstr = tempstr + '\n'
                f.write(tempstr)
            tempstr = wd + "\\dataset\\train_data\\" + annotation['image_id'] + '.jpg'
            tempname = annotation['image_id']
        first = True
        bbox = annotation['bbox']
        keypoints = annotation['keypoints']
        tempstr = tempstr + ' ' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(
            bbox[3]) + ',1,' + str(keypoints[0]) + \
                  ',' + str(keypoints[1]) + ',' + str(keypoints[3]) + ',' + str(keypoints[4]) + ',' + str(
            keypoints[6]) + ',' \
                  + str(keypoints[7])
    f.close()