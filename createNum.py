import os
import re
rootdir='./train_data/'
list = os.listdir(rootdir)
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if(re.match(r"[1-9][0-9][0-9][0-9].jpg",list[i])!=None):
        print(list[i]+'->0'+list[i])
        os.rename(path,os.path.join(rootdir,'0'+list[i]))
    if(re.match(r"[1-9][0-9][0-9].jpg",list[i])!=None):
        print(list[i]+'->00'+list[i])
        os.rename(path,os.path.join(rootdir,'00'+list[i]))
    if(re.match(r"[1-9][0-9].jpg",list[i])!=None):
        print(list[i]+'->000'+list[i])
        os.rename(path, os.path.join(rootdir, '000' + list[i]))
    if(re.match(r"[0-9].jpg",list[i])!=None):
        print(list[i]+'->0000'+list[i])
        os.rename(path, os.path.join(rootdir, '0000' + list[i]))
