import cv2
import pandas as pd
import numpy as np
import pickle
import json
import codecs
sigma=[32,50,32]
def computeOks(gts,dts,i):
    # dimention here should be Nxm
    ious = np.zeros((len(gts), len(dts)))
    gts=np.array(gts)
    dts = np.array(dts)
    for j, gt in enumerate(gts):
        xg = gt[0]
        yg = gt[1]

        for k, dt in enumerate(dts):
            xd = dt[0]
            yd = dt[1]

            dx = xd - xg
            dy = yd - yg
            e = (dx ** 2 + dy ** 2) / sigma[i]
            ious[j, k] = np.sum(np.exp(-e))
    if ious.size==0:
        iousmax=0
    else:
        iousmax = ious.max(axis=1)
    return np.sum(iousmax)/gts.shape[0]

if __name__ == '__main__':
    dataset = {}
    df=pd.read_csv('./val.csv')
    gf=pd.read_csv('./mul_fish_labels.csv')
    gnn=gf['im_path'].str[:]
    gtt=gf['joints'].str[:]
    dnn=df['im_path'].str[:]
    dt1 = df['joints1'].str[:]
    dt2 = df['joints2'].str[:]
    dt3 = df['joints3'].str[:]
    dtt = []
    gnn=gnn.tolist()
    gtt=gtt.tolist()
    dnn=dnn.tolist()
    dtt.append(dt1.tolist())
    dtt.append(dt2.tolist())
    dtt.append(dt3.tolist())
    gts=[]
    for gt in gtt:
        gts1 = []
        gts2 = []
        gts3 = []
        myjoint = gt.replace('[', '').replace(']', '').replace(' ', '').split(',')
        myjointintalone = list(map(eval, myjoint))
        for num in range(0, len(myjointintalone), 9):
            gts1temp = []
            gts2temp = []
            gts3temp = []
            tempnum=num
            temp = myjointintalone[tempnum:tempnum + 3]
            tempnum=tempnum+3
            gts1temp.append(temp[1])
            gts1temp.append(temp[2])
            temp = myjointintalone[tempnum:tempnum + 3]
            tempnum=tempnum+3
            gts2temp.append(temp[1])
            gts2temp.append(temp[2])
            temp = myjointintalone[tempnum:tempnum + 3]
            gts3temp.append(temp[1])
            gts3temp.append(temp[2])
            gts1.append(gts1temp)
            gts2.append(gts2temp)
            gts3.append(gts3temp)
        gts.append([gts1,gts2,gts3])
    dts=[]
    score=[]
    for i in range(len(dtt[0])):
        mydts = []
        mydt=[dtt[0][i],dtt[1][i],dtt[2][i]]
        for dt in mydt:
            dtstemp=[]
            myjoint = dt.replace('[', '').replace(']', '').replace(' ', '').replace('(', '').replace(')', '').split(',')
            if myjoint==['']:
                mydts.append([])
                continue
            myjointintalone = list(map(eval, myjoint))
            for num in range(0, len(myjointintalone), 2):
                dts1temp = []
                temp = myjointintalone[num:num + 2]
                dts1temp.append(temp[0])
                dts1temp.append(temp[1])
                dtstemp.append(dts1temp)
            mydts.append(dtstemp)
        dts.append(mydts)

    ious=[]
    gtt=gts
    dtt=dts
    APthre1=[i for i in np.arange(0.05,1,0.05)]
    iousmaxall=[]
    for i,dn in enumerate(dnn):
        gind=gnn.index(dn)
        gts=gtt[gind]
        dts=dtt[i]
        iousmax=np.zeros(3)
        if dts == []:
            continue
        for j in range(0,3):
            iousmax[j]=computeOks(gts[j],dts[j],j)
        iousmaxall.append(iousmax)
    '''for APthre in APthre1:
        for OKS in iousmax[j]:'''
    AP=0
    for APthre in APthre1:
        count=np.zeros(3)
        for img in iousmaxall:
            for j in range(0,3):
                if img[j]>=APthre:
                    count[j]=count[j]+1
        AP=AP+count/len(iousmaxall)
    mAP=AP/len(APthre1)
    print(mAP)




