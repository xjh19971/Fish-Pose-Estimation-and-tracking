import cv2
import pandas as pd
import numpy as np
import pickle
import json
import codecs

def computeOks(gts,dts):
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
            e = (dx ** 2 + dy ** 2) / 50
            ious[j, k] = np.sum(np.exp(-e))
    iousmax = ious.max(axis=1)
    return np.sum(iousmax)/gts.shape[0]

if __name__ == '__main__':
    dataset = {}
    df=pd.read_csv('./val.csv')
    gf=pd.read_csv('./mul_fish_labels.csv')
    gnn=gf['im_path'].str[:]
    gtt=gf['joints'].str[:]
    dnn=df['im_path'].str[:]
    dtt=df['joints'].str[:]
    gnn=gnn.tolist()
    gtt=gtt.tolist()
    dnn=dnn.tolist()
    dtt=dtt.tolist()
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
    for dt in dtt:
        dts1 = []
        dts2 = []
        dts3 = []
        myjoint = dt.replace('[', '').replace(']', '').replace(' ', '').split(',')
        if myjoint == ['']:
            dts.append([])
            score.append([0])
            continue
        myjointintalone = list(map(eval, myjoint))
        scorealone=[]
        num=0
        while num < len(myjointintalone):
            dts1temp = []
            dts2temp = []
            dts3temp = []
            lost=-1
            tempnum=num
            if myjointintalone[num+7]==2:
                if myjointintalone[num+2]==1:
                    lost=0
                else:
                    lost=2
            if lost==0:
                dts1temp.append(100000)
                dts1temp.append(100000)
            else:
                temp = myjointintalone[tempnum:tempnum + 3]
                tempnum=tempnum+3
                dts1temp.append(temp[0])
                dts1temp.append(temp[1])
            temp = myjointintalone[tempnum:tempnum + 3]
            tempnum=tempnum+3
            dts2temp.append(temp[0])
            dts2temp.append(temp[1])
            if lost == 2:
                dts3temp.append(100000)
                dts3temp.append(100000)
            else:
                temp = myjointintalone[tempnum:tempnum + 3]
                tempnum = tempnum + 3
                dts3temp.append(temp[0])
                dts3temp.append(temp[1])
            if lost==-1:
                scorealone.append(myjointintalone[num+9])
            else:
                scorealone.append(myjointintalone[num + 6])
            dts1.append(dts1temp)
            dts2.append(dts2temp)
            dts3.append(dts3temp)
            if lost!=-1:
                num=num-3
            num=num+11
        dts.append([dts1,dts2,dts3])

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
            iousmax[j]=computeOks(gts[j],dts[j])
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




