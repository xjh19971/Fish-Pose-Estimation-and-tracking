import cv2
import pandas as pd
import numpy as np
import pickle
import json
import codecs

def computeOks(gts,dts,score):
    # dimention here should be Nxm
    inds = np.argsort([-d for d in score], kind='mergesort')
    dts = [dts[i] for i in inds]
    # if len(gts) == 0 and len(dts) == 0:
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = np.array([.3, .3, .3]) / 10.0
    vars = (sigmas * 2) ** 2
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt)
        xg = g[0::2]
        yg = g[1::2]
        for i, dt in enumerate(dts):
            d = np.array(dt)
            xd = d[0::2]
            yd = d[1::2]
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
            e = (dx ** 2 + dy ** 2) / vars / (2500 + np.spacing(1)) / 2
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious

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
        myjoint = gt.replace('[', '').replace(']', '').replace(' ', '').split(',')
        myjointintalone = list(map(eval, myjoint))
        gtsalone=[]
        for num in range(0, len(myjointintalone), 9):
            gtstemp = []
            tempnum=num
            temp = myjointintalone[tempnum:tempnum + 3]
            tempnum=tempnum+3
            gtstemp.append(temp[1])
            gtstemp.append(temp[2])
            temp = myjointintalone[tempnum:tempnum + 3]
            tempnum=tempnum+3
            gtstemp.append(temp[1])
            gtstemp.append(temp[2])
            temp = myjointintalone[tempnum:tempnum + 3]
            gtstemp.append(temp[1])
            gtstemp.append(temp[2])
            gtsalone.append(gtstemp)
        gts.append(gtsalone)
    dts=[]
    score=[]
    for dt in dtt:
        myjoint = dt.replace('[', '').replace(']', '').replace(' ', '').split(',')
        if myjoint == ['']:
            continue
        myjointintalone = list(map(eval, myjoint))
        dtsalone=[]
        scorealone=[]
        num=0
        while num < len(myjointintalone):
            dtstemp = []
            lost=-1
            tempnum=num
            if myjointintalone[num+7]==2:
                if myjointintalone[num+2]==1:
                    lost=0
                else:
                    lost=2
            if lost==0:
                dtstemp.append(100000)
                dtstemp.append(100000)
            else:
                temp = myjointintalone[tempnum:tempnum + 3]
                tempnum=tempnum+3
                dtstemp.append(temp[0])
                dtstemp.append(temp[1])
            temp = myjointintalone[tempnum:tempnum + 3]
            tempnum=tempnum+3
            dtstemp.append(temp[0])
            dtstemp.append(temp[1])
            if lost == 2:
                dtstemp.append(100000)
                dtstemp.append(100000)
            else:
                temp = myjointintalone[tempnum:tempnum + 3]
                tempnum = tempnum + 3
                dtstemp.append(temp[0])
                dtstemp.append(temp[1])
            if lost==-1:
                scorealone.append(myjointintalone[num+9])
            else:
                scorealone.append(myjointintalone[num + 6])
            dtsalone.append(dtstemp)
            if lost!=-1:
                num=num-3
            num=num+11
        dts.append(dtsalone)
        score.append(scorealone)
    ious=[]
    gtt=gts
    dtt=dts
    APthre1=[i for i in np.arange(0.05,0.95,0.05)]
    realmAP=[]
    for i,dn in enumerate(dnn):
        gind=gnn.index(dn)
        gts=gtt[gind]
        dts=dtt[i]
        ious=computeOks(gts,dts,score[i])
        iousmax=ious.max(axis=1)
        mAP = 0
        for APthre in APthre1:
            AP=len((np.where(iousmax>APthre,iousmax,0)).nonzero())/iousmax.shape[0]
            mAP=mAP+AP
        mAP=mAP/len(APthre1)
        realmAP.append(mAP)
    print(realmAP)




