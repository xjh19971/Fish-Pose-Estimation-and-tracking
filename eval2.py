import numpy as np

sigma = [32, 32, 50]


def computeOks(gts, dts):
    # dimention here should be Nxm
    ious = np.zeros((len(gts), len(dts)))
    iousde = [np.zeros((len(gts), len(dts))) for i in range(3)]

    gts = np.array(gts)
    dts = np.array(dts)
    for j, gt in enumerate(gts):
        g = [gt[0:2], gt[2:4], gt[4:6]]
        for k, dt in enumerate(dts):
            d = [dt[0:2], dt[2:4], dt[4:6]]
            ioustemp = 0
            for l in range(3):
                xd = d[l][0]
                yd = d[l][1]
                if xd == -1 and yd == -1:
                    continue
                xg = g[l][0]
                yg = g[l][1]
                dx = xd - xg
                dy = yd - yg
                e = (dx ** 2 + dy ** 2) / sigma[l]
                ioustemp += np.exp(-e)
                iousde[l][j, k] = np.exp(-e)
            ious[j, k] = ioustemp / 3
    iousdemax = np.zeros(shape=(len(gts),3))
    if ious.size == 0:
        iousmax = np.array([0])
        return iousmax,iousdemax
    else:
        iousmax = np.max(ious,axis=1)
        index=np.argmax(ious,axis=1)

    for n in range(len(gts)):
        for i in range(3):
            iousdemax[n,i]=iousde[i][n,index[n]]
    return iousmax, iousdemax


if __name__ == '__main__':
    dataset = {}
    dtslist = []
    gtslist = []
    with open('./dt.txt') as f:
        df = f.readlines()
        for i in range(len(df)):
            dline = df[i].split()
            dts = np.array([np.array(list(map(int, data.split(',')))) for data in dline[1:]])
            dtslist.append(dts)
    with open('./gt.txt') as g:
        gf = g.readlines()
        for i in range(len(gf)):
            gline = gf[i].split()
            gts = np.array([np.array(list(map(int, data.split(',')))) for data in gline[1:]])
            gtslist.append(gts[:, 5:])

    ious = []
    APthreall = [i for i in np.arange(0.50, 1, 0.05)]
    iousmaxall = []
    iousdemaxall = []
    for i in range(len(dtslist)):
        gts = gtslist[i]
        dts = dtslist[i]
        if dts == []:
            continue
        ious, iousde = computeOks(gtslist[i], dtslist[i])
        iousmaxall.append(ious)
        iousdemaxall.append(iousde)
    AP = 0
    APde = [0 for i in range(3)]
    for APthre in APthreall:
        count = 0
        size = 0
        countde = np.zeros(shape=(3,))
        sizede = np.zeros(shape=(3,))
        for img in iousmaxall:
            for ind in img:
                if ind >= APthre:
                    count = count + 1
                size += 1
        for i in range(len(iousdemaxall)):
            for ind in iousdemaxall[i]:
                for j in range(3):
                    if ind[j] >= APthre:
                        countde[j] = countde[j] + 1
            sizede += len(iousdemaxall[i])
        for i in range(3):
            APde[i] = APde[i] + countde[i] / sizede[i]
        if APthre == .50:
            print('AP50=' + str(count / size) + '\n')
        if APthre >= .75 and APthre < 0.80:
            print('AP75=' + str(count / size) + '\n')
        AP = AP + count / size
    mAP = AP / len(APthreall)
    print('AP=' + str(mAP) + '\n')
    avgAP=0
    for i in range(3):
        avgAP+=APde[i] / len(APthreall)
        print('AP' + str(i) + '=' + str(APde[i] / len(APthreall)) + '\n')
    print('avgAP='+str(avgAP/3))