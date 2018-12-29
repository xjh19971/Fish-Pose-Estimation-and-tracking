import argparse
import math
import os
import sys
import time



import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import util
from config_reader import config_reader
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)
PAD = 45
video_process=1
# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[1,2],[2,3]]

# the middle joints heatmap correpondence
mapIdx = [[2,3],[4,5]]

# visualize
colors = [[255, 0, 0], [0, 255, 0],[0, 0, 255]]

input_names=['input_1']
output_names= ['batch_normalization_17/FusedBatchNorm_1','batch_normalization_22/FusedBatchNorm_1']

g1_1 = tf.Graph()
#g1_2 = tf.Graph()
g2 = tf.Graph()
sess1_1 = tf.Session(graph=g1_1)
sess2 = tf.Session(graph=g2)
def predict(oriImg,scale_search,model_params,tf_sess,lenimg=1,flist=None):
    t1 = time.time()
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in scale_search]
    scale = multiplier[0]
    if lenimg==1:
        ROI=np.zeros((1, oriImg.shape[0], oriImg.shape[1],3))
        ROI[0,:,:,:]=oriImg[:,:,:]
        heatmap_avg = np.zeros((lenimg,oriImg.shape[0], oriImg.shape[1], 4))
        paf_avg = np.zeros((lenimg,oriImg.shape[0], oriImg.shape[1], 4))
        orishape=[oriImg.shape[1],oriImg.shape[0]]
    else:
        oriImg_Re= cv2.copyMakeBorder(oriImg,PAD,PAD,PAD,PAD,cv2.BORDER_REPLICATE)
        ROI=np.zeros((len(flist),PAD*2,PAD*2,3))
        for fish in flist:
            ROI[flist.index(fish),:,:,:]=oriImg_Re[fish[3]+PAD:fish[1]+PAD,fish[2]+PAD:fish[0]+PAD,:]
        heatmap_avg = np.zeros((lenimg,PAD*2, PAD*2, 4))
        paf_avg = np.zeros((lenimg,PAD*2, PAD*2, 4))
        orishape = [PAD*2, PAD*2]
    for i in range(0,lenimg):
        imageToTest = cv2.resize(ROI[i,:,:,:], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
        if i==0:
            realinput_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
        else:
            input_img= np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
            realinput_img = np.append(realinput_img,input_img,axis=0)
    tf_input = tf_sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_paf = tf_sess.graph.get_tensor_by_name(output_names[0] + ':0')
    tf_heatmap = tf_sess.graph.get_tensor_by_name(output_names[1] + ':0')
    tf_paf, tf_heatmap = tf_sess.run([tf_paf, tf_heatmap],
                                         feed_dict={
                                             tf_input: realinput_img
                                         })
    output_blobs = [tf_paf, tf_heatmap]
    # extract outputs, resize, and remove padding
    heatmaptemp = output_blobs[1]  # output 1 is heatmaps
    paftemp = output_blobs[0]  # output 0 is PAFs
    for i in range(0, lenimg):
        heatmap = cv2.resize(heatmaptemp[i,:,:,:], (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (orishape[0], orishape[1]), interpolation=cv2.INTER_CUBIC)
        paf = cv2.resize(paftemp[i,:,:,:], (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (orishape[0], orishape[1]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg[i,:,:,:] = heatmap
        paf_avg[i, :, :, :] = paf
    t2 = time.time()
    input = sess2.graph.get_tensor_by_name('input:0')
    output0 = sess2.graph.get_tensor_by_name('output0:0')
    output0 = sess2.run(output0, feed_dict={input: heatmap_avg[:, :,:, 0:3]})
    all_peaks=[[] for i in range(lenimg)]
    check=0
    for i in range(0,lenimg):
        tempall=output0[output0[:, 0] == i]
        temp = tempall[tempall[:, 3] == 0].tolist()
        temp = [[x[1], x[2], x[4], temp.index(x) + check] for x in temp]
        check = check+len(temp)
        all_peaks[i].append(temp)
        temp = tempall[tempall[:, 3] == 1].tolist()
        temp = [[x[1], x[2], x[4], temp.index(x) + check] for x in temp]
        check = check+len(temp)
        all_peaks[i].append(temp)
        temp = tempall[tempall[:, 3] == 2].tolist()
        temp = [[x[1], x[2], x[4], temp.index(x) + check] for x in temp]
        all_peaks[i].append(temp)
        check = check+len(temp)
    t3 = time.time()
    mid_num = 10
    limit=[[80,10],[80,10]]
    subset_all=[]
    candidate_all = []
    checkpoint=0
    for l in range(0,lenimg):
        special_k = []
        connection_all = []
        for k in range(len(mapIdx)):
            score_mid = np.squeeze(paf_avg[l,:, :, [x - 2 for x in mapIdx[k]]]).transpose((1,2,0))
            candA = all_peaks[l][limbSeq[k][0] - 1]
            candB = all_peaks[l][limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            candlist=[]
            if (nA != 0 and nB != 0):
                for i in range(nA):
                    tempcand=[]
                    for j in range(nB):
                        vecx1 = candA[i][0]-candB[j][0]
                        vecy1 = candA[i][1]-candB[j][1]
                        disx = abs(vecx1)
                        disy = abs(vecy1)
                        r = math.sqrt(disx * disx + disy * disy)
                        if(r<limit[k][0] and r>limit[k][1]):
                            tempcand.append(j)
                    candlist.append(tempcand)
                connection_candidate = []
                for i in range(len(candlist)):
                    for j in candlist[i]:
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # failure case when 2 body parts overlaps
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array(
                            [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                            for I in range(len(startend))])
                        vec_y = np.array(
                            [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                            for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        #score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        #    0.5 * oriImg.shape[0] / norm - 1, 0)
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                        criterion1 = len(np.nonzero(score_midpts > 0.01)[0]) > 0.8 * len(
                            score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 5))
        candidate = np.array([item for sublist in all_peaks[l] for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int)-checkpoint, 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int)-checkpoint, 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 2:
                        row = -1 * np.ones(5)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int)-checkpoint, 2]) + \
                              connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # delete some rows of subset which has few parts occur
        deleteIdx = [];
        for i in range(len(subset)):
            if subset[i][-1] < 2 or subset[i][-2] / subset[i][-1] < 0.2:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        subset_all.append(subset)
        candidate_all.append(candidate)
        checkpoint=checkpoint+len(candidate)
    return candidate_all,subset_all,all_peaks,t1,t2,t3
def process (input_image,f, params, model_params,tf_sess,sess2,flist):

    scale_search = [2]

    oriImg = input_image  # B,G,R order
    if f%video_process==0:
        candidate_all, subset_all, all_peaks_all, t1, t2, t3=predict(oriImg,scale_search,model_params,tf_sess)
    else:
        candidate_all, subset_all, all_peaks_all, t1, t2, t3=predict(oriImg,scale_search,model_params,tf_sess,lenimg=len(flist) if len(flist)!=0 else 1,flist=flist)

    canvas = input_image  # B,G,R order

    t4 = time.time()
    stickwidth = 2
    flistnew=[]
    checkpoint=0
    for k in range(len(subset_all)):
        subset = subset_all[k]
        candidate = candidate_all[k]
        all_peaks = all_peaks_all[k]
        if k!=0:
            checkpoint=checkpoint+len(candidate_all[k-1])
        if f%video_process==0:
            locx=0
            locy=0
        else:
            locx=flist[k][2]
            locy =flist[k][3]
        for i in [0, 1, 2]:
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, tuple([int(all_peaks[i][j][0]), int(all_peaks[i][j][1])]), 4, colors[i],
                 thickness=-1)
        for n in range(len(subset)):
            maxx=-1
            maxy=-1
            minx=10000
            miny=10000
            centerx=0
            centery=0
            for i in [0,1,2]:
                idx=int(subset[n][i])
                if int(subset[n][i])!=-1:
                    location=list(map(int,all_peaks[i][int(idx-all_peaks[i][0][3])][0:2]))
                    location[0]=location[0]+locx
                    location[1]=location[1]+locy
                    location=tuple(location)
                    #cv2.circle(canvas,location, 2, colors[i], thickness=-1)
                    if(maxx<location[0]):maxx=location[0]
                    if (maxy < location[1]): maxy = location[1]
                    if (minx > location[0]): minx = location[0]
                    if (miny > location[1]): miny = location[1]
                    if i==1:
                        centerx = location[0]
                        centery = location[1]
            maxx=centerx+PAD
            maxy=centery+PAD
            minx = centerx - PAD
            miny = centery - PAD
            flistnew.append((maxx,maxy,minx,miny))

        for i in range(2):
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int)-checkpoint, 0]+locx
                X = candidate[index.astype(int)-checkpoint, 1]+locy
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    flist=flistnew
    return canvas,t1,t2,t3,t4,flist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='input video file name')
    parser.add_argument('--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('--process_speed', type=int, default=4, help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')

    args = parser.parse_args()

    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    ending_frame = args.end

    print('start processing...')

    # Video input
    video = args.video
    video_path = 'videos/'
    video_file =  video

    # Output location
    output_path = 'videos/outputs/'
    output_format = '.mp4'
    video_output = video + str(start_datetime) + output_format

    # load model
    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images

    # load config
    params, model_params = config_reader()

    # Video reader
    cam = cv2.VideoCapture(video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, input_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame == None:
        ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output,fourcc, output_fps, (int(input_image.shape[1]/2), int(input_image.shape[0]/2)))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction= 0.75
    tf_config.gpu_options.allow_growth = True
    with sess1_1.as_default():
        with sess1_1.graph.as_default():
            output_graph_def = tf.GraphDef()
            with open('tf_model.pb', "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
            init = tf.global_variables_initializer()
            sess1_1.run(init)
            sess1_1 = tf.Session(config=tf_config)
    with sess2.as_default():
        with sess2.graph.as_default():
            map_ori = tf.transpose(tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input'),perm=[0,2,1,3])
            mapshape = tf.shape(map_ori)
            '''w = tf.constant(filt, shape=(3, 3, 1), dtype=tf.float32)
            map_ori2 = tf.expand_dims(map_ori3, -1)
            map_ori1 = tf.expand_dims(map_ori2, 1)
            map_ori = tf.nn.conv2d(map_ori1, w, [1, 1], 'SAME')'''
            a = map_ori[:,1:, :]
            b = map_ori[:,:-1, :]
            c = map_ori[:,:,1:]
            d = map_ori[:,:, :-1]
            padx = tf.zeros([mapshape[0],1, mapshape[2],3])
            pady = tf.zeros([mapshape[0],mapshape[1], 1,3])
            mapx = tf.concat([np.subtract(a, b), padx],1)
            mapy = tf.concat([np.subtract(c, d), pady],2)
            padxb = tf.cast(padx,dtype=tf.bool)
            padyb = tf.cast(pady, dtype=tf.bool)
            tempa=mapx[:,:-1, :] > 0
            tempb=mapx[:,1:, :] < 0
            tempc=mapy[:,:, :-1] > 0
            tempd=mapy[:,:, 1:] < 0
            tempe=map_ori>0.1
            A = tf.expand_dims(tf.concat([tempa,padxb],1),-1)
            B = tf.expand_dims(tf.concat([tempb,padxb],1),-1)
            C = tf.expand_dims(tf.concat([tempc,padyb],2),-1)
            D = tf.expand_dims(tf.concat([tempd,padyb],2),-1)
            E = tf.expand_dims(tempe,-1)
            mask= tf.reduce_all(tf.concat([A,B,C,D,E],4),4)
            xy=tf.where(mask)
            floatxy=tf.cast(xy,tf.float32)
            score=tf.expand_dims(tf.cast(tf.gather_nd(map_ori,xy),dtype=tf.float32),1)
            output=tf.concat([floatxy,score],1,name='output0')
            init = tf.global_variables_initializer()
            sess2.run(init)
            sess2 = tf.Session(config=tf_config)
    i = 0 # default is 0
    flist=[]
    while(cam.isOpened()) and ret_val == True and i < ending_frame:
        if i%frame_rate_ratio == 0:
            scale=0.5
            input_image = cv2.resize(input_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            tic = time.time()
                    # generate image with body parts
            canvas,t1,t2,t3,t4,flist = process(input_image,i, params, model_params,sess1_1,sess2,flist)
            '''cv2.imshow('canvas',canvas)
            cv2.waitKey(1)'''
            print('Processing frame: ', i)
            toc = time.time()
            print ('processing time is %.5f' % (toc - tic))
            print('processing time is '+str(t1-tic)+str(t2-t1)+str(t3-t2)+str(t4-t3)+str(toc-t4))
            out.write(canvas)
        ret_val, input_image = cam.read()
        i += 1
    sess1_1.close()
    sess2.close()
