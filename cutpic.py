#!/bin/env python
#-*- encoding=utf8 -*-

import cv2
import os

if __name__=="__main__":
    video_path='./4.avi'                    
    image_path='./train_data_cut/'
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError(("Couldn't open video file or webcam. If you're "
                       "trying to open a webcam, make sure you video_path is an integer!"))
    vidw = 750
    vidh = 480
    n = 0
    sum_n=579							# start from 579 
    while True:
        retval, orig_image = vid.read()
        if not retval:
            print("Done!")
            break
        resize_image=cv2.resize(orig_image,(vidw,vidh),interpolation=cv2.INTER_CUBIC)
        if (n == 10):						# per 10 -> 1 pic
            cv2.imwrite(image_path+str(sum_n)+'.jpg',resize_image)
            n=0
            sum_n=sum_n+1
        else:
            n = n + 1

