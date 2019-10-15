#! /usr/bin/env python3

import sys
import argparse
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    return parser.parse_args()


def main():
    # args = parse_args()
    # print(args)
    template = cv2.imread("photos/true/cropped/img1238.jpeg",0)
    orb = cv2.ORB_create()
    template_kp = orb.detect(template ,None)
    template_kp, template_des = orb.compute(template, template_kp)

    bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True)
    # if img is None:
    #     print("Cannot read image")
    #     sys.exit(1)

    # orb = cv2.ORB_create(nfeatures=400)
    # orb = cv2.ORB_create()
    # find the keypoints with ORB
    # kp = orb.detect(img,None)

    # sobel_img = cv2.Laplacian(img,cv2.CV_8U)

    # compute the descriptors with ORB
    # kp, des = orb.compute(img, kp)
    # img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("video", 1)
    while(1):
        ret, frame = cap.read()

        if ret == True:
            kp = orb.detect(frame ,None)
            kp, des = orb.compute(frame, kp)
            if des is None:
                continue
            matches = bf.match(template_des, des)
            matches = sorted(matches, key = lambda x:x.distance)
            count = min(7, len(matches))
            out = cv2.drawMatches(template ,template_kp,frame, kp, matches[:count], None, flags=2)
            # out = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
            cv2.imshow('video', out)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

        else:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
