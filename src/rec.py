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
    args = parse_args()
    print(args)
    img = cv2.imread(args.image_path,0)
    if img is None:
        print("Cannot read image")
        sys.exit(1)

    # orb = cv2.ORB_create(nfeatures=400)
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)

    sobel_img = cv2.Laplacian(img,cv2.CV_8U)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    cv2.imshow('image2',img2)
    cv2.imshow('image',img)
    cv2.imshow('image_sobel',sobel_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
