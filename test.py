import cv2
import numpy as np
import glob

img_paths = glob.glob('data/images/*')

fld = cv2.ximgproc.createFastLineDetector(_length_threshold=150,_distance_threshold=30,_do_merge=True)
fld_ = cv2.ximgproc.createFastLineDetector(_length_threshold=150,_distance_threshold=30)

for img_path in img_paths:

    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = fld.detect(img_gray)
    lines_ = fld_.detect(img_gray)
    print(lines.shape, lines_.shape)
    drawn_img = fld.drawSegments(img,lines)
    drawn_img_ = fld.drawSegments(img,lines_)

    cv2.imshow('drawn_img',drawn_img)
    cv2.imshow('drawn_img_',drawn_img_)
    cv2.waitKey(0)
