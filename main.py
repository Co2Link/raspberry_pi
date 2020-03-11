import cv2
import requests
import numpy as np
import os
import time
import sys
import argparse

from server.utils import Frame_rate_calculator
from server.detector import Detector
from server.streamCapture import StreamCapture_socket
from server.face_reco import Face_wrapper
from server.BedDetector import BedDetector

def write_text(src, text):
    return cv2.putText(src,text,(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

def draw_bed(src,intxn_points):
    cv2.line(src,intxn_points['p1'],intxn_points['p2'],(0,255,255),3)
    cv2.line(src,intxn_points['p2'],intxn_points['p3'],(0,255,255),3)
    cv2.line(src,intxn_points['p3'],intxn_points['p4'],(0,255,255),3)
    cv2.line(src,intxn_points['p1'],intxn_points['p4'],(0,255,255),3)
    

def main():
    sys.path.append('F:/openpose/build/python/openpose/Release')
    os.environ['PATH']  = os.environ['PATH'] + ';' + 'F:/openpose/build/x64/Release;' + 'F:/openpose/build/bin;'
    import pyopenpose as op

    params = {'model_folder':'F:/openpose/models', 'render_threshold':0.4}

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # face_recognition
    # fw = Face_wrapper('images/test')

    fc = Frame_rate_calculator()

    cap = StreamCapture_socket('192.168.137.84', 8000)

    ret = True

    fc.start_record()

    bed_detector = BedDetector()

    count = 0

    while ret:
        ret, frame = cap.read()
        intxn_points, success, show_img = bed_detector.detect(frame)

        write_text(show_img, 'Detecting bed')
        cv2.imshow('i', show_img)
        if cv2.waitKey(1) == 27:
            exit(0)
        count += 1
        if success and count > 30:
            break

    # detector
    detector = Detector(bed_location = intxn_points, confidence=0.4)

    while ret:
        ret,frame = cap.read()

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # # detect
        detector.get_raw_data(datum.poseKeypoints)
        img = np.copy(datum.cvOutputData)

        fc.frame_end()

        write_text(img,detector.get_description())
        cv2.putText(img,'FPS:'+str(fc.get_frame_rate()),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        draw_bed(img, intxn_points)
        
        time1=time.time()
        # img = fw.recognize(frame,img)
        # print(time.time()-time1)
        cv2.imshow('i', img)
        
        if cv2.waitKey(1) == 27:
            exit(0)

if __name__ == "__main__":
    main()