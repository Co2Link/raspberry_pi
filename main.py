import cv2
import requests
import numpy as np
import os
import time
import sys
import argparse

from server.utils import Frame_rate_calculator
from server.detector import Detector
from server.streamCapture import StreamCapture_http
from server.face_reco import Face_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--url',type=str,default='http://172.20.10.10:8000/stream.mjpg')
args = parser.parse_args()


def main():
    sys.path.append('F:/openpose/build/python/openpose/Release')
    os.environ['PATH']  = os.environ['PATH'] + ';' + 'F:/openpose/build/x64/Release;' + 'F:/openpose/build/bin;'
    import pyopenpose as op

    params = {'model_folder':'F:/openpose/models'}

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # detector
    detector = Detector()

    # face_recognition
    fw = Face_wrapper('images/test')

    fc = Frame_rate_calculator()

    cap = StreamCapture_http(args.url)

    ret = True

    fc.start_record()

    while ret:
        ret,frame = cap.read()

        

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # # detect
        detector.get_raw_data(datum.poseKeypoints)
        img = np.copy(datum.cvOutputData)

        fc.frame_end()

        cv2.putText(img,detector.get_description(),(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,'FPS:'+str(fc.get_frame_rate()),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        
        time1=time.time()
        img = fw.recognize(frame,img)
        print(time.time()-time1)
        cv2.imshow('i', img)
        
        if cv2.waitKey(1) == 27:
            exit(0)


if __name__ == "__main__":
    main()