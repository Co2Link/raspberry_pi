import cv2
import picamera
import argparse
import time
from picamera.array import PiRGBArray

ap = argparse.ArgumentParser()
ap.add_argument('-o','--output',type=str,default='output.avi')
ap.add_argument('-res','--resolution',type=str,default='640x480')
ap.add_argument('--frame_rate',type=int,default=30)

args = vars(ap.parse_args())

RESOLUTION = tuple([int(i)for i in args['resolution'].split('x')])
FRAME_RATE = args['frame_rate']

cam = picamera.PiCamera()
cam.resolution = RESOLUTION
cam.framerate = FRAME_RATE

rawCapture = PiRGBArray(cam, size=RESOLUTION)

time.sleep(1)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(args['output'],fourcc,FRAME_RATE,RESOLUTION,True)

for frame in cam.capture_continuous(rawCapture, format='bgr',use_video_port=True):
    writer.write(frame.array)
    rawCapture.truncate(0)
    
    cv2.imshow('frame',frame.array)
    if cv2.waitKey(1) == ord('q'):
        break

writer.release()


