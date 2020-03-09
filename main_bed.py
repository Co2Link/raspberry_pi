import cv2

from server.streamCapture import StreamCapture_socket
from server.BedDetector import BedDetector


cap = StreamCapture_socket('192.168.137.164', 8000)

detector = BedDetector(kernelSize=(20,20))

ret = True

while ret:
    ret, img = cap.read()
    
    detector.detect(img)