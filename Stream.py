from server.streamCapture import StreamCapture_socket
import cv2

cap = StreamCapture_socket('192.168.137.164', 8000)

ret = True

while ret:
    ret, img = cap.read()
    
    cv2.imshow('img', img)

    if cv2.waitKey(30) == 27:
        exit(0)