import numpy as np
import cv2
import time
import sys

def face_and_eye_detect(image_name):
    face_cascade = cv2.CascadeClassifier("/home/pi/.virtualenvs/cv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("/home/pi/.virtualenvs/cv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml")
    fullbody_cascade = cv2.CascadeClassifier("/home/pi/.virtualenvs/cv/lib/python3.7/site-packages/cv2/data/haarcascade_fullbody.xml")
    upperbody_cascade = cv2.CascadeClassifier("/home/pi/.virtualenvs/cv/lib/python3.7/site-packages/cv2/data/haarcascade_upperbody.xml")
    img = cv2.imread("/home/pi/Desktop/{}".format(image_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    fullbodies = fullbody_cascade.detectMultiScale(gray,1.3,5)
    upperbodies = upperbody_cascade.detectMultiScale(gray,1.3,5)
    print(time.time()-start)
    for rect in fullbodies:
        cv2.rectangle(img,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),(0,0,255),2)
    for rect in upperbodies:
        cv2.rectangle(img,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),(255,255,255),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.imshow('img',gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = sys.argv
    image_name = args[1]

    face_and_eye_detect(image_name)
