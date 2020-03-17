from server.streamCapture import StreamCapture_socket
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--record', action="store_true")
parser.add_argument('--frame_rate', type=int, default=24)
parser.add_argument('--duration_time', type=int, default=20)

args = parser.parse_args()


cap = StreamCapture_socket('192.168.137.124', 8000)

if args.record:
    _, test_frame = cap.read()
    frame_W = test_frame.shape[1]
    frame_H = test_frame.shape[0]
    
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), args.frame_rate, (frame_W, frame_H))

    ret = True
    frame_count = 0
    while ret and frame_count < args.duration_time * args.frame_rate:
        ret, frame = cap.read()
        out.write(frame)
        frame_count += 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    out.release()
    
else:
    #Stream
    ret = True
    count = 0
    if not os.path.exists('output'):
        os.makedirs('output')
    while ret:
        ret, img = cap.read()
        
        cv2.imshow('img', img)

        key = cv2.waitKey(30)
        if key == 27:
            exit(0)
        elif key == 32:
            path = 'output/image_{}.jpg'.format(count)
            cv2.imwrite(path,img)
            print('saving image at {}'.format(path))
            count += 1

