import io
import socket
import struct
import time
import picamera

try:
    camera = picamera.PiCamera()
    camera.resolution = (640,480)
    camera.start_preview()
    time.sleep(2)

    start = time.time()
    stream = io.BytesIO()

    for foo in camera.capture_continuous(stream,'jpeg'):
        a = stream.tell()
        s = struct.pack('<L',a)
        print(a)
        print(s)
        print(type(a))
        print(type(s))
finally:
    pass
