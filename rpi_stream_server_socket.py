import io
import socket
import struct
import numpy as np
import cv2


server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)
connection = server_socket.accept()[0].makefile('rb')

bytes = bytes()
try:
    while True:
        bytes += connection.read(1024)
        # All jpeg frames start with marker 0xff 0xd8 and end with 0xff 0xd9
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('i', i)
            if cv2.waitKey(1) == 27:
                exit(0)
finally:
    connection.close()
    server_socket.close()
    exit(0)