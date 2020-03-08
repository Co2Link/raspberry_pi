import requests
import socket
import cv2
import numpy as np

class StreamCapture_http:

    def __init__(self,url):
        self.r = requests.get(url,auth=('user','password'),stream=True)
        assert self.r.status_code == 200,'connection fail'

        self.gen = self._frame_generator()
        self.bytes = bytes()
    
    def _frame_generator(self):
        for chunk in self.r.iter_content(chunk_size=1024):
            self.bytes += chunk
            a = self.bytes.find(b'\xff\xd8')
            b = self.bytes.find(b'\xff\xd9')
            if a != -1 and b!= -1:
                jpg = self.bytes[a:b+2]
                self.bytes = self.bytes[b+2:]
                yield cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

    def read(self):
        try:
            img = next(self.gen)
            return True,img
        except Exception as e:
            return False,None


class StreamCapture_socket:

    def __init__(self,ipaddr,port):
        
        self.client_socket = socket.socket()
        self.client_socket.connect((ipaddr,port))
        self.conn = self.client_socket.makefile('rb')
        self.gen = self._frame_generator()
        self.bytes = bytes()

    def _frame_generator(self):
        try:
            while True:
                self.bytes += self.conn.read(1024)
                # All jpeg frames start with marker 0xff 0xd8 and end with 0xff 0xd9
                a = self.bytes.find(b'\xff\xd8')
                b = self.bytes.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = self.bytes[a:b+2]
                    self.bytes = self.bytes[b+2:]
                    yield cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        finally:
            self.conn.close()
            server_socket.close()

    def read(self):
        try:
            img = next(self.gen)
            return True,img
        except Exception as e:
            print(e)
            return False,None

