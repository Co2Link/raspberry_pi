import io
import socket
import struct
import time
import picamera
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resolution',type=str,default='640x480',help='WxH')
parser.add_argument('--frame_rate',type=int,default=30)

args = parser.parse_args()

RESOLUTION = args.resolution
FRAME_RATE = args.frame_rate

class SplitFrames(object):
    def __init__(self, connection):
        self.connection = connection
        self.stream = io.BytesIO()
        self.count = 0

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; send the old one's length
            # then the data
            size = self.stream.tell()
            if size > 0:
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                self.connection.write(self.stream.read(size))
                self.count += 1
                self.stream.seek(0)
        self.stream.write(buf)

def listenToClient(client,address):
    connection = client.makefile('wb')
    try:
        output = SplitFrames(connection)
        with picamera.PiCamera(resolution=RESOLUTION, framerate=FRAME_RATE) as camera:
            time.sleep(2)
            start = time.time()
            camera.start_recording(output, format='mjpeg')
            camera.wait_recording(1000000)
    except ConnectionResetError as e:
        print('connection closed by client({})'.format(address))
        client.close()
        camera.close()
    finally:
        finish = time.time()
    print('Sent %d images in %d seconds at %.2ffps' % (
        output.count, finish-start, output.count / (finish-start)))

server_socket = socket.socket()
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

while True:
    client, address = server_socket.accept()
    threading.Thread(target=listenToClient,args=(client,address)).start()

