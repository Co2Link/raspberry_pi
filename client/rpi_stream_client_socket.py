
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming

import io
import picamera
import socket
from threading import Condition

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

client_socket = socket.socket()
client_socket.connect(('172.20.10.2',8000))
connection = client_socket.makefile('wb')

with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
    output = StreamingOutput()

    camera.start_recording(output, format='mjpeg')
    try:
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame
            connection.write(frame)
            connection.flush()
            print(frame[:10])
            print(frame[-10:])

    finally:
        camera.stop_recording()
