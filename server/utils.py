import time
from collections import deque


class Frame_rate_calculator:
    def __init__(self):
        self.start = 0
        self.time_list = deque(maxlen=5)
        self.current_frame_rate = 0
        self.output_frame_rate = 0
        self.counter = 0
        self.frame_rate_update_time = 0
    def start_record(self):
        self.start = time.time()
    def frame_end(self):
        self.time_list.append(time.time()-self.start)
        self.start = time.time()
        self.current_frame_rate = round(1/(sum(self.time_list)/len(self.time_list)),2)
        return self.current_frame_rate
    def get_frame_rate(self):
        if time.time() - self.frame_rate_update_time > 1:
            self.output_frame_rate = self.current_frame_rate
            self.frame_rate_update_time = time.time()
        return self.output_frame_rate