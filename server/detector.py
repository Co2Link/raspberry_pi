import time
import math
from collections import deque
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def cal_distance(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def cal_dist_by_rawdata(index_1,index_2,raw_data):
    return cal_distance(raw_data[0,index_1,0],raw_data[0,index_2,0],raw_data[0,index_1,1],raw_data[0,index_2,1])

class Detector:
    def __init__(self,bed_location):
        self.keypoints = deque(maxlen=20)
        self.status = deque(maxlen=20)
        self.start = time.time()
        self.last_time = self.start
        self.bed_location = bed_location
        self.on_the_bed = True
    def get_raw_data(self,keypoint):
        self.keypoints.append(keypoint)
        self.status.append(self._raw_data_to_status(keypoint))
    
    def _raw_data_to_status(self,raw_data):
        try:
            index_list = []
            for index,i in enumerate(raw_data[0,:,0]):
                if i != 0:
                    index_list.append(index)
            self._person_on_bed(raw_data, index_list)
            
            if any(x in index_list for x in [0,15,16,17,18]) or len(index_list) > 10 :
                # if time.time()-self.last_time > 1:
                #     print('time: {}'.format(time.time()-self.start))
                #     self.last_time = time.time()

                #     if 0 in index_list and 1 in index_list:
                #         print('dist_01 = {}'.format(cal_dist_by_rawdata(0,1,raw_data)))
                #     if 1 in index_list and 2 in index_list:
                #         print('dist_12 = {}'.format(cal_dist_by_rawdata(2,1,raw_data)))
                #     if 5 in index_list and 1 in index_list:
                #         print('dist_15 = {}'.format(cal_dist_by_rawdata(5,1,raw_data))) 

                #     if 17 in index_list and 1 in index_list:
                #         print('dist_17_1 = {}'.format(cal_dist_by_rawdata(17,1,raw_data)))
                #     if 16 in index_list and 1 in index_list:
                #         print('dist_16_1 = {}'.format(cal_dist_by_rawdata(16,1,raw_data)))
                #     if 15 in index_list and 1 in index_list:
                #         print('dist_15_1 = {}'.format(cal_dist_by_rawdata(15,1,raw_data)))
                #     if 18 in index_list and 1 in index_list:
                #         print('dist_18_1 = {}'.format(cal_dist_by_rawdata(18,1,raw_data)))
                    
                #     print()
                if any(x in index_list for x in [1,2,5]):
                    if all(x in index_list for x in [0,1,2]) and (cal_dist_by_rawdata(0,1,raw_data) < 1.05*cal_dist_by_rawdata(1,2,raw_data) or cal_dist_by_rawdata(0,1,raw_data) > 1.6*cal_dist_by_rawdata(1,2,raw_data)):
                        return 2
                    if all(x in index_list for x in [0,1,5]) and (cal_dist_by_rawdata(0,1,raw_data) < 1.05*cal_dist_by_rawdata(1,5,raw_data) or cal_dist_by_rawdata(0,1,raw_data) > 1.6*cal_dist_by_rawdata(1,5,raw_data)):
                        return 2
                    if 1 in index_list and (2 in index_list or 5 in index_list) and any(x in index_list for x in [17,18]):
                        if 2 in index_list:
                            com = cal_dist_by_rawdata(1,2,raw_data)
                        elif 5 in index_list:
                            com = cal_dist_by_rawdata(1,5,raw_data)
                        dist_list = []
                        for index in [15,16,17,18]:
                            if index in index_list:
                                dist_list.append(cal_dist_by_rawdata(index,1,raw_data))
                        # print('比例： ',sum(dist_list)/len(dist_list)/com)
                        if sum(dist_list)/len(dist_list) < 1.1*com or sum(dist_list)/len(dist_list) > 1.55*com:
                            return 2

                return 1
            else:
                return 0
            
        except:
            return 0

    def _get_status(self):
        if sum([status == 2 for status in self.status]) >= 14:
            # human awake
            return 2
        elif sum([(status == 2 or status ==1) for status in self.status]) >= 14:
            # human asleep
            return 1
        else:
            return 0

    def _person_on_bed(self, raw_data, index_list):
        points = [raw_data[0,idx,:] for idx in index_list]
        mid_point = [int(sum(x)/len(x)) for x in zip(*points)]
        
        mid_point = Point(mid_point[0], mid_point[1])
        bed_polygon = Polygon([self.bed_location['p1'], self.bed_location['p2'],self.bed_location['p3'],self.bed_location['p4']])

        ret = bed_polygon.contains(mid_point)
        self.on_the_bed = ret
        return ret
    
    def get_description(self):
        if self._get_status() == 1:
            if self.on_the_bed:
                return 'Human detected: asleep'
            else:
                return 'Human detected: fell on the ground'
        elif self._get_status() == 2:
                return 'Human detected: awake'
        elif self._get_status() == 0:
            return 'No human'
        else:
            return 'empty'