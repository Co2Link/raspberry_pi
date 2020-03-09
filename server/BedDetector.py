import numpy as np
import cv2
import glob
import copy
from server.utils import createLineIterator

# RGB
Black = (0,0,0)
White = (255,255,255)
MediumGray = (128,128,128)
RED = (255,0,0)
Green = (0,255,0)
Blue = (0,0,255)
Yellow = (255,255,0)
Orange = (255,165,0)
NavyBlue = (0,0,128)

def BGR2RGB(src):
    return (src[2],src[1],src[0])

class Line:
    def __init__(self,x1=1,y1=1,x2=2,y2=2):
        # tan(pi - alpha) = k
        self.x1,self.y1,self.x2,self.y2 = x1,y1,x2,y2
        self.mid_point = (int((x1 + x2)/2), int((y1 + y2)/2))
        self.k = (y2 - y1)/(x2 - x1)
        alpha = np.arctan((y2 - y1)/(x2 - x1)) * (180/np.pi)
        self.alpha = -alpha if alpha < 0 else 180 - alpha
        self.group = ''

    def set_group(self,group):
        self.group = group

    def set_point(self,point_1,point_2):
        self.x1, self.y1 = point_1
        self.x2, self.y2 = point_2

    def set_MidPoint(self, mid_point):
        self.mid_point = mid_point

    def update_MidPoint(self):
        self.mid_point = (int((self.x1 + self.x2)/2), int((self.y1 + self.y2)/2))

    def set_alpha(self, alpha):
        if alpha == 90:
            raise Exception('unhandle of alpha == 90')
        self.alpha = alpha
        alpha_reverse = -self.alpha
        self.k = np.tan(alpha_reverse * np.pi / 180)
    
    def set_info(self, info):
        self.info = info


class BedDetector:
    def __init__(self,threshold = 100,minLineLength = 100,maxLineGap = 20,kernelSize = (5,5),pixel_deviation = 10,alpha_deviation = 15 ):
        self.threshold = threshold
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap
        self.kernelSize = kernelSize
        self.pixel_deviation = pixel_deviation
        self.alpha_deviation = alpha_deviation

    def detect(self,img):
        self.H, self.W, _ = img.shape
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        high_thresh, thresh_im = cv2.threshold(img_gray, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5*high_thresh

        edges_1 = cv2.Canny(img_gray, low_thresh, high_thresh)

        kernel = np.ones(self.kernelSize, np.uint8)

        dilate = cv2.dilate(edges_1,kernel)

        edges_2 = cv2.Canny(dilate, low_thresh, high_thresh)

        raw_lines = cv2.HoughLinesP(edges_2, 1, np.pi/180, threshold=self.threshold, minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)

        if raw_lines is not None:
            lines = self._classify_lines(raw_lines)
            lines_extended = self._extend_lines(lines)
            lines_average = self._average_lines(lines_extended)
            lines_adjusted = self._adjust_lines(edges_2, lines_average)
        
        # debug
        cv2.imshow('img', img)
        cv2.imshow('img_gray', img_gray)
        cv2.imshow('edges_1', edges_1)
        cv2.imshow('dilate', dilate)
        cv2.imshow('edges_2', edges_2)

        if raw_lines is not None:
            cv2.imshow('lines', self._draw_debug(np.copy(img), lines))

            cv2.imshow('lines_extended', self._draw_debug(np.copy(img), lines_extended))

            cv2.imshow('lines_average', self._draw_debug(np.copy(img), lines_average))

            cv2.imshow('lines_adjuested', self._draw_debug(np.copy(img), lines_adjusted))
            
        raw_lines_2 = cv2.HoughLinesP(edges_1, 1, np.pi/180, threshold=self.threshold, minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)
        if raw_lines_2 is not None:
            lines_2 = self._classify_lines(raw_lines_2)
            cv2.imshow('lines_2', self._draw_debug(np.copy(img), lines_2))

        # if cv2.waitKey(5) == 27:
        #     exit(0)
        cv2.waitKey(0)
    def _adjust_lines(self,img,lines):
        lines = copy.deepcopy(lines)
        lines_adjusted = []
        for line in lines:
            biggest_overlap = 0
            best_line = line
            if line.group in ['LVL','RVL']:
                # x deviation
                for xD in range(-self.pixel_deviation,self.pixel_deviation+1):
                    x_new = line.mid_point[0] + xD
                    if x_new < 0 or x_new >= self.W:
                        continue
                    # alpha deviation
                    for alphaD in range(-self.alpha_deviation,self.alpha_deviation+1):
                        # update line
                        line_sup = copy.deepcopy(line)
                        line_sup.set_alpha(line_sup.alpha + alphaD)
                        line_sup.set_MidPoint((x_new, line_sup.mid_point[1]))
                        line_sup = self._extend_lines([line_sup])[0]
                        overlap = createLineIterator(np.array([line_sup.x1,line_sup.y1]),np.array([line_sup.x2,line_sup.y2]),img).sum()
                        if overlap > biggest_overlap:
                            biggest_overlap = overlap
                            best_line = line_sup
                        
            elif line.group in ['UHL', 'LHL']:
                for yD in range(-self.pixel_deviation,self.pixel_deviation+1):
                    y_new = line.mid_point[1] + yD
                    if y_new < 0 or y_new >= self.H:
                        continue
                    for alphaD in range(-self.alpha_deviation,self.alpha_deviation+1):
                        line_sup = copy.deepcopy(line)
                        line_sup.set_alpha(line_sup.alpha + alphaD)
                        line_sup.set_MidPoint((line_sup.mid_point[0], y_new))
                        line_sup = self._extend_lines([line_sup])[0]
                        overlap = createLineIterator(np.array([line_sup.x1,line_sup.y1]),np.array([line_sup.x2,line_sup.y2]),img).sum()
                        if overlap > biggest_overlap:
                            biggest_overlap = overlap
                            best_line = line_sup

            lines_adjusted.append(best_line)

        return lines_adjusted
            
    def _classify_lines(self,raw_lines):
        lines = []
        for raw_line in raw_lines:
            line = Line(*raw_line[0])
            # Vertical
            if 45 < line.alpha and line.alpha < 135:
                # Left
                if line.mid_point[0] < self.W/2:
                    line.set_group('LVL')
                # Right
                else:
                    line.set_group('RVL')
            # Horizontal
            else:
                # Upper
                if line.mid_point[1] < self.H/2:
                    line.set_group('UHL')
                # Lower
                else:
                    line.set_group('LHL')

            lines.append(line)
        return lines

    def _extend_lines(self,lines):
        # Point oblique type: y-y1 = k(x-x1)
        lines = copy.deepcopy(lines)
        lines_extended = []
        for line in lines:
            x1, y1 = line.mid_point
            k = line.k

            # intersection with x axis and y axis
            x0 = x1 - y1/k
            y0 = y1 - k * x1

            # intersection with x = W and y = H
            y_ = k*(self.W - x1) + y1
            x_ = (self.H - y1)/k + x1

            # there are 4 points: (x0,0),(0,y0),(W,y_),(x_,H)
            points = [(x0,0), (0,y0), (self.W,y_), (x_,self.H)]
            # the point we want is the point that in the middle
            sorted_poinsts = sorted(points, key=lambda a: a[0])

            line.set_point(tuple(map(int, sorted_poinsts[1])), tuple(map(int, sorted_poinsts[2])))
            line.update_MidPoint()

            lines_extended.append(line)
        
        return lines_extended

    def _average_lines(self,lines):
        lines = copy.deepcopy(lines)
        lines_average = {}
        
        alpha_sum_H = 0
        # the num of both UHL and LHL
        length_H = 0
        for group in ['LVL', 'RVL', 'UHL', 'LHL']:
            lines_sup = list(filter(lambda a: a.group == group,lines))

            if len(lines_sup) != 0:
                mid_points = map(lambda line: line.mid_point, lines_sup)
                mid_point_average = [int(sum(x)/len(x)) for x in zip(*mid_points)]

                line_average = Line()
                line_average.set_group(group)
                line_average.set_MidPoint(tuple(mid_point_average))

                if group in ['LVL', 'RVL']:
                    alpha_average = sum([line.alpha for line in lines_sup])/len(lines_sup)
                    line_average.set_alpha(alpha_average)
                else:
                    alpha_sum_H = sum([line.alpha for line in lines_sup])
                    length_H += len(lines_sup)
                
                lines_average[group] = line_average

        if 'UHL' in lines_average:
            lines_average['UHL'].set_alpha(alpha_sum_H/length_H)
        elif 'LHL' in lines_average:
            lines_average['LHL'].set_alpha(alpha_sum_H/length_H)
        
        lines_average = [value for key, value in lines_average.items()]
        lines_average = self._extend_lines(lines_average)

        return lines_average
        
    def _draw_lines(self,src,lines):
        # decide color by line's group
        for line in lines:
            if line.group == 'LVL':
                color = Orange
            elif line.group == 'RVL':
                color = Yellow
            elif line.group == 'UHL':
                color = NavyBlue
            elif line.group == 'LHL':
                color = MediumGray
            else:
                color = White
                
            cv2.line(src, (line.x1,line.y1), (line.x2,line.y2), BGR2RGB(color), 3)
        return src

    def _draw_debug(self,src,lines):
        src = self._draw_lines(src,lines)
        for line in lines:
            cv2.circle(src, line.mid_point, 3, (255,0,0), -1)
            cv2.putText(src, str(int(line.alpha)) + ' k: {:.2f}'.format(line.k), line.mid_point,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return src

if __name__ == "__main__":
    image_paths = glob.glob('../data/images/*.jpg')

    threshold = 100
    minLineLength = 100
    maxLineGap = 20
    kernelSize = (20,20)

    detector = BedDetector(threshold = threshold, minLineLength=minLineLength, maxLineGap=maxLineGap, kernelSize = kernelSize)

    for path in image_paths:
        img = cv2.imread(path)
        detector.detect(img)