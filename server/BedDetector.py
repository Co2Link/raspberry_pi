import numpy as np
import cv2
import glob
import copy
from utils import createLineIterator, line_intersection

# RGB
Black = (0, 0, 0)
White = (255, 255, 255)
MediumGray = (128, 128, 128)
RED = (255, 0, 0)
Green = (0, 255, 0)
Blue = (0, 0, 255)
Yellow = (255, 255, 0)
Orange = (255, 165, 0)
NavyBlue = (0, 0, 128)


def BGR2RGB(src):
    return (src[2], src[1], src[0])


class Line:
    def __init__(self, point1, point2):
        # tan(pi - alpha) = k
        self.point1, self.point2 = point1, point2
        self.mid_point = (
            int((point1[0] + point2[0])/2), int((point1[1] + point2[1])/2))
        dy = point2[1] - point1[1]
        dx = point2[0] - point1[0]
        if dx != 0:
            self.k = dy/dx
            alpha = np.arctan(self.k) * (180/np.pi)
            self.alpha = -alpha if alpha < 0 else 180 - alpha
        else:
            self.k = np.inf
            self.alpha = 90
        self.group = ''

    def get_lower_point(self):
        return self.point1 if self.point1[1] > self.point2[1] else self.point2

    def set_group(self, group):
        self.group = group

    def set_info(self, info):
        self.info = info

    @classmethod
    def create_line_from_midPoint_and_alpha(self, mid_point, alpha, shape):
        k = np.tan(np.pi-alpha/180*np.pi)
        x1, y1 = mid_point
        dx = x1 if x1 > shape[1] - x1 - 1 else shape[1] - x1 - 1
        x2 = x1 + dx
        y2 = y1 + dx*k
        return Line(mid_point, (x2, y2))


class BedDetector:
    def __init__(self, threshold=100, minLineLength=100, maxLineGap=20, kernelSize=(5, 5), pixel_deviation=10, alpha_deviation=15):
        self.threshold = threshold
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap
        self.kernelSize = kernelSize
        self.pixel_deviation = pixel_deviation
        self.alpha_deviation = alpha_deviation

    def detect(self, img, debug=False):
        self.H, self.W, _ = img.shape
        self.shape = self.H, self.W

        intxn_points = {'p1': None, 'p2': None, 'p3': None, 'p4': None}
        success = True
        img_debug = None

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        high_thresh, thresh_im = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5*high_thresh

        edges_1 = cv2.Canny(img_gray, low_thresh, high_thresh)

        kernel = np.ones(self.kernelSize, np.uint8)

        dilate = cv2.dilate(edges_1, kernel)

        edges_2 = cv2.Canny(dilate, low_thresh, high_thresh)

        raw_lines = cv2.HoughLinesP(edges_2, 1, np.pi/180, threshold=self.threshold,
                                    minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)

        if raw_lines is not None:
            lines = self._classify_lines(raw_lines)
            lines_extended = self._extend_lines(lines)
            lines_average = self._average_lines(lines_extended)
            lines_adjusted = self._adjust_lines(edges_2, lines_average)

            # p1: uppper-left, p2: lower-left, p3: lower-right, p4: upper-right
            for idx_1 in range(len(lines_adjusted)):
                for idx_2 in range(idx_1+1, len(lines_adjusted)):
                    line_1, line_2 = lines_adjusted[idx_1], lines_adjusted[idx_2]
                    x, y = line_intersection([line_1.point1, line_1.point2], [
                                             line_2.point1, line_2.point2])
                    print(x, y)
                    if x >= 0 and x < self.W and y >= 0 and y < self.H:
                        print(x, y)
                        if line_1.group in ['UHL', 'LVL'] and line_2.group in ['UHL', 'LVL']:
                            intxn_points['p1'] = (x, y)
                        elif line_1.group in ['LHL', 'LVL'] and line_2.group in ['LHL', 'LVL']:
                            intxn_points['p2'] = (x, y)
                        elif line_1.group in ['LHL', 'RVL'] and line_2.group in ['LHL', 'RVL']:
                            intxn_points['p3'] = (x, y)
                        elif line_1.group in ['UHL', 'RVL'] and line_2.group in ['UHL', 'RVL']:
                            intxn_points['p4'] = (x, y)

            line_groups = self._group_lines(lines_adjusted)
            print(intxn_points)
            if all(intxn_points[i] != None for i in ['p1', 'p4']) and all(intxn_points[i] == None for i in ['p2', 'p3']):
                LVL, RVL = line_groups['LVL'][0], line_groups['RVL'][0]
                intxn_points['p2'] = LVL.get_lower_point()
                intxn_points['p3'] = RVL.get_lower_point()
            elif all(intxn_points[i] != None for i in ['p1', 'p2', 'p3', 'p4']):
                pass
            else:
                print('cant detect the upper left or the upper right point')
                success = False
                # return None
        # debug
        if debug:
            cv2.imshow('img', img)
            cv2.imshow('img_gray', img_gray)
            cv2.imshow('edges_1', edges_1)
            cv2.imshow('dilate', dilate)
            cv2.imshow('edges_2', edges_2)

            if raw_lines is not None:
                cv2.imshow('lines', self._draw_debug(np.copy(img), lines))

                cv2.imshow('lines_extended', self._draw_debug(
                    np.copy(img), lines_extended))

                cv2.imshow('lines_average', self._draw_debug(
                    np.copy(img), lines_average))

                img_debug = self._draw_debug(
                    np.copy(img), lines_adjusted, intxn_points)

                cv2.imshow('lines_adjuested', img_debug)

            if cv2.waitKey(5) == 27:
                exit(0)
            # cv2.waitKey(0)

        return intxn_points, success, img_debug

    def _adjust_lines(self, img, lines):
        lines_adjusted = []
        for line in lines:
            biggest_overlap = 0
            best_line = line
            if line.group in ['LVL', 'RVL']:
                # x deviation
                for xD in range(-self.pixel_deviation, self.pixel_deviation+1):
                    x_new = line.mid_point[0] + xD
                    if x_new < 0 or x_new >= self.W:
                        continue
                    # alpha deviation
                    for alphaD in range(-self.alpha_deviation, self.alpha_deviation+1):
                        # update line
                        line_sup = Line.create_line_from_midPoint_and_alpha(
                            (x_new, line.mid_point[1]), line.alpha+alphaD, self.shape)
                        line_sup.set_group(line.group)
                        line_sup = self._extend_lines([line_sup])[0]
                        overlap = createLineIterator(
                            np.array(line_sup.point1), np.array(line_sup.point2), img).sum()
                        if overlap > biggest_overlap:
                            biggest_overlap = overlap
                            best_line = line_sup

            elif line.group in ['UHL', 'LHL']:
                for yD in range(-self.pixel_deviation, self.pixel_deviation+1):
                    y_new = line.mid_point[1] + yD
                    if y_new < 0 or y_new >= self.H:
                        continue
                    for alphaD in range(-self.alpha_deviation, self.alpha_deviation+1):
                        line_sup = Line.create_line_from_midPoint_and_alpha(
                            (line.mid_point[0], y_new), line.alpha+alphaD, self.shape)
                        line_sup.set_group(line.group)
                        line_sup = self._extend_lines([line_sup])[0]
                        overlap = createLineIterator(
                            np.array(line_sup.point1), np.array(line_sup.point2), img).sum()
                        if overlap > biggest_overlap:
                            biggest_overlap = overlap
                            best_line = line_sup

            lines_adjusted.append(best_line)

        return lines_adjusted

    def _classify_lines(self, raw_lines):
        lines = []
        for raw_line in raw_lines:
            line = Line((raw_line[0][0], raw_line[0][1]),
                        (raw_line[0][2], raw_line[0][3]))
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

    def _extend_lines(self, lines):
        # Point oblique type: y-y1 = k(x-x1)
        lines_extended = []
        for line in lines:
            x1, y1 = line.mid_point
            k = line.k

            if k == np.inf:
                line_extended = Line(
                    (line.point1[0], 0), (line.point1[0], self.H-1))
                line_extended.set_group(line.group)
            elif k == 0:
                line_extended = Line(
                    (0, line.point1[1]), (self.W-1, line.point1[1]))
                line_extended.set_group(line.group)
            else:
                # intersection with x axis and y axis
                x0 = x1 - y1/k
                y0 = y1 - k * x1

                # intersection with x = W and y = H
                y_ = k*(self.W - x1) + y1
                x_ = (self.H - y1)/k + x1

                points = [(int(x0), 0), (0, int(y0)),
                          (self.W-1, int(y_)), (int(x_), self.H-1)]
                # the point we want is the point that in the middle
                sorted_poinsts = sorted(points, key=lambda a: a[0])

                line_extended = Line(
                    tuple(sorted_poinsts[1]), tuple(sorted_poinsts[2]))
                line_extended.set_group(line.group)

            lines_extended.append(line_extended)

        return lines_extended

    def _average_lines(self, lines):
        # lines = copy.deepcopy(lines)
        lines_average = []

        alpha_sum_H = 0
        # the num of both UHL and LHL
        length_H = 0

        mid_point_buf = {}
        alpha_buf = {}
        for group in ['LVL', 'RVL', 'UHL', 'LHL']:
            lines_sup = list(filter(lambda a: a.group == group, lines))

            if len(lines_sup) != 0:
                mid_points = map(lambda line: line.mid_point, lines_sup)
                mid_point_average = [int(sum(x)/len(x))
                                     for x in zip(*mid_points)]

                mid_point_buf[group] = tuple(mid_point_average)

                if group in ['LVL', 'RVL']:
                    alpha_average = sum(
                        [line.alpha for line in lines_sup])/len(lines_sup)
                    alpha_buf[group] = alpha_average
                else:
                    alpha_sum_H = sum([line.alpha for line in lines_sup])
                    length_H += len(lines_sup)

        for key, value in mid_point_buf.items():
            if key in ['LVL', 'RVL']:
                line = Line.create_line_from_midPoint_and_alpha(
                    mid_point_buf[key], alpha_buf[key], self.shape)
            else:
                line = Line.create_line_from_midPoint_and_alpha(
                    mid_point_buf[key], alpha_sum_H/length_H, self.shape)
            line.set_group(key)
            lines_average.append(line)

        lines_average = self._extend_lines(lines_average)

        return lines_average

    def _draw_lines(self, src, lines):
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

            cv2.line(src, line.point1, line.point2, BGR2RGB(color), 3)
        return src

    def _draw_debug(self, src, lines, points={}):
        src = self._draw_lines(src, lines)
        for line in lines:
            cv2.circle(src, line.mid_point, 3, (255, 0, 0), -1)
            cv2.putText(src, str(int(line.alpha)) + ' k: {:.2f}'.format(
                line.k), line.mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for key, value in points.items():
            if value is not None:
                cv2.circle(src, value, 5, (0, 0, 0), -1)
                cv2.putText(src, key, value,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return src

    def _group_lines(self, lines):
        groups = {}
        for line in lines:
            if line.group not in groups:
                groups[line.group] = [line]
            else:
                groups[line.group].append(line)
        return groups


if __name__ == "__main__":
    image_paths = glob.glob('../data/images/*.jpg')

    threshold = 100
    minLineLength = 100
    maxLineGap = 20
    kernelSize = (20, 20)

    detector = BedDetector(threshold=threshold, minLineLength=minLineLength,
                           maxLineGap=maxLineGap, kernelSize=kernelSize)

    for path in image_paths:
        img = cv2.imread(path)
        detector.detect(img, True)