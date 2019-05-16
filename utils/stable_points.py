import cv2
import numpy as np


class StablePointsCollector:

    def __init__(self, max_distance_thr : float = 5.0, min_consecutive : float = 10.0):
        self.same_cnt = 0
        self.pt_x = []
        self.pt_y = []
        self.min_consecutive = min_consecutive
        self.max_dist_thr = max_distance_thr
        self.prevPt = (-1, -1)
        self.num_points = 0

    def append(self, pt : tuple):
        if pt[0] > 0 and pt[1] > 0:
            if self.prevPt[0] != -1:
                d = cv2.norm(pt - self.prevPt)
                if d < self.max_dist_thr:
                    self.same_cnt += 1
                    if self.same_cnt > self.min_consecutive:
                        self.pt_x.append(pt[0])
                        self.pt_y.append(pt[1])
                        self.same_cnt = 0
                        self.num_points += 1
                else:
                    self.same_cnt = 0
                    self.prevPt = pt
            else:
                self.prevPt = pt
