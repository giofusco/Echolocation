import cv2
from cv_tools import cv_tools as cvt
import numpy as np
from utils import find_center

import math

def get_target_position(frame: np.matrix, center: tuple, scale: float=2025, range: float=1) -> [tuple, float, tuple]:
    pt, centroid, _ = cvt.get_rod_reference_point(frame)
    centroid = (centroid[0], centroid[1])
    v = np.asarray(center) - np.asarray(centroid)
    dot = np.dot(v, np.array([-1.0, 0.0])) / cv2.norm(v)
    angle = math.acos(dot) * 180 / 3.14
    if centroid[1] > center[1]: # if rod is in 3rd or 4th quadrant, undo 180 wrapping
        angle = 360 - angle
    n = np.array([ np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))]) #change of sign for sign because of image ref. system
    d = scale * range
    # print(center, n, n*d)
    target_pt = center + n * d
    return target_pt, angle, pt, centroid

