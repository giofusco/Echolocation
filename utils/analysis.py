import cv2
from cv_tools import cv_tools as cvt
import numpy as np
from utils import find_center
from utils.find_center import find_center_of_rotation, fine_tune_center
from utils import find_target_position

from math import acos


def find_center(video_src, center_file):
    center_ratio = [0, 0]
    # if we have stored a center, use that
    try:
        fh = open(center_file, 'r')
        data = fh.readlines()
        print(data[0])
        center_ratio[0] = float(data[0])
        center_ratio[1] = float(data[1])
        center_ratio = fine_tune_center(video_src, center_ratio)
    except FileNotFoundError:
        fh = open(center_file, 'w')
        center_ratio = find_center_of_rotation(video_src, 30, 0)
        center_ratio = fine_tune_center(video_src, center_ratio)
        fh = open(center_file, 'w')
        fh.write(str(center_ratio[0]) + "\n")
        fh.write(str(center_ratio[1]) + "\n")
        fh.close()
    return center_ratio

def process_frame(frame, center_ratio, marker_det, scale):
    alpha = None
    gamma = None
    pt_head = None
    pt_nose = None

    undist_orig = cvt.undo_distortion(frame)
    undist = cv2.resize(undist_orig, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))

    center = (int(center_ratio[0]*undist.shape[1]), int((center_ratio[1]*undist.shape[0])))

    # pt_target is in the scaled frame coordinates, while marker coordinates are in full frame
    pt_target, theta, pt_rod, rod_centroid = find_target_position.get_target_position(undist, center, scale/3, 1.0)
    pt_head, pt_nose = marker_det.detect_marker(frame)

    cv2.circle(undist, (int(rod_centroid[0]), int(rod_centroid[1])), 2, (0, 255, 255), 4)
    cv2.putText(undist, str(theta), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

    if pt_nose is not None:
        pt_head = np.asarray([pt_head[0] / 3, pt_head[1] / 3])
        pt_nose = np.asarray([pt_nose[0] / 3, pt_nose[1] / 3])

        cv2.line(undist, (int(pt_head[0]), int(pt_head[1])), (int(pt_target[0]), int(pt_target[1])),
                 (0, 200, 50), 2, cv2.LINE_AA)
        cv2.arrowedLine(undist,(int(pt_head[0]), int(pt_head[1])), (int(pt_nose[0]), int(pt_nose[1])), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(undist, (int(center[0]), int(center[1])),
                        (int(pt_target[0]), int(pt_target[1])), (255, 255, 255), 2, cv2.LINE_AA)

        head_pose_vec = pt_nose - pt_head
        gt_vec = pt_head - pt_target
        dot = -gt_vec[0] / cv2.norm(gt_vec)
        # alpha = acos(dot) * 180 / 3.14
        alpha = cv2.fastAtan2(gt_vec[1],-gt_vec[0])
        dot = head_pose_vec[0] / cv2.norm(head_pose_vec)
        # gamma = acos(dot) * 180 / 3.14
        gamma = cv2.fastAtan2(-head_pose_vec[1], head_pose_vec[0])
        cv2.putText(undist, str(alpha), (int(pt_head[0]), int(pt_head[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 50))
        cv2.putText(undist, str(gamma), (int(pt_nose[0]), int(pt_nose[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255))

    cv2.circle(undist, center, 5, (0,255,0), 6)

    return undist, alpha, gamma, theta, pt_target, pt_head, pt_nose
