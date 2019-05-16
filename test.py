import cv2
from cv_tools import cv_tools as cvt
import numpy as np
from utils import find_center
from utils import find_target_position
from cv_tools import marker_detector as md

video = './videos/GOPR3954.MP4'

center_ratio = find_center.find_center_of_rotation(video, 30, 0)
# print(center_ratio)
# center_ratio = (0.5590342613731353, 0.5702927276884127)


center_ratio = find_center.fine_tune_center(video, center_ratio)
marker_det = md.MarkerDetector()
cap = cv2.VideoCapture(video)
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

cnt = 0
scale = 2025 # pixels x meter @ 1635 x 1053

while True:

    ret, frame = cap.read()
    cnt += 1
    if frame is not None:
        if cnt % 10 == 0:
            undist = cvt.undo_distortion(frame)
            orig_size = undist.shape
            undist = cv2.resize(undist, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
            marker_det.detect_marker(undist)
            center = (int(center_ratio[0]*undist.shape[1]), int((center_ratio[1]*undist.shape[0])))
            pt, angle, rod_pt, centroid = find_target_position.get_target_position(undist, center, scale/3, 0.2)
            print(pt)
            cv2.putText(undist, str(round(angle,2)), center, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0))
            cv2.circle(undist, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), 10)
            # cv2.line(undist, center, (int(pt[0]), int(pt[1])), (0,255,255), 2)
            cv2.circle(undist, center, 5, (0,255,0), 6)
            cv2.imshow("Frame", undist)
            cv2.waitKey(1)
    if cnt == num_frames:
        break
