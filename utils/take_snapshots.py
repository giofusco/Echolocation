import cv2
import numpy as np
from cv_tools import cv_tools as cvt

def take_snapshots(video : str):

    cap = cv2.VideoCapture(video)

    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cnt = 0

    while True:

        ret, frame = cap.read()
        cnt += 1
        if frame is not None:
            undist = cvt.undo_distortion(frame)
            cv2.imshow("Frame", undist)
            k = cv2.waitKey(-1)
            if k == ord('s'):
                cv2.imwrite('frame_' + str(cnt) + ".jpg", undist)
            elif k == ord('q'):
                break

take_snapshots("./videos/scale.MP4")