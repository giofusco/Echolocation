import cv2
import numpy as np
from cv_tools import cv_tools as cvt
from utils.stable_points import StablePointsCollector
from utils.progress_bar import print_progress_bar
from utils import fit_ellipse as fe
import matplotlib.pyplot as plt
from utils import find_target_position


def find_center_of_rotation(video: str, max_dist: float = 5, min_consecutive: int = 2) -> tuple:

    cap = cv2.VideoCapture(video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cnt = 0

    pointCollector = StablePointsCollector(max_dist, min_consecutive)

    while True:

        ret, frame = cap.read()

        if frame is not None:
            cnt += 1
            if cnt % 30 == 0:
                undist = cvt.undo_distortion(frame)

                undist = cv2.resize(undist, (int(frame.shape[1]/3), int(frame.shape[0]/3)))
                pt, vis, centroid = cvt.get_rod_reference_point(undist)

                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0,250,255), 3)
                # cv2.imshow("Pts", vis)
                # cv2.waitKey(1)
                pt[1] = undist.shape[0] - pt[1]
                pointCollector.append(pt)

                scat = plt.scatter(pointCollector.pt_x, pointCollector.pt_y, c="r")
                plt.xlim([0, undist.shape[1]])
                plt.ylim([0, undist.shape[0]])
                plt.draw()
                plt.pause(0.001)

            # if cnt%120 == 0:
                print_progress_bar(cnt + 1, num_frames, prefix='Progress:', suffix='Complete, numPoints: ' + str(pointCollector.num_points), length=50)
                cv2.imshow("Frame", undist)
            # k = cv2.waitKey(1)
            # plt.close()
            # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

        if cnt >= num_frames:
            break

    a = fe.fitEllipse(np.asarray(pointCollector.pt_x), np.asarray(pointCollector.pt_y))
    center = fe.ellipse_center(a)

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax.set_xlim(0, undist.shape[1])
    ax.set_ylim(0, undist.shape[0])
    scat = plt.scatter(pointCollector.pt_x, pointCollector.pt_y, c = "r")
    plt.plot(center[0], center[1], 'o')
    plt.show()

    plt.show()
    return ( center[0]/undist.shape[1], (undist.shape[0] - center[1])/undist.shape[0] )


def grab_next_frame(cap : cv2.VideoCapture) -> np.matrix:
    frame = None
    cnt = 0
    while frame is None or cnt < 240:
        ret, frame = cap.read()
        cnt += 1
        if cnt == 1000 and frame is None:
            break
    return frame


def fine_tune_center(video: str, center: tuple) -> tuple:
    cap = cv2.VideoCapture(video)

    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cnt = 0

    frame = grab_next_frame(cap)
    frame = cvt.undo_distortion(frame)
    frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))

    vis = np.copy(frame)

    new_center = center

    c_offset = [0,0]

    while True:

        if frame is None:
            cap = cv2.VideoCapture(video)

            num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            cnt = 0

            frame = grab_next_frame(cap)
            frame = cvt.undo_distortion(frame)
            frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))

            vis = np.copy(frame)

        c = (int(center[0] * frame.shape[1]), int((center[1] * frame.shape[0])))
        c = (c[0]+c_offset[0],c[1]+c_offset[1])

        pt, angle, rod_pt, rod_centroid = find_target_position.get_target_position(frame, c, 1, 0)
        # print(c)
        cv2.line(vis, ( int(c[0]), int(c[1])), (int(rod_centroid[0]), int(rod_centroid[1])), (255,0,0,), 2)
        cv2.circle(vis, ( int(c[0]), int(c[1])), 3, (255,255,0), 5)
        cv2.circle(vis, (int(rod_centroid[0]), int(rod_centroid[1])), 3, (255, 255, 0), 5)
        cv2.putText(vis, str(round(angle,2)), c, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        cv2.imshow("Fine tune center of rotation", vis)
        k = cv2.waitKey(-1)

        if k == ord('q'):
            break
        elif k == ord('c'):
            frame = grab_next_frame(cap)
            if frame is not None:
                frame = cvt.undo_distortion(frame)
                frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
                vis = np.copy(frame)
        elif k == ord('i'):
            c_offset[1] -= 1
            vis = np.copy(frame)
        elif k == ord('k'):
            c_offset[1] += 1
            vis = np.copy(frame)
        elif k == ord('j'):
            c_offset[0] -= 1
            vis = np.copy(frame)
        elif k == ord('l'):
            c_offset[0] += 1
            vis = np.copy(frame)
        elif k == ord('r'):
            cap = cv2.VideoCapture(video)
            frame = grab_next_frame(cap)
            if frame is not None:
                frame = cvt.undo_distortion(frame)
                frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
                vis = np.copy(frame)

        # print(( c[0]/frame.shape[1], c[1]/frame.shape[0] ))

    return (c[0]/frame.shape[1], c[1]/frame.shape[0])




