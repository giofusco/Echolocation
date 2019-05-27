import cv2
import numpy as np
from cv_tools import cv_tools as cvt
from utils.stable_points import StablePointsCollector
from utils.progress_bar import print_progress_bar
from utils import fit_ellipse as fe
import matplotlib.pyplot as plt
from utils import find_target_position
import math


def find_center_of_rotation(video: str, max_dist: float = 5, min_consecutive: int = 2) -> tuple:

    cap = cv2.VideoCapture(video)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cnt = 0

    pointCollector = StablePointsCollector(max_dist, min_consecutive)

    while True:
        frame, cnt = grab_next_frame_with_led_on(cap, cnt)

        if frame is not None:

            undist = cvt.undo_distortion(frame)
            undist = cv2.resize(undist, (int(frame.shape[1]/3), int(frame.shape[0]/3)))
            pt, vis, centroid = cvt.get_rod_reference_point(undist)

            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0,250,255), 3)

            pt[1] = undist.shape[0] - pt[1]
            pointCollector.append(pt)

            scat = plt.scatter(pointCollector.pt_x, pointCollector.pt_y, c="r")
            plt.xlim([0, undist.shape[1]])
            plt.ylim([0, undist.shape[0]])
            plt.draw()
            plt.pause(0.001)

            print_progress_bar(cnt + 1, num_frames, prefix='Progress:', suffix='Complete, numPoints: ' + str(pointCollector.num_points), length=50)
            cv2.imshow("Frame", undist)

        if cnt >= num_frames:
            break

    print_progress_bar(num_frames, num_frames, prefix='Progress:',
                       suffix='Complete, numPoints: ' + str(pointCollector.num_points), length=50)
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


def grab_next_frame_with_led_on(cap : cv2.VideoCapture, cnt) -> np.matrix:
    prev_frame = None
    prev_led_on = False
    frame = None
    done = False
    brigthness = -1
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    while not done and cnt < num_frames:
        ret, frame = cap.read()
        cnt += 1
        if frame is not None:
            led_on, brigthness = cvt.is_led_on(frame, brigthness)
            if prev_led_on is True and led_on is not True:
                done = True
                return prev_frame, cnt
            prev_led_on = led_on
            prev_frame = frame
    return frame, cnt


def grab_next_frame(cap : cv2.VideoCapture) -> np.matrix:
    frame = None
    cnt = 0
    while frame is None or cnt < 600:
        ret, frame = cap.read()
        cnt += 1
        if cnt == 1000 and frame is None:
            break
    return frame


def fine_tune_center(video: str, center: tuple) -> tuple:
    cap = cv2.VideoCapture(video)

    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cnt = 0

    # frame = grab_next_frame(cap)
    frame, cnt = grab_next_frame_with_led_on(cap, cnt)
    frame = cvt.undo_distortion(frame)
    frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))

    vis = np.copy(frame)

    new_center = center
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    c_offset = [0,0]
    marked_points = []
    while True:
        vis = np.copy(frame)

        for p in range(0, len(marked_points)):
            pts = marked_points[p]
            p1 =  pts[0]
            p2 = pts[1]
            print(p1, p2)
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 255,), 2)

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
            frame, cnt = grab_next_frame_with_led_on(cap, cnt)
            if frame is not None:
                frame = cvt.undo_distortion(frame)
                frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
                vis = np.copy(frame)
            else:
                cap = cv2.VideoCapture(video)
                frame, cnt = grab_next_frame_with_led_on(cap, cnt)
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
            frame, cnt = grab_next_frame_with_led_on(cap, cnt)
            if frame is not None:
                frame = cvt.undo_distortion(frame)
                frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
                vis = np.copy(frame)
        elif k == ord('m'):
            pt1, pt2, angle = get_vanishing_points(rod_centroid, c, 2025, 2)
            marked_points.append([pt1, pt2])
            print(marked_points)
        elif k == ord('u'):
            if len(marked_points) > 0:
                marked_points.pop(-1)

    return (c[0]/frame.shape[1], c[1]/frame.shape[0])


def get_vanishing_points(pt: tuple, center: tuple, scale: float=2025, range: float=1) -> [tuple, float, tuple]:
    v = np.asarray(center) - np.asarray(pt)
    dot = np.dot(v, np.array([-1.0, 0.0])) / cv2.norm(v)
    angle = math.acos(dot) * 180 / 3.14
    if pt[1] > center[1]: # if rod is in 3rd or 4th quadrant, undo 180 wrapping
        angle = 360 - angle

    n = np.array([ np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))]) #change of sign for sign because of image ref. system
    d = scale * range

    pt1 = center + n * -d
    pt2 = center + n * d
    #cv2.circle(frame, (int(target_pt[0]), int(target_pt[1])), 2, (0,255,255), 4)
    return pt1, pt2, angle

