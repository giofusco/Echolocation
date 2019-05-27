import cv2
import numpy as np
import math
from scipy.interpolate import splprep, splev

def find_largest_contour(contours):
    largest_c = -1
    maxarea = 0
    for c in range(0, len(contours)):
        area = cv2.contourArea(contours[c])
        if area > maxarea:
            largest_c = c
            maxarea = area
    return largest_c


def get_box_midpoint(box : np.array, frame : np.matrix) -> [tuple, np.matrix]:
    box = np.int0(box)

    p0 = box[0]
    p1 = box[1]
    p2 = box[2]
    p3 = box[3]

    frame_center = (int(frame.shape[1]/2), int(frame.shape[0]/2))

    d1 = cv2.norm(p1 - p0)
    d2 = cv2.norm(p0 - p3)
    midPoint = []
    dc1 = 0
    dc2 = 0
    if (d2 < d1):
        midPoint1 = (p0 + p3) * .5
        midPoint2 = (p1 + p2) * .5
        dc1 = cv2.norm(midPoint1-frame_center)
        dc2 = cv2.norm(midPoint2 - frame_center)
    else:
        midPoint1 = (p0 + p1) * .5
        midPoint2 = (p3 + p2) * .5
        dc1 = cv2.norm(midPoint1 - frame_center)
        dc2 = cv2.norm(midPoint2 - frame_center)

    cv2.circle(frame, (int(box[0][0]), int(box[0][1])), 2, (255, 0, 0))
    cv2.circle(frame, (int(box[1][0]), int(box[1][1])), 2, (0, 255, 0))
    cv2.circle(frame, (int(box[2][0]), int(box[2][1])), 2, (0, 255, 255))
    cv2.circle(frame, (int(box[3][0]), int(box[3][1])), 2, (0, 0, 255))

    if dc1 < dc2:
        midPoint = midPoint1
    else:
        midPoint = midPoint2

    return midPoint, frame


def smooth_contour(contour: np.array):
    smoothened = []
    x, y = contour.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    tck, u = splprep([x, y], u=None, s=1.0, per=1)
    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
    u_new = np.linspace(u.min(), u.max(), 25)
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    x_new, y_new = splev(u_new, tck, der=0)
    # Convert it back to numpy format for opencv to be able to display it
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
    smoothened.append(np.asarray(res_array, dtype=np.int32))

    return smoothened


def segment_rod(frame : np.matrix, th1 : int = 250, th2 : int = 295, ts1 : int = 120, ts2 : int = 200) -> np.array:
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
    mask = cv2.inRange(hls[:, :, 0], th1, th2) & cv2.inRange(hls[:, :, 2], ts1, ts2)
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("Mask", mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_c = find_largest_contour(contours)
    if largest_c >= 0:
        return contours[largest_c]
    else:
        return None


def undo_distortion(frame : np.matrix) -> np.matrix:
    mtx = gopro_intrinsic()
    dist = gopro_distortion_coeffs()
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(frame, mtx, dist, None, None)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


def get_rod_reference_point(frame : np.matrix):
    pt = None
    contour = segment_rod(frame)
    centroid = cv2.mean(contour)
    if contour is not None:
        rrect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rrect)
        bbox = np.int0(box)
        pt, vis = get_box_midpoint(box, frame)
        # cv2.putText(vis, str(rrect[2]), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,0,0))
    return pt, centroid, vis

def is_led_on(frame, prev_brightness):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
    led_quad = gray[280:320, 500:560]

    led_triggered = False
    if prev_brightness < 0:
        prev_brightness = cv2.mean(led_quad)[0]
        led_delta = 0
        return led_triggered, prev_brightness
    else:
        curr_brightness = cv2.mean(led_quad)[0]
        led_delta = curr_brightness - prev_brightness
        if led_delta > 30:
            cv2.rectangle(frame, (500, 280), (560, 320), (0, 0, 255), 2)
            led_triggered = True
        else:
            led_triggered = False
    return led_triggered, curr_brightness

def gopro_intrinsic() -> np.matrix:
    # return np.matrix([[1.06158880e+03, 0.00000000e+00, 9.63653719e+02],
    #              [0.00000000e+00, 1.05869356e+03, 5.48963329e+02],
    #              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    return np.array([[1034.74047817257,	0.0,               959.273355077146 ],
                 [0,                1030.43959904139,  541.277229460405],
                 [0,                0,                 1]])


def gopro_distortion_coeffs() -> np.array:
    # return np.array([[ 1.90622012e+01, 6.47648093e+01, 5.09826047e-03,  1.17481012e-03,
    #               -8.72367013e+01, 1.97066705e+01, 6.45949420e+01, -8.80120382e+01,
    #                0.00000000e+00, 0.00000000e+00, 0.00000000e+00,  0.00000000e+00,
    #                0.00000000e+00, 0.00000000e+00]])

    return np.array([[-0.0180376858725902, 0.0473241322917970, 0, 0, 0]])