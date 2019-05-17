from cv_tools import cv_tools as cvt
import numpy as np
import cv2
class MarkerDetector:

    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.adaptiveThreshWinSizeMax = 11
        self.aruco_params.cornerRefinementWinSize = 3
        self.aruco_params.minMarkerPerimeterRate = 0.005
        self.aruco_params.maxErroneousBitsInBorderRate = 0.
        self.aruco_params.minDistanceToBorder = 10

        self.mtx = cvt.gopro_intrinsic()
        self.dist = cvt.gopro_distortion_coeffs()
        #
        # aruco_params.cornerRefinementMinAccuracy = 0.1
        # aruco_params.polygonalApproxAccuracyRate = 0.1

    def detect_marker(self, frame, id=1001):
        pt = []
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        idx = -1
        res = (np.where(ids == id))
        print(len(res[0]))
        if len(res[0]) > 0:
            idx = np.where(ids == id)[0][0]

        if not corners == [] and idx>=0:
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners[idx], 0.1, cameraMatrix=self.mtx,
                                                                       distCoeffs=self.dist)
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, self.mtx, self.dist, rvecs[idx], tvecs[idx], .1)
            R = cv2.Rodrigues(rvecs[idx])
            # print (R)
            R = np.asmatrix(R[0])
            # cv2.aruco.drawAxis(frame, mtx, dist, rvecs[0], tvecs[0], 1)
            pt = R.dot(np.transpose([0, 1, 0])) + tvecs[idx]
            pt = np.resize(pt, 3)
            print("*", pt)
        return pt
        # rotM = np.zeros(shape=(3, 3))
        # cv2.Rodrigues(rvecs, rotM, jacobian=0)
        # retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotM)