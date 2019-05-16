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

    def detect_marker(self, frame):

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        print(ids)
        if not corners == []:
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.1, cameraMatrix=self.mtx,
                                                                       distCoeffs=self.dist)
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, self.mtx, self.dist, rvecs, tvecs, 2)
        # rotM = np.zeros(shape=(3, 3))
        # cv2.Rodrigues(rvecs, rotM, jacobian=0)
        # retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotM)