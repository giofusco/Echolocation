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

    def detect_marker(self, frame, rescale_factor= 1/3, id=1001):
        pt_top = None
        pt_center = None
        frame_scaled = cv2.resize(frame, (int(frame.shape[1] * rescale_factor), int(frame.shape[0] * rescale_factor)))

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame_scaled, self.aruco_dict, parameters=self.aruco_params)


        res = (np.where(ids == id))
        frame_center = (int(frame_scaled.shape[1]/2), int(frame_scaled.shape[0]/2))

        if len(res[0]) > 0:
            idx = np.where(ids == id)[0][0]
            corners_orig = corners[idx] * 1 / rescale_factor
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners_orig, 0.1, cameraMatrix=self.mtx,
                                                                       distCoeffs=self.dist)
            pt, _ = cv2.projectPoints(np.float32([[0, 0, 0], [0.1,0,0]]), rvecs, tvecs, self.mtx, self.dist)
            pt_center = np.resize(pt[0], 2)
            pt_top = np.resize(pt[1], 2)

        return pt_center, pt_top