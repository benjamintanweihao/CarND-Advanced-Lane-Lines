import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


######################
# Camera Calibration #
######################

def calibrate_camera(draw_chessboard=False):
    nx, ny = 9, 6

    calibration_images = []
    for fn in glob.glob('camera_cal/*.jpg'):
        calibration_images.append(fn)

    # https://docs.opencv.org/3.3.1/dc/dbb/tutorial_py_calibration.html

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    obj_points = []  # 3D points in read world space
    image_points = []  # 2D points in image plane

    if draw_chessboard:
        fig = plt.figure()

    for i, fn in enumerate(calibration_images):
        img = cv2.imread(fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        # Find chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object and image points
        if ret:
            obj_points.append(objp)

            # refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

            if draw_chessboard:
                cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)
                fig.add_subplot(5, 4, i + 1)  # arrange images in 4x5
                plt.imshow(img)

    if draw_chessboard:
        plt.show()

    _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, image_points, image_size, None, None)

    return mtx, dist


######################
# Image Undistortion #
######################

def undistort_image(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None)


image = cv2.imread('test_images/test4.jpg')
mtx, dist = calibrate_camera()
undistorted = undistort_image(image, mtx, dist)
cv2.imshow('', undistorted)
cv2.waitKey(0)

