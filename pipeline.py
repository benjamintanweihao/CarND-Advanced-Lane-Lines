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


#########################
# Perspective Transform #
#########################

def warp_image(image):
    (h, w, _) = image.shape

    src = np.array((
        [[607, 440],
         [670, 440],
         [1117, h],
         [194, h]
         ]), dtype=np.float32)

    offset = 250
    dst = np.array([
        [offset, 0],
        [w - offset, 0],
        [w - offset, h],
        [offset, h]], dtype=np.float32)

    # Uncomment to show red lines drawn for region of interest
    # src = src.reshape(-1, 1, 2)
    # undistorted = cv2.polylines(undistorted,
    #                             pts=np.int32([src]),
    #                             isClosed=True,
    #                             color=(0, 0, 255),
    #                             thickness=1)
    # cv2.imshow('', undistorted)
    # cv2.waitKey(0)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undistorted, M, (w, h))
    MInv = cv2.getPerspectiveTransform(dst, src)

    # Uncomment to show red lines drawn for region of interest
    # dst = dst.reshape(-1, 1, 2)
    # warped = cv2.polylines(warped,
    #                        pts=np.int32([dst]),
    #                        isClosed=True,
    #                        color=(0, 0, 255),
    #                        thickness=1)
    # cv2.imshow('', warped)
    # cv2.waitKey(0)

    return warped, M, MInv


image = cv2.imread('test_images/test2.jpg')
mtx, dist = calibrate_camera()
undistorted = undistort_image(image, mtx, dist)
warped, M, MInv = warp_image(image)

cv2.imshow('', warped)
cv2.waitKey(0)
