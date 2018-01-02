import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_binary(bgr_img):
    threshold = [10, 255]

    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    bin = np.zeros_like(bgr_img)
    bin[(bgr_img >= threshold[0]) & (bgr_img <= threshold[1])] = 1

    return bin


def filter_white_and_yellow(undistort):
    # white_bgr_lower = np.array([100, 100, 200], dtype=np.uint8)
    # white_bgr_upper = np.array([255, 255, 255], dtype=np.uint8)
    # white_bgr_range = cv2.inRange(img, white_bgr_lower, white_bgr_upper)
    # white_bgr = cv2.bitwise_and(img, img, mask=white_bgr_range)
    # white_bgr = convert_to_binary(white_bgr)
    #
    # yellow_bgr_lower = np.array([0, 180, 225], dtype=np.uint8)
    # yellow_bgr_upper = np.array([170, 255, 255], dtype=np.uint8)
    # yellow_bgr_range = cv2.inRange(img, yellow_bgr_lower, yellow_bgr_upper)
    # yellow_bgr = cv2.bitwise_and(img, img, mask=yellow_bgr_range)
    # yellow_bgr = convert_to_binary(yellow_bgr)
    #
    # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    # yellow_upper = np.array([40, 200, 255], dtype=np.uint8)
    # yellow_range = cv2.inRange(hls, yellow_lower, yellow_upper)
    #
    # white_lower = np.array([200, 100, 100], dtype=np.uint8)
    # white_upper = np.array([255, 255, 255], dtype=np.uint8)
    # white_range = cv2.inRange(hls, white_lower, white_upper)
    # yellows_or_whites = yellow_range | white_range
    # hls = cv2.bitwise_and(img, img, mask=yellows_or_whites)
    # hls = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    # hls = convert_to_binary(hls)
    #
    # return hls, white_bgr, yellow_bgr
    #

    # undistort = cv2.cvtColor(undistort, cv2.COLOR_RGB2BGR)

    white_bgr_lower = np.array([100, 100, 200], dtype=np.uint8)
    white_bgr_upper = np.array([255, 255, 255], dtype=np.uint8)
    white_bgr_range = cv2.inRange(undistort, white_bgr_lower, white_bgr_upper)
    white_bgr = cv2.bitwise_and(undistort, undistort, mask=white_bgr_range)
    white_bgr = convert_to_binary(white_bgr)

    yellow_bgr_lower = np.array([84, 191, 200], dtype=np.uint8)
    yellow_bgr_upper = np.array([170, 255, 255], dtype=np.uint8)
    yellow_bgr_range = cv2.inRange(undistort, yellow_bgr_lower, yellow_bgr_upper)
    yellow_bgr = cv2.bitwise_and(undistort, undistort, mask=yellow_bgr_range)
    yellow_bgr = convert_to_binary(yellow_bgr)

    hls = cv2.cvtColor(undistort, cv2.COLOR_BGR2HLS)
    yellow_lower = np.array([20, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([40, 200, 255], dtype=np.uint8)
    yellow_range = cv2.inRange(hls, yellow_lower, yellow_upper)

    white_dark = np.array([0, 0, 0], dtype=np.uint8)
    white_light = np.array([0, 0, 255], dtype=np.uint8)
    white_range = cv2.inRange(hls, white_dark, white_light)
    yellows_or_whites = yellow_range | white_range

    hls = cv2.bitwise_and(undistort, undistort, mask=yellows_or_whites)
    hls = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    hls = convert_to_binary(hls)

    combined_binary = np.zeros_like(hls)
    combined_binary[(hls == 1) | (white_bgr == 1) | (yellow_bgr == 1)] = 1

    return hls, white_bgr, yellow_bgr


orig = cv2.imread("test_images/test4.jpg")

hls, white_bgr, yellow_bgr = filter_white_and_yellow(orig)

combine = np.zeros_like(hls)
combine[(hls == 1) | (white_bgr == 1) | (yellow_bgr == 1)] = 1

plt.imshow(combine, cmap='gray')
plt.show()
