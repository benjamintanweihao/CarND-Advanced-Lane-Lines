import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_binary(bgr_img):
    threshold = [10, 255]

    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    bin = np.zeros_like(bgr_img)
    bin[(bgr_img >= threshold[0]) & (bgr_img <= threshold[1])] = 1

    return bin

def color_threshold(undistort):
    undistort = np.copy(undistort)
    undistort_rgb = cv2.cvtColor(undistort, cv2.COLOR_BGR2RGB)
    undistort_hls = cv2.cvtColor(undistort, cv2.COLOR_BGR2HLS)
    undistort_hsv = cv2.cvtColor(undistort, cv2.COLOR_BGR2HSV)

    white_rgb_lower = np.array([100, 100, 200], dtype=np.uint8)
    white_rgb_upper = np.array([255, 255, 255], dtype=np.uint8)
    white_rgb_range = cv2.inRange(undistort_rgb, white_rgb_lower, white_rgb_upper)
    white_rgb = cv2.bitwise_and(undistort_rgb, undistort_rgb, mask=white_rgb_range)
    white_rgb = convert_to_binary(white_rgb)

    yellow_lower = np.array([20, 100, 100], dtype=np.uyyint8)
    yellow_upper = np.array([40, 200, 255], dtype=np.uint8)
    yellow_range = cv2.inRange(undistort_hls, yellow_lower, yellow_upper)
    yellow = cv2.bitwise_and(undistort_hls, undistort_hls, mask=yellow_range)
    yellow = cv2.cvtColor(yellow, cv2.COLOR_HLS2BGR)
    yellow = convert_to_binary(yellow)

    white_lower = np.array([0, 0, 250], dtype=np.uint8)
    white_upper = np.array([255, 100, 255], dtype=np.uint8)
    white_range = cv2.inRange(undistort_hsv, white_lower, white_upper)
    white = cv2.bitwise_and(undistort_hsv, undistort_hsv, mask=white_range)
    white = cv2.cvtColor(white, cv2.COLOR_HSV2BGR)
    white = convert_to_binary(white)

    combined_binary = np.zeros_like(yellow)
    combined_binary[(white == 1) | (white_rgb == 1) | (yellow == 1)] = 1

    # return combined_binary
    return yellow
    # return yellow_bgr
    # return combined_binary


orig = cv2.imread("test_images/test4.jpg")
result = color_threshold(orig)
plt.imshow(result, cmap='gray')
plt.show()
