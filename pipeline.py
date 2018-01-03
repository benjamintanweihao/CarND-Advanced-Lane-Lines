import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

######################
# Camera Calibration #
######################
from moviepy.video.io.VideoFileClip import VideoFileClip


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
    image = np.copy(image)
    return cv2.undistort(image, mtx, dist, None)


#########################
# Perspective Transform #
#########################

def warp_image(threshold):
    threshold = np.copy(threshold)
    h, w = threshold.shape[0], threshold.shape[1]

    src = np.array((
        [[576, 461],
         [706, 461],
         [1117, h],
         [194, h]
         ]), dtype=np.float32)

    offset = 300
    dst = np.array([
        [offset, 0],
        [w - offset, 0],
        [w - offset, h],
        [offset, h]], dtype=np.float32)

    # Uncomment to show red lines drawn for region of interest
    # src = src.reshape(-1, 1, 2)
    # undistorted = cv2.polylines(threshold,
    #                             pts=np.int32([src]),
    #                             isClosed=True,
    #                             color=(0, 0, 255),
    #                             thickness=1)
    # cv2.imshow('', undistorted)000
    # cv2.waitKey(0)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(threshold, M, (w, h))
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


###################
# Color Threshold #
###################

def convert_to_binary(bgr_img):
    bgr_img = np.copy(bgr_img)
    threshold = [10, 255]

    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    bin = np.zeros_like(bgr_img)
    bin[(bgr_img >= threshold[0]) & (bgr_img <= threshold[1])] = 1

    return bin


def color_threshold(undistort):
    undistort = np.copy(undistort)
    undistort = cv2.cvtColor(undistort, cv2.COLOR_RGB2BGR)

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

    return combined_binary


def compute_best_fit(binary_warped):
    binary_warped = np.copy(binary_warped)
    # Assuming you have created a warped binary image called "binary_warped" pixethe
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,
                      (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)

        cv2.rectangle(out_img,
                      (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high),
                      (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    #
    # plt.show()

    ret = {'left_fit': left_fit,
           'left_fitx': left_fitx,
           'leftx': leftx,
           'lefty': lefty,
           'right_fit': right_fit,
           'right_fitx': right_fitx,
           'rightx': rightx,
           'righty': righty,
           'ploty': ploty}

    return ret


# A*y**2 + B*y + C
# Radius = (1 + (2*A*y + B)**2)**(3/2) / abs(2*A)

def compute_curvature(binary_warped, ret):
    left_fit = ret['left_fit']
    lefty = ret['lefty']
    leftx = ret['leftx']
    right_fit = ret['right_fit']
    righty = ret['righty']
    rightx = ret['rightx']

    y_eval = binary_warped.shape[0] - 1

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curve_rad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    left_pos = (left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2])
    right_pos = (right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2])
    middle_pos = (left_pos + right_pos) / 2.0
    mid_dist = binary_warped.shape[1] / 2.0 - middle_pos
    mid_dist_in_m = xm_per_pix * mid_dist

    ret = {'left_curvature': left_curve_rad,
           'right_curvature': right_curve_rad,
           'middle_distance': mid_dist_in_m}

    return ret


def color_lane(orig, undist, warped, MInv, ret_1, ret_2):
    left_fitx = ret_1['left_fitx']
    right_fitx = ret_1['right_fitx']
    ploty = ret_1['ploty']

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, MInv, (orig.shape[1], orig.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    left_curve_rad = ret_2['left_curvature']
    right_curve_rad = ret_2['right_curvature']
    middle_distance = ret_2['middle_distance']

    curvature = 'Radius: ' + str(left_curve_rad) + ' m, ' + str(right_curve_rad) + " m"
    lane_dist = 'Distance From Road Center: ' + str(middle_distance) + ' m'
    font = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.putText(result, curvature, (25, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    result = cv2.putText(result, lane_dist, (25, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result


def pipeline(orig, mtx, dist):
    undistorted = undistort_image(orig, mtx, dist)
    binary = color_threshold(undistorted)
    binary_warped, M, MInv = warp_image(binary)
    ret_1 = compute_best_fit(binary_warped)
    ret_2 = compute_curvature(binary_warped, ret_1)
    result = color_lane(orig, undistorted, binary_warped, MInv, ret_1, ret_2)

    return result


def process_image(image):
    try:
        return pipeline(image, mtx, dist)
    except Exception:
        cv2.imshow('', image)
        cv2.waitKey(0)
        return image


mtx, dist = calibrate_camera()


video_file_name = "project_video.mp4"
write_output = 'test_video_output/' + video_file_name
clip1 = VideoFileClip(video_file_name)
clip2 = clip1.fl_image(process_image)
clip2.write_videofile(write_output, audio=False)

# orig = cv2.imread("test_images/project_video/frame-14.jpg")
# result = pipeline(orig, mtx, dist)
# plt.imshow(result)
# plt.show()
