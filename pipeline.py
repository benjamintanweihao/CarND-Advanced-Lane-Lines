import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    return cv2.undistort(image, mtx, dist, None)


def region_of_interest(image):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(image)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    h, w = image.shape[0], image.shape[1]

    # 4. Define a polygon to mask
    vertices = np.array([[(727, 450), (598, 450), (180, h), (1240, h)]], dtype=np.int32)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


#########################
# Perspective Transform #
#########################

def warp_image(threshold):
    h, w = threshold.shape[0], threshold.shape[1]

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
    # undistorted = cv2.polylines(threshold,
    #                             pts=np.int32([src]),
    #                             isClosed=True,
    #                             color=(0, 0, 255),
    #                             thickness=1)
    # cv2.imshow('', undistorted)
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


######################
# Color and Gradient #
######################

def color_and_gradient_threshold(undistort, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # grayscale
    gray = cv2.cvtColor(undistort, cv2.COLOR_BGR2GRAY)

    # convert to HLS
    hls = cv2.cvtColor(undistort, cv2.COLOR_BGR2HLS).astype(np.float)
    # separate the S channel
    s_channel = hls[:, :, 2]

    # Sobel X
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gray)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


def compute_best_fit(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
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
    margin = 100
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

    binary = color_and_gradient_threshold(undistorted)
    binary_cropped = region_of_interest(binary)
    binary_warped, M, MInv = warp_image(binary_cropped)

    ret_1 = compute_best_fit(binary_warped)
    ret_2 = compute_curvature(binary_warped, ret_1)
    result = color_lane(orig, undistorted, binary_warped, MInv, ret_1, ret_2)

    return result


mtx, dist = calibrate_camera()
orig = cv2.imread('./test_images/test4.jpg')
undistorted = undistort_image(orig, mtx, dist)

binary = color_and_gradient_threshold(undistorted)
binary_cropped = region_of_interest(binary)
binary_warped, M, MInv = warp_image(binary_cropped)

ret_1 = compute_best_fit(binary_warped)
ret_2 = compute_curvature(binary_warped, ret_1)
result = color_lane(orig, undistorted, binary_warped, MInv, ret_1, ret_2)

cv2.imshow('', result)
cv2.waitKey(0)




def process_image(image):
    return pipeline(image, mtx, dist)

# # video_file_name = "project_video.mp4"
# # video_file_name = "challenge_video.mp4"
# video_file_name = "harder_challenge_video.mp4"
#
# white_output = 'test_video_output/' + video_file_name
# clip1 = VideoFileClip(video_file_name)
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(white_output, audio=False)
