import cv2

vidcap = cv2.VideoCapture('./challenge_video.mp4')
success,image = vidcap.read()
count = 0
success = True

while success:
    success,image = vidcap.read()
    cv2.imwrite("./test_images/challenge_video/frame-%d.jpg" % count, image)
    count += 1