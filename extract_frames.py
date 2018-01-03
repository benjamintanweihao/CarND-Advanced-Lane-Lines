import cv2

vidcap = cv2.VideoCapture('./project_video.mp4')
success,image = vidcap.read()
count = 0
success = True

while success:
    success,image = vidcap.read()
    cv2.imwrite("./test_images/project_video/frame-%d.jpg" % count, image)
    count += 1