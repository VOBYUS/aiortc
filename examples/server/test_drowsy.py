import drowsiness_stable.dd as dd
import cv2

# loop over frames from the video stream
path = 0
dd.stream = cv2.VideoCapture(path)


while True:
    (grabbed, frame) = dd.stream.read()
    (frame, drowsy_level, ret) = dd.process_image(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key != 0xFF:
        break 

# do a bit of cleanup
dd.stream.release()
cv2.destroyAllWindows()
