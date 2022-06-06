import cv2
import numpy as np
import time
import mediapipe as mp
from scenedetect import detect, ContentDetector
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

angle_made = []
time_bent = []
sum = 0
avg_count = 0
video_name = "KneeBendVideo.mp4"
cap1 = cv2.VideoCapture(video_name)
success1, image1 = cap1.read()
height, width, layers = image1.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap1.get(cv2.CAP_PROP_FPS)
fps = 12
video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
start_frame_time = 0


scene_list = detect(video_name, ContentDetector())
fluctated_frame_start = {}
fluctated_frame_start = set()

for i, scene in enumerate(scene_list):
    fluctated_frame_start.add(scene[0].get_frames())


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle

frame_count = 0
flag = False
temp_flag = False
count = 0
starttime = 0
start_time = []
lasttime = 0
prev_angle = 180
fluctated_frame_start.add(2175)
if(0 in fluctated_frame_start):
    fluctated_frame_start.remove(0)
flag_p = False
flag_f = False
len_temp = 0
start_frame_count = 0

cap = cv2.VideoCapture(video_name)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    output = ""
    success, image = cap.read()
    if not success:
      # If loading a video, use 'break' instead of 'continue'.
      break
    if start_frame_count == 0:
        start_frame_time = time.time()
        start_frame_count = 1
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    if(frame_count in fluctated_frame_start):
        #print("Inside")
        st = time.time()
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(image, "Fluctuated Frame",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        video.write(image)
        cv2.imshow('MediaPipe Pose', image)
        for i in range(0,10):
            #print("Inside" + str(i))
            success, image = cap.read()
            if not success:
                # If loading a video, use 'break' instead of 'continue'.
                break
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(image, "Fluctuated Frame",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            video.write(image)
            cv2.imshow('MediaPipe Pose', image)
            frame_count += 1

        et = time.time()
        len_temp += et - st


    else:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks == None:
            continue

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.

        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        angle = calculate_angle(hip, l_knee, ankle)
        angle = round(angle, 2)
        k = str(count)
        an = str(angle)
        cv2.putText(image, "Repetition : " + str(count),
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Angle made by Knee : " + str(angle),
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        #print("Angle made:" + str(angle))
        if angle < 140:
            if not flag:
                flag = True
                temp_flag = False
                starttime = time.time()
                #print(starttime)
            len1 = time.time() - starttime - len_temp
            len1 = round(len1, 2)
            cv2.putText(image, "Knee is bent for time: " + str(len1),
                        (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            if len1 > 8 and not flag_p:
                flag_p = True
                count += 1
            sum += angle
            avg_count += 1
        else:
            if flag:
                flag = False
                len1 = time.time() - starttime - len_temp
                if len1 >= 8:
                    start_time.append(starttime)
                    angle_made.append(round(sum / avg_count, 2))
                    time_bent.append(round(len1, 2))
                elif len1 > 1:
                    temp_flag = True
                sum = 0
                avg_count = 0
                flag_p = False
            len_temp = 0

        if(temp_flag):
            cv2.putText(image, "Keep your knees bent",
                        (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        video.write(image)
        cv2.imshow('MediaPipe Pose', image)

    frame_count += 1
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
video.release()

f = open("stats.txt", "a")

for i in range(len(time_bent)):
    f.write("The " + str(i + 1) + " Knee bend" + " started at " + str(round(start_time[i] - start_frame_time, 2)) +  "sec and was for duration: " + str(time_bent[i]) + "sec during which average angle made by the knee was: " + str(angle_made[i]) + "Deg")
    f.write("\n")

f.close()