# KneeBendDetection
In order to execute this program keep the code and kneebend.mp4 video in same folder
Create a virtual Environment and execute the requirements.txt file by command pip install -r /path/to/requirements.txt
Then execute the main.py file.
Output video file is provided by name output.mp4

Discussion on Approach
1) I have used Mediapipe module for detecting pose of individual.
2) From pose I have calculated the angle made by knee from coordinates of knee, hip and ankle (all left sided).
    Relation used is Angle between hip and knee subtracted by angle between knee and ankle.
    If knee was not properly bent for 8sec warning sign is displayed till knee is not bent.
3) I have also detected fluctuated frames in input video using scenedetect library, fluctuated frames are accurately detected and are not considered in output they
    are changed to BGR format and term fluctuated is displayed on the frame.
4) User Stats of Bendness are provided in a separate file stats.txt.
