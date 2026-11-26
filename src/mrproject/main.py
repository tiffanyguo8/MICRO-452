import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from mrproject import vision
from mrproject import navigation
from kalman import *


def test() -> None:
    # vision.video_hough_circles()
    # vision.video_detect_aruco()

    # image = cv2.imread("/Users/tiffyg/Downloads/IMG_2404.png")
    image = cv2.imread("/Users/tiffyg/Downloads/IMG_2432.png")
    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
    print(vision.get_occupancy_grid(image, 1))

def state_machine() -> None:
    state = "INIT"
    resolution = 10
    cap = cv2.VideoCapture(0)  # Use cv2.CAP_DSHOW for Windows to avoid warning messages
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot capture image. Exiting")
            break

        if state == "INIT":
            Map, SearchStart, SearchGoal = vision.get_occupancy_grid(frame, 0)
            # img = np.where(Map == -1, 0, 255).astype(np.uint8)   # obstacle = black, free = white
            # cv2.imwrite('/Users/tiffyg/Downloads/clearances/test.png', img)
            state = "PLAN"

        elif state == "PLAN":

            # filtering
            x, y, heading = vision.detect_aruco(frame)
            
        

            #plan path with A*
            #smooth path
            #transfer waypoints
            path_found = navigation.nav(Map, SearchStart, SearchGoal, resolution)
            if path_found:
            #     state = "MOVE"
                print("Yay")
            # else:
            #     state = "ERROR"
            # break 


            step = 0
            if step == 0 : pos_est, P = SearchStart, np.diag([1.0, 1.0, 0.001])
            step = 1
            # add the motors' speed from sensor
            l_speed, r_speed = _, _
            pos_est, P = kalman(pos_est, l_speed, r_speed, P, dt=0.1, x_measured=SearchStart)
            
#         #elif state == "MOVE":
#             #follow path
#             #get the sensors mes

#             # if goal_reached:
#             #     state = "DONE"
#             # elif obstacle_detected:
#             #     state = "OBST_DETECTED"
    
#         # elif state == "OBST_DETECTED":
#         #     #eval if path can still followable
#         #     if path_still_valid:
#         #         state = "MOVE"
#         #     else:
#         #         state = "REPLAN"
    
#         # elif state == "REPLAN":
#         #     #act map
#         #     # get act pos°
#         #     # replan path with A*

#         #     state = "PLAN"
    
#         # elif state == "DONE":
#         #     #stop
#         #     break

#         # elif state == "ERROR":
#         #     #stop and signal pbm
#         #     break

        cv2.imshow("webcam", frame)
        
        
        if cv2.waitKey(1) == ord('x'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


