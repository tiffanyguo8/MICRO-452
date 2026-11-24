import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from mrproject import vision

from navigation.py import nav

def test() -> None:
    # vision.video_hough_circles()
    vision.video_detect_aruco()

    # image = cv2.imread("/Users/tiffyg/Downloads/IMG_2404.png")
    # image = cv2.imread("/Users/tiffyg/Downloads/IMG_2405.png")
    # image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
    # print(vision.get_occupancy_grid(image, 100))

def main() -> None:
    state = "INIT"

    while True:
        if state == "INIT":
            #get map, start, goal
            state = "PLAN"

        elif state == "PLAN":
            #plan path with A*
            #smooth path
            #transfer waypoints
            path_found = nav(Map, SearchStart, SearchGoal, "Octile")
            if path_found:
                state = "MOVE"
            else:
                state = "ERROR" 

        elif state == "MOVE":
            #follow path
            #get the sensors mes

            if goal_reached:
                state = "DONE"
            elif obstacle_detected:
                state = "OBST_DETECTED"
    
        elif state == "OBST_DETECTED":
            #eval if path can still followable
            if path_still_valid:
                state = "MOVE"
            else:
                state = "REPLAN"
    
        elif state == "REPLAN":
            #act map
            # get act pos°
            # replan path with A*

            state = "PLAN"
    
        elif state == "DONE":
            #stop
            break

        elif state == "ERROR":
            #stop and signal pbm
            break