import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def test() -> None:
    # open camera and create VideoCapture object
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while(1):
        #capture image
        ret, frame = cap.read()
        if not ret:
            print("Cannot capture image. Exiting")
            break
        
        # find the aruco
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, id, rejected = detector.detectMarkers(frame)

        ## print aruco detector outputs
        # print("corners", corners)
        # print("id", id)

        # draw box around aruco
        if(len(corners) > 0):
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, id):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(markerID))

        cv2.imshow("webcam", frame)


        if cv2.waitKey(1) == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()