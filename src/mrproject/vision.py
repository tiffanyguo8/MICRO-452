import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# preconditions: only one aruco marker in image
# shows a video stream with a red dot marking the center of the aruco marker
def video_detect_aruco() -> None:
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
            pts = corners[0].reshape((4,2))
            # separate into four corner variables
            (tl, tr, br, bl) = pts
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(tr[0]), int(tr[1]))
            bottomRight = (int(br[0]), int(br[1]))
            bottomLeft = (int(bl[0]), int(bl[1]))
            topLeft = (int(tl[0]), int(tl[1]))

            markerID = int(id[0])

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(frame, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            #label coordinates of center of aruco
            cv2.putText(frame, str(cX) + ", " + str(cY),
                (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

            # display coordinates of center of aruco marker at top left corner
            cv2.putText(frame, str(cX) + ", " + str(cY),
            (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
            3, (0, 255, 0), 2)

            # find heading from corner coordinates –— heading = direction from the center to the midpoint of the top edge
            dx = tr[0] - tl[0]
            dy = tr[1] - tl[1]
            heading = math.atan2(-dy, dx)   # flip y: image y points down

            top_mid = (int(cX + dx), int(cY + dy))

            # display heading as arrow
            cv2.arrowedLine(frame, (int(cX), int(cY)),
                top_mid, (255, 0, 0), 2)

            cv2.putText(frame, f'{heading:.3f}',
            (1000, 120), cv2.FONT_HERSHEY_SIMPLEX,
            3, (0, 255, 0), 2)
            
            print("[INFO] ArUco marker ID: {}".format(markerID))

        cv2.imshow("webcam", frame)


        if cv2.waitKey(1) == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

# preconditions: only one aruco marker in image
# returns [x, y, h] coordinates and heading of center of marker as a list. heading is towards the right. 
def detect_aruco(image) -> list[int]:
    # find the aruco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, id, rejected = detector.detectMarkers(image)

    if(len(corners) > 0):
        # loop over the detected ArUCo corners
        pts = corners[0].reshape((4,2))
        # separate into four corner variables
        (tl, tr, br, bl) = pts

        # compute and draw the center (x, y)-coordinates of the ArUco
        cX = int((tl[0] + br[0]) / 2.0)
        cY = int((tl[1] + br[1]) / 2.0)

        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]
        heading = math.atan2(-dy, dx)   # flip y: image y points down

    return [cX, cY, heading]

def video_hough_circles() -> None:
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

        image = cv2.medianBlur(frame,5)
        cimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,100, param1=50,param2=30,minRadius=40,maxRadius=100)
        # min and max radius determined by experimentation
        # print(circles[:, 2].min())

        if circles is not None:
            circles = circles[0]
            circles = np.uint16(np.around(circles))
            for x, y, r in circles:
                # draw the outer circle
                cv2.circle(image,(x,y),r,(0,255,0),2)
                # draw the center of the circle
                cv2.circle(image,(x,y),2,(0,0,255),3)

                #label coordinates of center of aruco
                cv2.putText(image, str(x) + ", " + str(y),
                (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

        cv2.imshow("webcam", image)
        
        if cv2.waitKey(1) == ord('x'):
            break
    cap.release()
    cv2.destroyAllWindows()

# returns list of [x, y, r] of circle coordinates and radius as a list
def hough_circles(image) -> list[int]:
    image = cv2.medianBlur(image, 5)
    cimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,100, param1=50,param2=30,minRadius=40,maxRadius=100)

    if circles is not None:
        circles = circles[0]
        circles = np.uint16(np.around(circles))

    return circles

# returns 192 x 108 matrix with -1 for pixels with objects and 0 for empty pixels (that thymio can travel to)
# @param clearance: # of pixels to increase circle obstacle radius by. use a value less than 100
def get_occupancy_grid(image, clearance: int) -> np.ndarray:
    # get pixel width and height of frame for setting hough circle parameters
    w = image.shape[1]
    h = image.shape[0]

    obstacles = hough_circles(image)

    print(obstacles)

    # create 1920 x 1080 array and initialize every pixel to 0 (false)
    mask = np.zeros((h, w), dtype=bool)

    # turn pixels within obstacle circles to 1 (true)
    for circle in obstacles:
        cx = circle[0]
        cy = circle[1]
        r = circle[2] + clearance
        ys, xs = np.ogrid[0:h, 0:w]
        mask |= ((xs - cx)**2 + (ys - cy)**2 <= r**2)

    grid = np.where(mask, 1, 0).astype(np.int8)

    # reduce resolution for A*
    fraction = grid.reshape(h//10, 10, w//10, 10).mean(axis=(1, 3))
    reduced_grid = np.where(fraction >= 0.5, -1, 0).astype(np.int8)

    # img = np.where(reduced_grid == -1, 0, 255).astype(np.uint8)   # obstacle = black, free = white
    # cv2.imwrite('/Users/tiffyg/Downloads/clearances/100.png', img)
    
    return reduced_grid