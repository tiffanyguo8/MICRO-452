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

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        if id is not None and 28 in id:
            idx = list(id.flatten()).index(28)
            # loop over the detected ArUCo corners
            pts = corners[idx].reshape((4,2))
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
            cv2.putText(frame, str(cX) + ", " + str(h-cY),
                (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

            # display coordinates of center of aruco marker at top left corner
            cv2.putText(frame, str(cX) + ", " + str(h-cY),
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
    # get pixel height for y-axis inversion
    h = image.shape[0]

    # find the aruco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, id, rejected = detector.detectMarkers(image)

    if id is not None and 28 in id:
        idx = list(id.flatten()).index(28)
        # loop over the detected ArUCo corners
        pts = corners[idx].reshape((4,2))
        # separate into four corner variables
        (tl, tr, br, bl) = pts

        # compute and draw the center (x, y)-coordinates of the ArUco
        cX = int((tl[0] + br[0]) / 2.0)
        cY = int((tl[1] + br[1]) / 2.0)
        cY = h - cY

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

        circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,100, param1=50,param2=30,minRadius=40,maxRadius=200)
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

    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,100, param1=50,param2=30,minRadius=40,maxRadius=200)

    if circles is not None:
        circles = circles[0]
        circles = np.uint16(np.around(circles))

    return circles

# returns 192 x 108 matrix with -1 for pixels with objects and 0 for empty pixels (that thymio can travel to)
# also returns the [x, y] coordinates of thymio and goal
# @param clearance: # of pixels to increase circle obstacle radius by. use a value less than 100
def get_occupancy_grid(image, clearance: int) ->tuple[np.ndarray, list[list[int]]]:
    # get pixel width and height of frame for setting hough circle parameters
    w = image.shape[1]
    h = image.shape[0]

    # get circles
    obstacles = hough_circles(image)

    # create 1920 x 1080 array and initialize every pixel to 0 (false)
    mask = np.zeros((h, w), dtype=bool)

    # turn pixels within obstacle circles to 1 (true)
    for circle in obstacles:
        cx = circle[0]
        cy = circle[1]
        r = circle[2] + clearance
        ys, xs = np.ogrid[0:h, 0:w]
        mask |= ((xs - cx)**2 + (ys - cy)**2 <= r**2)

    # get aruco codes at four corners
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, id, rejected = detector.detectMarkers(image)

    # get the coordinate of the center of each aruco
    # id 0 = bottom right
    # id 1 = bottom left
    # id 2 = top left
    # id 3 = top right

    if id is not None:
        try:
            idx = list(id.flatten()).index(0)
            pts = corners[idx].reshape((4,2))
            (tl, tr, br, bl) = pts
            br_cX = int((tl[0] + br[0]) / 2.0)
            br_cY = int((tl[1] + br[1]) / 2.0)
            br_cY = h - br_cY

            idx = list(id.flatten()).index(1)
            pts = corners[idx].reshape((4,2))
            (tl, tr, br, bl) = pts
            bl_cX = int((tl[0] + br[0]) / 2.0)
            bl_cY = int((tl[1] + br[1]) / 2.0)
            bl_cY = h - bl_cY

            idx = list(id.flatten()).index(2)
            pts = corners[idx].reshape((4,2))
            (tl, tr, br, bl) = pts
            tl_cX = int((tl[0] + br[0]) / 2.0)
            tl_cY = int((tl[1] + br[1]) / 2.0)
            tl_cY = h - tl_cY

            idx = list(id.flatten()).index(3)
            pts = corners[idx].reshape((4,2))
            (tl, tr, br, bl) = pts
            tr_cX = int((tl[0] + br[0]) / 2.0)
            tr_cY = int((tl[1] + br[1]) / 2.0)
            tr_cY = h - tr_cY

            idx = list(id.flatten()).index(28)
            pts = corners[idx].reshape((4,2))
            (tl, tr, br, bl) = pts
            thymio_x = int((tl[0] + br[0]) / 2.0)
            thymio_y = int((tl[1] + br[1]) / 2.0)                 # no "h - ..."
            thymio_position = (thymio_y // 10, thymio_x // 10)    # (row, col)

            idx = list(id.flatten()).index(8)
            pts = corners[idx].reshape((4,2))
            (tl, tr, br, bl) = pts
            goal_x = int((tl[0] + br[0]) / 2.0)
            goal_y = int((tl[1] + br[1]) / 2.0)                   # no "h - ..."
            goal_position = (goal_y // 10, goal_x // 10)          # (row, col)
        except:
            print("Did not detect four corners, goal, and Thymio. Quitting...")

    # set all coordinates outside of aruco corners to -1
    pts = np.array([[tl_cX, h - tl_cY], [tr_cX, h - tr_cY], [br_cX, h - br_cY], [bl_cX, h - bl_cY]], dtype=np.int32)
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(region, [pts], 1)   # inside rectangle = 1
    mask |= (region == 0)            # add everything outside the rectangle to the circles

    grid = np.where(mask, 1, 0).astype(np.int8)

    # reduce resolution for A*
    fraction = grid.reshape(h//10, 10, w//10, 10).mean(axis=(1, 3))
    reduced_grid = np.where(fraction >= 0.5, -1, 0).astype(np.int8)

    # img = np.where(reduced_grid == -1, 0, 255).astype(np.uint8)   # obstacle = black, free = white
    # cv2.imwrite('/Users/tiffyg/Downloads/clearances/goal.png', img)

    img = np.where(reduced_grid == -1, 0, 255).astype(np.uint8)   # obstacle = black, free = white
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                   # convert to color

    img[thymio_position[0], thymio_position[1]] = (0, 0, 255)   # red
    img[goal_position[0], goal_position[1]] = (0, 255, 0)       # green

    cv2.imwrite('/Users/tiffyg/Downloads/clearances/goal.png', img)
    
    return reduced_grid, thymio_position, goal_position