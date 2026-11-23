import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from mrproject import vision

def test() -> None:
    # vision.video_hough_circles()
    # vision.video_detect_aruco()

    # image = cv2.imread("/Users/tiffyg/Downloads/IMG_2404.png")
    image = cv2.imread("/Users/tiffyg/Downloads/IMG_2405.png")
    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
    print(vision.get_occupancy_grid(image, 100))