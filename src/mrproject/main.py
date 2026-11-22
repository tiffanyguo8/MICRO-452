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
    print(vision.hough_circles(image))