import cv2
from matplotlib import pyplot as plt

def test() -> None:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    plt.imshow(frame)
    plt.show()