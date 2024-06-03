import math
import time
import cv2 as cv
import numpy as np
import requests
import urllib3
urllib3.disable_warnings()
import pyautogui
import pygetwindow as gw
import pytesseract

import gymnasium as gym
from gymnasium import spaces

def _capture_screen(x, y, w, h):
        screenshot = pyautogui.screenshot(region=(x,y,w,h))
        screenshot_np = np.array(screenshot)
        return screenshot_np

def _locate_area(screenshot, template):
    # Convert to grayscale
    gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]

    result = cv.matchTemplate(gray, template, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # Draw a rectangle around the matched area
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return int(top_left[0] + (w / 2)), int(top_left[1] + (h / 2))

# time.sleep(2)

# minimap = _capture_screen(1522, 680, 380, 380)
# player = cv.imread("assets\yi_border.jpg")
# x, y = _locate_area(minimap, player)

# print(x, y)

# img = np.zeros((380,380,3), np.uint8)

# cv.rectangle(img, (0, 0), (380, 380), (0, 255, 0), 2)
# cv.circle(img, (x, y), 10, (0,0,255), 1)
# cv.imwrite("result.png", img)
# cv.imshow("Result", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

player_pos = np.array([1, 4])

# Example jungle monsters' positions (10 monsters with x, y coordinates)
jungle_monsters_pos = np.array([
    [2, 5],
    [2, 8],
    [4, 4],
    [4, 5],
    [6, 6],
    [7, 3],
    [5, 7],
    [8, 1],
    [9, 2],
    [3, 5]
])

distances = np.linalg.norm(jungle_monsters_pos - player_pos, axis=1)

print(distances)
