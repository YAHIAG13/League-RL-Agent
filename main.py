import math
import time
import cv2 as cv
import numpy as np
import requests
import urllib3
urllib3.disable_warnings()

from matplotlib import pyplot as plt

import pyautogui
import pygetwindow as gw
from pywinauto.application import Application


def capture_map_screen():
    screenshot = pyautogui.screenshot(region=(1646,806,270,270))
    screenshot_np = np.array(screenshot)
    return screenshot_np

def find_player_position_in_map(screenshot):
    # Convert to grayscale
    gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    template = cv.imread("assets\yi_border.jpg")
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    cv.imshow("gray", gray)
    cv.imshow("temp", template)
    w, h = template.shape[::-1]

    result = cv.matchTemplate(gray, template, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    

    # Draw a rectangle around the matched area
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)




def get_player_data():
    url = "https://127.0.0.1:2999/liveclientdata/allgamedata"
    response = requests.get(url, verify=False)
    return response.json()

def find_window(title):
    windows = gw.getWindowsWithTitle(title)
    if windows:
        return windows[0]
    else:
        return None
    
def activate_and_get_size(window):
    if window:
        window.activate()
        time.sleep(1)  # Wait for the window to become active
        width, height = window.width, window.height
        return width, height
    else:
        print("Window not found")
        return None, None
    
window = find_window("League of Legends (TM) Client")
width, height = activate_and_get_size(window)

steps = 5
step = steps
angle_step = 2 * math.pi / steps
center_x = 870
center_y = 510
radius = 150

ss = capture_map_screen()
find_player_position_in_map(ss)

# main loop
while(gw.getActiveWindowTitle() == "League of Legends (TM) Client"):

    # get player stats
    data = get_player_data()
    maxHealth = data['activePlayer']['championStats']['maxHealth']
    currentHealth = data['activePlayer']['championStats']['currentHealth']
    maxMana = data['activePlayer']['championStats']['resourceMax']
    currentMana = data['activePlayer']['championStats']['resourceValue']

    print("health: ", currentHealth, "/", maxHealth)
    print("mana: ", currentMana, "/", maxMana)

    # perform actions
    ## movement
    step -= 1

    if step == 0:
        step = steps

    angle = step * angle_step
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    pyautogui.click(x, y, button="SECONDARY")

    time.sleep(1)
