import math
import time
import cv2 as cv
import numpy as np
import requests
import urllib3
urllib3.disable_warnings()
import pytesseract

from matplotlib import pyplot as plt

import pyautogui
import pygetwindow as gw
from pywinauto.application import Application

def _find_window( title):
        windows = gw.getWindowsWithTitle(title)
        print(windows)

        if windows:
            return windows[0]
        else:
            return None
        
def _activate_and_get_size( window):
    if window:

        window.activate()

        time.sleep(2)  # Wait for the window to become active
        x, y, w, h = window.top, window.left, window.width, window.height

        return x, y, w, h
    else:
        print("Window not found")
        return None, None, None, None

def _capture_screen(x, y, w, h):
    screenshot = pyautogui.screenshot(region=(x,y,w,h))
    screenshot_np = np.array(screenshot)
    return screenshot_np

def _locate_area( screenshot, template):
    # Convert to grayscale
    gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]

    result = cv.matchTemplate(gray, template, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # Draw a rectangle around the matched area
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv.rectangle(screenshot, top_left, bottom_right, (0, 255, 0), 2)
    # cv.imwrite("result.png", screenshot)
    # cv.imshow("Result", screenshot)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return top_left[0] + (w / 2), top_left[1] + (h / 2)



## switch to client
window = _find_window("League of Legends")
x, y, w, h = _activate_and_get_size(window)
print(gw.getActiveWindowTitle())

## play
screen_capture = _capture_screen(0, 0, 1920, 1080)
play_template = cv.imread("assets\play_btn.jpg")
x, y = _locate_area(screen_capture, play_template)
pyautogui.click(x, y, button="PRIMARY")

## wait
time.sleep(2)

screen_capture = _capture_screen(0, 0, 1920, 1080)
play_template = cv.imread("assets\\training.jpg")
x, y = _locate_area(screen_capture, play_template)
pyautogui.click(x, y, button="PRIMARY")

## wait
time.sleep(2)

screen_capture = _capture_screen(0, 0, 1920, 1080)
play_template = cv.imread("assets\practice_tool.jpg")
x, y = _locate_area(screen_capture, play_template)
pyautogui.click(x, y, button="PRIMARY")

## wait
time.sleep(2)

screen_capture = _capture_screen(0, 0, 1920, 1080)
play_template = cv.imread("assets\confirm.jpg")
x, y = _locate_area(screen_capture, play_template)
pyautogui.click(x, y, button="PRIMARY")

## wait
time.sleep(2)

screen_capture = _capture_screen(0, 0, 1920, 1080)
play_template = cv.imread("assets\start.jpg")
x, y = _locate_area(screen_capture, play_template)
pyautogui.click(x, y, button="PRIMARY")

## wait
time.sleep(4)

screen_capture = _capture_screen(0, 0, 1920, 1080)
play_template = cv.imread("assets\select_yi.jpg")
x, y = _locate_area(screen_capture, play_template)
pyautogui.click(x, y, button="PRIMARY")

## wait
time.sleep(2)

screen_capture = _capture_screen(0, 0, 1920, 1080)
play_template = cv.imread("assets\lock_in.jpg")
x, y = _locate_area(screen_capture, play_template)
pyautogui.click(x, y, button="PRIMARY")

## wait

time.sleep(14)

while (gw.getActiveWindowTitle() != "League of Legends (TM) Client") :
    window = _find_window("League of Legends (TM) Client")
    print("Waiting for game to start ...")
    time.sleep(2)

print("Waiting for game to load ...")
time.sleep(10)

def get_player_data():
    url = "https://127.0.0.1:2999/liveclientdata/allgamedata"
    response = requests.get(url, verify=False)
    return response

print("Reseting observations ...")

while(True) :
    try :
        response = get_player_data()

        if response.status_code == 200:
            print("Received a valid response")
            data = response.json()
            break
    except requests.RequestException as e:
        print(f"Request failed: {e}")

    print(f"Retrying in {2} seconds...")
    time.sleep(2)

print(data)
