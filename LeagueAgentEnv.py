import math
import time
import cv2 as cv
import numpy as np
import requests
import urllib3
urllib3.disable_warnings()
import pydirectinput
import pyautogui
import pygetwindow as gw

pydirectinput.FAILSAFE = False
pyautogui.FAILSAFE = False

import gymnasium as gym
from gymnasium import spaces

MINIMAP_AREA = (1522, 680, 380, 380)


class LeagueAgentEnv(gym.Env):

    def __init__(self, size = 380, render_mode=None):
        
        self.size = size  # The size of the square grid

        self.epoch_number = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                # game data
                "time" : spaces.Discrete(9999),
                "side" : spaces.Discrete(2), # 0 -> bottom side (ORDER), 1 -> top side (CHEOS)


                # champions
                "champion_position": spaces.Box(0, size - 1, shape=(2,), dtype=np.float32),
                "champion_health" : spaces.Discrete(9999),
                "champion_mana" : spaces.Discrete(9999),
                "champion_level" : spaces.Discrete(19),
                # "champions_health": spaces.Box(0, 1, shape=(10,), dtype=np.float32),
                # "champions_mana": spaces.Box(0, 1, shape=(10,), dtype=np.float32),
                # "champions_level": spaces.Box(0, 18, shape=(10,), dtype=np.int32),

                # player stats :
                "gold" : spaces.Discrete(9999),

                "abilities_cooldown": spaces.Box(0, 1, shape=(4,), dtype=np.float32),
                "abilities_level" : spaces.Box(0, 5, shape=(4,), dtype=np.int32),
                "spells_cooldown": spaces.Box(0, 1, shape=(2,), dtype=np.float32),

                # jungle
                "jungle_monsters_position": spaces.Box(0, size - 1, shape=(10,2), dtype=np.float32),
                "jungle_monsters_health": spaces.Box(0, 1, shape=(10,), dtype=np.float32),
                "jungle_monsters_timer": spaces.Box(0, 1, shape=(10,), dtype=np.float32),

                # structures
                # "towers_position": spaces.Box(0, size - 1, shape=(11,2,2), dtype=np.float32),
                # "towers_health": spaces.Box(0, 1, shape=(11,2,), dtype=np.float32),

                # "inhib_position": spaces.Box(0, size - 1, shape=(6,2,2), dtype=np.float32),
                # "inhib_health": spaces.Box(0, 1, shape=(6,2,), dtype=np.float32),

                # "nexus_position": spaces.Box(0, size - 1, shape=(2,2,2), dtype=np.float32),
                # "nexus_health": spaces.Box(0, 1, shape=(2,2,), dtype=np.float32),
            }
        )

        # We have a total of 15 actions
        # 8 movement actions corresponding to "north", "south", "east", "west", and in between them.
        # 4 ability actions for q, w, e, r
        # two summoner spells actions d (ghost) and f (smite)
        # an action to base with b : recall(1), no_recall(0)
        # open shop and buy items with p
        self.action_space = spaces.MultiDiscrete(
            [
                9,  # movement: 0 -> no movement, ... etc
                3,  # q ability: 0 -> do nothing, 1 -> activate ability, 2-> upgrade
                3,  # w
                3,  # e
                3,  # r
                2,  # d
                2,  # f
                2,  # b
            ]
        )
    
    def _find_window(self, title):
        windows = gw.getWindowsWithTitle(title)
        if windows:
            return windows[0]
        else:
            return None
        
    def _activate_and_get_size(self, window):
        if window:
            window.activate()
            time.sleep(1)  # Wait for the window to become active
            x, y, w, h = window.top, window.left, window.width, window.height
            return x, y, w, h
        else:
            print("Window not found")
            return None, None
    
    def _capture_screen(self, x, y, w, h):
        screenshot = pyautogui.screenshot(region=(x,y,w,h))
        screenshot_np = np.array(screenshot)
        return screenshot_np
    
    def _locate_area(self, screenshot, template):
        # Convert to grayscale
        gray = cv.cvtColor(screenshot, cv.COLOR_BGR2GRAY)
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        w, h = template.shape[::-1]

        result = cv.matchTemplate(gray, template, cv.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        # Draw a rectangle around the matched area
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        return top_left[0] + (w / 2), top_left[1] + (h / 2)
    
    def _get_player_data(self):
            url = "https://127.0.0.1:2999/liveclientdata/allgamedata"
            response = requests.get(url, verify=False)
            return response
    
    def _get_obs(self):
        return {
            "champion_position": self._champion_position,
            "champion_health" : self._champion_health,
            "champion_mana" : self._champion_mana,
            "champion_level" : self._champion_level,
            "gold" : self._gold,
            "time" : self._time,
            "side" : self._side,
            "abilities_cooldown": self._abilities_cooldown,
            "abilities_level": self._abilities_level,
            "spells_cooldown": self._spells_cooldown,
            "jungle_monsters_position": self._jungle_monsters_position,
            "jungle_monsters_health": self._jungle_monsters_health,
            "jungle_monsters_timer": self._jungle_monsters_timer,
        }

    def _get_info(self):
        return {
            "champion_max_health": self._champion_max_health,
            "champion_max_mana" : self._champion_max_mana,
            "jungle_monsters_distances": np.linalg.norm(self._jungle_monsters_position - self._champion_position, axis=1)
        }

    def reset(self, seed=None, options=None):

        self.epoch_number += 1

        if self.epoch_number > 1 :
            # Reset the game
            print("Reseting game ...")

            while(True) :
                try :
                    response = self._get_player_data()
                    if response.status_code == 200:
                        print("Game still running")

                        isEndGame = False
                        for event in response.json()["events"]["Events"] :
                            if event["EventName"] == "GameEnd" : isEndGame = True

                        if not isEndGame and self._time > 900:
                            pydirectinput.press("enter")
                            pydirectinput.write("/ff")
                            pydirectinput.press("enter")
                        
                except requests.RequestException as e:
                    print("Game ended")
                    break

                print(f"Retrying in {2} seconds...")
                time.sleep(4)
            
            time.sleep(10)

            # press continue
            screen_capture = self._capture_screen(0, 0, 1920, 1080)
            play_template = cv.imread("assets\play_again.jpg")
            x, y = self._locate_area(screen_capture, play_template)
            pydirectinput.click(x, y, button="primary")
            time.sleep(4)



        print("Starting the game ...")
        ## wait till game exits
        time.sleep(4)
        ## switch to client
        window = self._find_window("League of Legends")
        self._activate_and_get_size(window)

        ## play
        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\play_btn.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x, y, button="primary")
        time.sleep(4)

        ## create custom
        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\\create_custom.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x, y, button="primary")
        time.sleep(4)

        ## confirm
        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\confirm.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x, y, button="primary")
        time.sleep(4)

        ## add first enemy bot
        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\\add_first_bot_enemy.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x+100, y, button="primary")
        time.sleep(4)

        ## change difficulty to intermidiat (helps ending game faster)
        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\\difficulty.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x, y, button="primary")
        time.sleep(4)

        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\\intermid_diff.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x, y, button="primary")
        time.sleep(4)

        ## start game
        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\start.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x, y, button="primary")
        time.sleep(4)

        ## hover master yi
        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\select_yi.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x, y, button="primary")
        time.sleep(4)

        ## lock in
        screen_capture = self._capture_screen(0, 0, 1920, 1080)
        play_template = cv.imread("assets\lock_in.jpg")
        x, y = self._locate_area(screen_capture, play_template)
        pydirectinput.click(x, y, button="primary")

        ## wait till game start

        time.sleep(14)

        while (gw.getActiveWindowTitle() != "League of Legends (TM) Client") :
            window = self._find_window("League of Legends (TM) Client")
            print("Waiting for game to start ...")
            time.sleep(4)

        print("Waiting for game to load ...")
        time.sleep(10)

        data = None

        while(True) :
            try :
                response = self._get_player_data()
                if response.status_code == 200:
                    print("Received a valid response")
                    data = response.json()
                    break
            except requests.RequestException as e:
                print(f"Request failed: {e}")

            print(f"Retrying in {2} seconds...")
            time.sleep(4)

        time.sleep(1)

        # Reset observations
        print("Reseting observations ...")

        ## collecting data
        currentHealth = data['activePlayer']['championStats']['currentHealth']
        currentMana = data['activePlayer']['championStats']['resourceValue']
        currentLevel = data['activePlayer']['level']
        currentGold = data['activePlayer']['currentGold']
        currentTime = data['gameData']['gameTime']

        self._champion_max_health = data['activePlayer']['championStats']['maxHealth']
        self._champion_max_mana = data['activePlayer']['championStats']['resourceMax']

        ## We need the following line to seed self.np_random
        super().reset(seed=seed)

        ## champion position
        x, y, w, h = MINIMAP_AREA
        minimap = self._capture_screen(x, y, w, h)
        player = cv.imread("assets\yi_border.jpg")
        x, y = self._locate_area(minimap, player)
        self._champion_position = np.array([x,y], dtype=np.float32)
        
        ## champions health
        self._champion_health = currentHealth

        ## champions mana
        self._champion_mana = currentMana

        ## champions level
        self._champion_level = currentLevel

        ## gold
        self._gold = currentGold

        ## time
        self._time = currentTime

        ## side
        currPlayerIndex = 0
        for playerInd in range(len(data['allPlayers'])) :
            if data['allPlayers'][playerInd]["riotId"] == data['activePlayer']['riotId']:
                currPlayerIndex = playerInd
        self._side = 0 if data['allPlayers'][currPlayerIndex]['team'] == "ORDER" else 1

        ## abilities cooldown
        self._abilities_cooldown = np.array([0, 0, 0, 0], dtype=np.float32)

        ## abilities level
        self._abilities_level = np.array([0, 0, 0, 0], dtype=np.int32)

        ## spells cooldown
        self._spells_cooldown = np.array([0, 0], dtype=np.float32)

        ## jungle monsters position
        if self._side == 0:
            self._jungle_monsters_position = np.array([
                [217, 316], # krugs
                [198, 277], # red
                [182, 245], # chickens
                [96, 214],  # wolves
                [96, 177],  # blue
                [55, 166],  # gromp
                [111, 132], # scuttle top
                [125, 112], # grubs, herald, baron
                [275, 244], # scuttle bot
                [257, 268], # dragon
            ], dtype=np.float32)
        else :
            self._jungle_monsters_position = np.array([
                [168, 63], # krugs
                [184, 99], # red
                [202, 135], # chickens
                [287, 167],  # wolves
                [288, 206],  # blue
                [228, 217],  # gromp
                [111, 132], # scuttle top
                [125, 112], # grubs, herald, baron
                [275, 244], # scuttle bot
                [257, 268], # dragon
            ], dtype=np.float32)


        ## jungle monsters health
        self._jungle_monsters_health = np.zeros((10,), dtype=np.float32)

        ## jungle monsters timer in seconds
        self._jungle_monsters_timer = np.array([
            100 , # krugs
            90 , # red
            90 , # chickens
            90 , # wolves
            90 , # blue
            100 , # gromp
            210 , # scuttle top
            300 , # grubs
            210 , # scuttle bot
            300 , # dragon
        ], dtype=np.float32)

        ## towers position
        ## towers health
        ## inhib position
        ## inhib health
        ## nexus position
        ## nexus health


        observation = self._get_obs()
        info = self._get_info()

        self.distanceToRed = 380.0

        return observation, info
    
    def _move_champion(self, direction):

        reward = 0

        if( direction != 0 ):
            reward = -1
            steps = 8
            angle_step = 2 * math.pi / steps
            center_x = 870
            center_y = 510
            radius = 150

            step = steps - direction

            angle = step * angle_step
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            pyautogui.click(x, y, button="SECONDARY")

        return reward

    def step(self, action):

        interval = 1.0 / 4 # Calculate the interval in seconds
        start_time = time.time()  # Get the current time at the start of the loop
        

        reward = 0
        direction, q, w, e, r, d, f, b = action
        data = self._get_player_data().json()

        # Move the champion
        reward += self._move_champion(direction)

        # Update champion position
        xm, ym, wm, hm = MINIMAP_AREA
        minimap = self._capture_screen(xm, ym, wm, hm)
        player = cv.imread("assets\yi_border_3.jpg")
        xc, yc = self._locate_area(minimap, player)
        self._champion_position = np.array([xc,yc], dtype=np.float32)

        # Update champion health and mana level
        self._champion_health = data['activePlayer']['championStats']['currentHealth']
        self._champion_mana = data['activePlayer']['championStats']['resourceValue']
        self._champion_level = data['activePlayer']['level']

        # Other update
        self._time = data['gameData']['gameTime']
        self._gold = data['activePlayer']['currentGold']
        # Death
        if self._champion_health == 0 : reward -= 100

        # Abilities
        match q:
            case 1:
                pydirectinput.press("q")
                reward -= 1
            case 2:
                pastLevel = data["activePlayer"]["abilities"]["Q"]["abilityLevel"]
                with pydirectinput.hold("ctrlleft") : pydirectinput.press("q")
                if data["activePlayer"]["abilities"]["Q"]["abilityLevel"] > pastLevel :
                    reward += 10
                else :
                    reward -= 1

        match w:
            case 1:
                pydirectinput.press("w")
                reward -= 1
            case 2:
                pastLevel = data["activePlayer"]["abilities"]["W"]["abilityLevel"]
                with pydirectinput.hold("ctrlleft") : pydirectinput.press("w")
                if data["activePlayer"]["abilities"]["W"]["abilityLevel"] > pastLevel :
                    reward += 10
                else :
                    reward -= 1

        match e:
            case 1:
                pydirectinput.press("e")
                reward -= 1
            case 2:
                pastLevel = data["activePlayer"]["abilities"]["E"]["abilityLevel"]
                with pydirectinput.hold("ctrlleft") : pydirectinput.press("e")
                if data["activePlayer"]["abilities"]["E"]["abilityLevel"] > pastLevel :
                    reward += 10
                else :
                    reward -= 1
        
        match r:
            case 1:
                pydirectinput.press("r")
                reward -= 1
            case 2:
                pastLevel = data["activePlayer"]["abilities"]["R"]["abilityLevel"]
                with pydirectinput.hold("ctrlleft") : pydirectinput.press("r")
                if data["activePlayer"]["abilities"]["R"]["abilityLevel"] > pastLevel :
                    reward += 10
                else :
                    reward -= 1

        # Spells
        ## flash, ghost ...
        if(d == 1):
                reward -= 50
                pydirectinput.press("d")

        ## smite
        if(f == 1):
                reward -= 1
                pydirectinput.press("f")

        # Basing
        # if(b == 1):
        #         pydirectinput.press("b")

        #         # check if player is in base
        #         x1, y1, x2, y2 =  2, 347, 31, 380 # base area from topleft to bottomright
        #         x, y = self._champion_position
                
        #         if self._champion_health < self._champion_max_health * 0.3 or self._champion_mana < self._champion_max_mana * 0.1:
        #             reward += 1
        #         else:
        #             reward -= 1

        #         if (x > x1 and x < x2 and y < y1 and y > y2) :
        #             reward += 10
        #         else :
        #             reward -= 1


        # abilities = [q, w, e, r]
        # for i, ability in enumerate(abilities):
        #     if ability == 1:  # Activate ability
        #         pydirectinput.press()
        #         self._abilities_cooldown[i] = 1.0  # Assume some cooldown value
        #     elif ability == 2:  # Upgrade ability
        #         self._abilities_level[i] = min(self._abilities_level[i] + 1, 5)  # Max level 5

        # Distance to jungle monsters : Red
        distances = np.linalg.norm(self._jungle_monsters_position - self._champion_position, axis=1)

        if distances[1] < self.distanceToRed :
            reward += 1
        else :
            reward -= 1

        self.distanceToRed = distances[1]

        # An episode is done if the game is done
        terminated = False
        for event in data["events"]["Events"] :
            if event["EventName"] == "GameEnd" :
                terminated = True
                reward += 1000 if (event["Result"] == "Win") else -1000  # reward for winning / losing the game
        
        # Update observation and info
        observation = self._get_obs()
        info = self._get_info()

        elapsed_time = time.time() - start_time # Calculate the elapsed time
        sleep_time = max(0, interval - elapsed_time)
        time.sleep(sleep_time)  # Sleep for the remaining time to maintain the loop frequ

        print("Action: ", action, ", Reward: ", reward, "Pos: ", self._champion_position, "Health: ", self._champion_health)

        return observation, reward, terminated, False, info

    def render(self):

        pass

