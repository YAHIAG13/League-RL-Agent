import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class LeagueAgentEnv(gym.Env):

    def __init__(self, size, render_mode=None):
        
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "champions_positions": spaces.Box(0, size - 1, shape=(10,2), dtype=np.float32),
                "champions_health": spaces.Box(0, 1, shape=(10,), dtype=np.float32),
                "champions_mana": spaces.Box(0, 1, shape=(10,), dtype=np.float32),
                "champions_level": spaces.Box(0, 18, shape=(10,), dtype=np.int32),

                "jungle_monsters_position": spaces.Box(0, size - 1, shape=(10,2), dtype=np.float32),
                "jungle_monsters_health": spaces.Box(0, 1, shape=(10,), dtype=np.float32),
                "jungle_monsters_timer": spaces.Box(0, 1, shape=(10,), dtype=np.float32),

                "abilities_cooldown": spaces.Box(0, 1, shape=(4,), dtype=np.float32),
                "spells_cooldown": spaces.Box(0, 1, shape=(2,), dtype=np.float32),

                # player stats :
                "gold" : spaces.Discrete(),
                "time" : spaces.Discrete(),
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
                9,  # movement
                2,  # q
                2,  # w
                2,  # e
                2,  # r
                2,  # d
                2,  # f
                2,  # b
                2   # p
            ]
        )

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. -1 corresponds to "south", 1 to "north", while 0 coresspond to "none" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 0]),    # no movement
            1: np.array([1, 0]),    # north
            2: np.array([-1, 0]),   # south
            3: np.array([0, 1]),    # east
            4: np.array([0, -1]),   # west
            5: np.array([1, -1]),   # north-west
            6: np.array([1, 1]),    # north-east
            7: np.array([-1, -1]),  # south-west
            8: np.array([-1, 1]),   # south-east
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        
        pass

