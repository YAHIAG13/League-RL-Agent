# Reinforcement Learning Agent for Video Game Strategy Optimization
using the moba game League of Legends as an environment 

## Motivation:
motivated by OpenAI work on OpenFive, the Dota 2 Team of RL Agents that where able to beat the International champions. we are trying to build an agent that learns to play the famous MOBA game, League of Legends.\
[Read](https://arxiv.org/abs/1912.06680) OpenAi Papers about OpenFive AI System.

## Why a video game:
video games provides one of the most challenging environments for reinforcement learning agents, a real time environment where each split of a second matter and with so many variables to observe and countless actions that could be executed. it would be fascinating to see how these agents could adapt to the chalenges, and what long term strategies that they could come up with that for us humans seems illogical.

## Tools:
**Gymnasium [\[Link\]](https://gymnasium.farama.org/index.html):** an interface that helps representing RL problems.\
**Stable-Baselines3:** a set of reliable implementations of reinforcement learning algorithms in PyTorch.\
**pyautogui, pydirectinput, pygetwindow:** keyboard and mouse input from a python script.\
**League Live Client Data API [[Link]](https://developer.riotgames.com/docs/lol#game-client-api_live-client-data-api):** provides a method for gathering data during an active league of legends game. It includes general information about the game as well player data.

## Agent Environment:
League of legends uses "fog of war" that acts as the field of view of the agent. this fog of war prevents him for knowing something like enemy position, health or other things, making the environment state is not fully observable unlike chess for example. so instead, the agent receives observations that provide some information about the state of the environment like enemy position, health... when there is one inside the field of view, also some other informations about the agent itself such as gold, health, etc. The agent decision-making involves three key components:

-   **States**: Different observations of the environment, partially known to the agent.\
*example:* elapsed time, structures health and positions, creeps health and positions, agent's and allies (enemy when observable) health, mana, gold, position, etc.

-   **Actions**: Choices available to the agent that can affect the state.\
*example:* move in all directions, activate or upgrade ability, base, shop for items, etc.

-   **Rewards**: Feedback on the actions taken, guiding the agent towards its goal. \
*example:* receive a positive reward when low in health and then base to get health back, a negative reward when agent dies (health = 0), etc.

## Technicalities:
**RL Algorithm:** Proximal Policy Optimization (PPO). [Read](https://arxiv.org/abs/1707.06347) Original OpenAI Papers.\
**Policy:** MultiInputPolicy. works well with Gym Dictionary observation space. [Read More](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations).

**//TODO in depth explanation of the chosen observations , actions, and policies:**
