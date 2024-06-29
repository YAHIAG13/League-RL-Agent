import time

from LeagueAgentEnv import LeagueAgentEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = LeagueAgentEnv()

model = PPO("MultiInputPolicy", env, verbose=2)
model.learn(total_timesteps=160)
model.save("LeagueAgentEnv")



# states = env.observation_space.shape
# actions = env.action_space.shape

# print(states, actions)




# for episode in range(1, episodes+1) :
#     observation, info = env.reset()
#     done = False
#     score = 0

#     while not done:

#         interval = 1.0 / fps # Calculate the interval in seconds
#         start_time = time.time()  # Get the current time at the start of the loop
        
#         action = env.action_space.sample()
#         observation, reward, terminated, done, info = env.step(action)
#         score += reward
        
        
#         elapsed_time = time.time() - start_time # Calculate the elapsed time
#         sleep_time = max(0, interval - elapsed_time)
#         time.sleep(sleep_time)  # Sleep for the remaining time to maintain the loop frequency

#     print(f'Episode: {episode}, Score: {score}')
        
    



