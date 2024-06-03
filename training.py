from LeagueAgentEnv import LeagueAgentEnv

env = LeagueAgentEnv()
observation, info = env.reset()
print(observation)
print(info)