import gym

from agents.rmax import RMaxAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from collections import defaultdict

""" SAMPLE OF HOW TO USE OPENAI GYM
env = gym.make('FrozenLake-v0')
env.reset()
env.render()
for _ in range(1000):
    
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    if(done):
        break

env.close()
"""


env = gym.make('FrozenLake-v0')
num_episodes = 2

rmax = RMaxAgent(env)
rmax.trainAgent(num_episodes)
print("RMax Agent Reward", rmax.avg_cumulative_reward)

