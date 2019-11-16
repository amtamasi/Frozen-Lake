import gym

from rmax import RMaxAgent
from q_learning import QLearningAgent
from sarsa import SarsaAgent

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
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
num_episodes = 10000
'''
rmax = RMaxAgent(env)
rmax.trainAgent(num_episodes)
print("RMax Agent Reward", np.mean(rmax.avg_cumulative_reward))
plt.plot(rmax.episodes_tested, rmax.avg_cumulative_reward, '.b-', markersize=1)
plt.title("Comparing agents mean reward over time")
plt.xlabel("Number of Episodes")
plt.ylabel("Mean Reward")
plt.legend(['Q-Learning Agent', 'SARSA Agent'])
plt.show()
'''


#Q-Learning
qlearner = QLearningAgent(env)
qlearner.trainAgent(num_episodes)
print("Mean Q Learner Reward", np.mean(qlearner.avg_cumulative_reward))
final_mean = qlearner.test(action_limit=100, render=True, num_test_episodes=10)
print("Q Learner final mean reward", final_mean)
print("Q Table", qlearner.Q)


#SARSA
sarsa = SarsaAgent(env)
sarsa.trainAgent(num_episodes)
print("Mean SARSA Reward", np.mean(sarsa.avg_cumulative_reward))
final_mean = sarsa.test(action_limit=100, render=False, num_test_episodes=10)
print("SARSA final mean reward", final_mean)
print("Q Table", sarsa.Q)



plt.plot(qlearner.episodes_tested, qlearner.avg_cumulative_reward, '.r-', markersize=1)
plt.plot(sarsa.episodes_tested, sarsa.avg_cumulative_reward, '.b--', markersize=1)


plt.title("Comparing agents mean reward over time")
plt.xlabel("Number of Episodes")
plt.ylabel("Mean Reward")
plt.legend(['Q-Learning Agent', 'SARSA Agent'])
plt.show()



env.close()


