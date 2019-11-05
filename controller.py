import gym

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