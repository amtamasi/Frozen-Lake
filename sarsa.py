from collections import defaultdict
import numpy as np
import random

class SarsaAgent():
    '''
    Our model-free Reinforcement Learning agent based on the tabular SARSA algorithm
    '''
    def __init__(self, env, gamma=0.9, alpha=0.8, epsilon=0.1):
        #self.Q = defaultdict(lambda: np.zeros(env.action_space.n)) # Map from state -> action -> value
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.avg_cumulative_reward = []
        self.episodes_tested = []

    
    def trainAgent(self, num_episodes, action_limit=200, test_every_n_episodes=1, test=True):

        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.selectAction(state)

            for _ in range(action_limit):
                
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.selectAction(next_state)
                self.update(state, action, next_state, reward, next_action)

                state = next_state
                action = next_action
                if(done):
                    break
            
            if(test and (episode % test_every_n_episodes == 0)):
                mean_tested_reward = self.test(action_limit=action_limit)
                self.avg_cumulative_reward.append(mean_tested_reward)
                self.episodes_tested.append(episode)
            
        

    def selectAction(self, state):
        #Select actions according to Epsilon Greedy policy, for training
        random_num = random.random()
        if(random_num <= self.epsilon):
            return self.env.action_space.sample()
        else:
            return self.selectGreedyAction(state)

    def selectGreedyAction(self, state):
        #Strictly follow the policy, for testing
        return np.argmax(self.Q[state])

    def update(self, state, action, next_state, reward, next_action):
        #Update the Q table based on the observations had from this time step.

        next_best_action = self.selectGreedyAction(next_state)
        target_val = reward + self.gamma*self.Q[next_state][next_best_action]
        estimate = self.Q[state][action]
        difference = target_val - estimate
        self.Q[state][action] += self.alpha*difference

    def test(self, action_limit, num_test_episodes=10, render=False):
        #Test our current policy
        rewards = []
        self.env.reset()

        for episode in range(num_test_episodes):
            state = self.env.reset()
            if(render):
                print()
                print("NEW GAME")

            for _ in range(action_limit):
                
                action = self.selectGreedyAction(state)
                next_state, reward, done, _ = self.env.step(action)
                
                if(render):
                    self.env.render()

                state = next_state
                if(done):
                    break
            
            rewards.append(reward)
            self.env.reset()

        self.env.reset()
        return np.mean(rewards)
