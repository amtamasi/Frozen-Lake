from collections import defaultdict
import numpy as np
import random

class RMaxAgent():
    '''
    Our model-based Reinforcement Learning agent based on the RMax algorithm
    '''

    def __init__(self, env, rmax=1.0, sa_threshold=10, gamma=1.0):
        self.env = env #Environment we are training our agent with
        self.rmax = rmax
        self.gamma = gamma

        self.known_policy = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.R = np.zeros(self.env.observation_space.n + 1) # Map from state s to the reward received from state s (i.e. R(s)). Start as RMax
        self.R[16] = self.rmax #Last value represents reward for s0
        self.transitions = self.init_transitions() #trans[s][a][s'] = probability
        self.N_sas = np.zeros((self.env.observation_space.n, self.env.action_space.n, self.env.observation_space.n)) # Number of times you have seen (s, a, s')
        self.N_sa = np.zeros((self.env.observation_space.n, self.env.action_space.n)) # Number of times you have seen (s, a)
        self.sa_threshold = sa_threshold #If the number of times we have seen this (s, a) exceeds this threshold, mark this (s, a) as 'known'
        self.is_known = defaultdict(bool) #Keep track of which (state, action) are known
        self.new_known = [] #Keep track of which (state, action) have been newly found as 'known' this episode
        self.newKnown = False #Were there any newly known

        self.avg_cumulative_reward = [] #List of the average reward (averaged over x tests) for each episode of learning
        self.episodes_tested = []

    def init_transitions(self):
        transitions = np.zeros((self.env.observation_space.n+1, self.env.action_space.n, self.env.observation_space.n+1))
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                transitions[state][action][16] = 1 #Every state-action pair transitions to an added s0.

        return transitions
            

    def trainAgent(self, num_episodes, action_limit=100):
        '''
        Incorporate trade-off exploit/explore with GLIE (Greedy in the limit of infinite exploration) 
        action selection policy.
        Parameters:
            num_episodes (int): Number of games the agent can play
            action_limit (int): Limit for the number of actions the agent can take in one game
        '''
        observation = self.env.reset() #Initial observation

        for episode in range(num_episodes):
            #Play this number of games
            #print("NEW GAME!")
            self.env.render() #Render the start state

            #Do planning by computing your policy if first episode OR we have new known transitions
            if(episode == 0 or self.newKnown):
                self.known_policy = self.valueIteration()
                #print("Policy", self.known_policy)
                self.newKnown = False #This should be false now for the new episode

            for _ in range(action_limit):
                #Do actions until the end of this single game

                prev_observation = observation
            
                action = self.chooseAction(prev_observation)
                observation, reward, done, info = self.env.step(action) #Observation is the state after taking action
                self.update(prev_observation, action, observation, reward) #Update values for N(s, a, s'), N(s, a), R(s)
                self.env.render()

                if(done):
                    #print("REWARD FROM THIS RUN:", reward)
                    self.avg_cumulative_reward.append(reward)
                    self.episodes_tested.append(episode)
                    break
            
            #End of episode
            if(len(self.new_known) > 0):
                #If we have new known stuff, update the known MDP
                self.newKnown = True
                for state, action in self.new_known:
                        for next_state in range(self.env.observation_space.n):
                            self.transitions[state][action][next_state] = self.N_sas[state][action][next_state] / self.N_sa[state][action]
                #self.updateMDP()
            else:
                self.newKnown = False

            observation = self.env.reset() #Reset the environment for the next episode
            self.new_known = []

    '''
    def updateMDP(self):
        for known_transition in self.new_known:
            self.known_policy[known_transition] = 
    '''


    def chooseAction(self, state):
        '''
        Chose an action based on the state the game is currently in
        Parameters:
            state (State): An observation returned from env.step() or env.reset()
        '''

        return np.argmax(self.known_policy[state])

    def update(self, state, action, next_state, reward):
        '''
        #Update values for N(s, a, s'), N(s, a), R(s)
        '''
        self.N_sas[state][action][next_state] += 1
        self.N_sa[state][action] += 1
        self.R[state] = reward

        if(self.N_sa[state][action] == self.sa_threshold):
            #This state-action pair is now "known"
            self.is_known[(state, action)] = True
            self.new_known.append( (state, action) )
            self.newKnown = True

    
    def lookahead(self, state, utilities):
        ''' Helper for value iteration
        Calculate the vector of values of each action for a given state.
        aka calculates the value for each action in a state.
        '''
        actions = np.zeros(self.env.action_space.n)
        for next_state in range(self.env.observation_space.n):
            for action in actions:
                #HOW TO DO THIS UPDATE?
                action = int(action)
                '''
                print("Action:", action)
                print("State:", state)
                print("NEXT state", next_state)
                print("R", self.R[state])
                print(utilities[next_state])
                print("Transition prob", self.transitions[state][action][next_state])
                '''
                actions[action] += self.R[state] + (self.gamma * utilities[next_state] * self.transitions[state][action][next_state])

        return np.max(actions), np.argmax(actions)

    def valueIteration(self, epsilon=0.0001, gamma=1.0, num_iterations=10000):
        '''
        Run value iteration
        Parameters:
            epsilon (float): Stop once our updates for every state are less than epsilon
            gamma (float): The discount factor we apply to future rewards in our update
        Returns:
            known_policy (Dict): Policy mapping from each state to the best action
        '''      
        u1 = np.zeros(self.env.observation_space.n + 1) # u[state][action] = estimate
        policy = np.zeros(self.env.observation_space.n)
        delta = 10 #Arbitrary initial value
        counter = 0

        while delta > epsilon and counter < num_iterations:
            u0 = np.copy(u1)

            for state in range(self.env.observation_space.n):
                best_action_value, best_action = self.lookahead(state=state, utilities=u0)
                u1[state] = best_action_value
                policy[state] = best_action

            delta = np.max(np.abs(u1 - u0))
            counter += 1
            if(delta < epsilon):
                break
        
        return policy

