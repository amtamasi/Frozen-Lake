from collections import defaultdict
import numpy as np

class RMaxAgent():
    '''
    Our model-based Reinforcement Learning agent based on the RMax algorithm
    '''

    def __init__(self, env, rmax=1.0, sa_threshold=2):
        self.env = env #Environment we are training our agent with
        self.rmax = rmax

        self.known_policy = self.init_policy()
        self.real_policy = dict()
        self.R = defaultdict(lambda: self.rmax) # Dictionary mapping from state s to the reward received from state s (i.e. R(s)). Start as RMax
        self.N_sas = defaultdict(int) # Number of times you have seen (s, a, s')
        self.N_sa = defaultdict(int) # Number of times you have seen (s, a)
        self.sa_threshold = sa_threshold #If the number of times we have seen this (s, a) exceeds this threshold, mark this (s, a) as 'known'
        self.is_known = defaultdict(bool) #Keep track of which (state, action) are known
        self.new_known = [] #Keep track of which (state, action) have been newly found as 'known' this episode

        self.avg_cumulative_reward = [] #List of the average reward (averaged over x tests) for each episode of learning


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
            print("NEW GAME!")
            #self.env.render() #Render the start state

            #Do planning by computing your policy if this isn't the first round
            if(episode > 0):
                self.known_policy, v = self.valueIteration()

            for _ in range(action_limit):
                #Do actions until the end of this single game

                prev_observation = observation
            
                action = self.chooseAction(observation)
                observation, reward, done, info = self.env.step(action) #Observation is the state after taking action
                self.update(prev_observation, action, observation, reward) #Update values for N(s, a, s'), N(s, a), R(s)
                #self.env.render()

                if(done):
                    print("REWARD FROM THIS RUN:", reward)
                    self.avg_cumulative_reward.append(reward)
                    break

            observation = self.env.reset() #Reset the environment for the next episode
            self.new_known = []

        self.env.close()

    def init_policy(self):
        policy = defaultdict(lambda: random.choice(self.env.))

        return policy



    def chooseAction(self, state):
        '''
        Chose an action based on the state the game is currently in
        Parameters:
            state (State): An observation returned from env.step() or env.reset()
        '''

        return np.argmax(self.known_policy[state])

    def update(self, prev_observation, action, observation, reward):
        '''
        #Update values for N(s, a, s'), N(s, a), R(s)
        '''
        self.N_sas[(prev_observation, action, observation)] += 1
        self.N_sa[(observation, action)] += 1
        self.R[observation] = reward 

        if(self.N_sa[(observation, action)] >= self.sa_threshold):
            #This state-action pair is now "known"
            self.is_known[(observation, action)] = True
            self.new_known.append( (observation, action) )


    def valueIteration(self, epsilon=0.0001, gamma=1.0):
        '''
        Run value iteration
        Parameters:
            epsilon (float): Stop once our updates for every state are less than epsilon
            gamma (float): The discount factor we apply to future rewards in our update
        Returns:
            known_policy (Dict): Policy mapping from each state to the best action
        '''
        def one_step_lookahead(state, V):
            '''
            Helper function to calculate the value for all action in a given state.
            
            Args:
                state: The state to consider (int)
                V: The value to use as an estimator, Vector of length env.nS
            
            Returns:
                A vector of length env.nA containing the expected value of each action.
            '''
            A = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[state][a]:
                    A[a] += prob * (reward + gamma * V[next_state])
            return A
        
        V = np.zeros(self.env.nS)
        while True:
            # Stopping condition
            delta = 0
            # Update each state...
            for s in range(self.env.nS):
                # Do a one-step lookahead to find the best action
                A = one_step_lookahead(s, V)
                best_action_value = np.max(A)
                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function. Ref: Sutton book eq. 4.10. 
                V[s] = best_action_value        
            # Check if we can stop 
            if delta < epsilon:
                break
        
        # Create a deterministic policy using the optimal value function
        policy = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(s, V)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s, best_action] = 1.0
        
        return policy, V

