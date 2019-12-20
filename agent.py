import numpy as np
import pandas as pd
import random


class Agent:
    ### START CODE HERE ###

    def __init__(self, actions):
        self.actions = actions
        self.epsilon = 0.4
        self.epsilon_min = 0.02
        self.epsilon_decre = self.epsilon_min/5


    def choose_action(self, observation):
        pos = observation[0]
        q_table = observation[1]
        epi = observation[2]
        if epi>100:
            self.epsilon-=self.epsilon_decre
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            move = []
            for act in self.actions:
                if q_table[pos][act]==max(q_table[pos]):
                    move.append(act)
            action = np.random.choice(move)
        if (pos[0]==0 and action==3)or(pos[0]==5 and action==2)or(pos[1]==0 and action==0)or(pos[1]==5 and action==1):
            q_table[pos][action] = -9999
            return self.choose_action(observation)
        return action

    ### END CODE HERE ###