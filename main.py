from agent import Agent
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import random
maze = '1'

if maze == '1':
    from maze_env1 import Maze
elif maze == '2':
    from maze_env2 import Maze


if __name__ == "__main__":
    ### START CODE HERE ###
    # This is an agent with random policy. You can learn how to interact with the environment through the code below.
    # Then you can delete it and write your own code.

    env = Maze()
    agent = Agent(actions=list(range(env.n_actions)))

    #parameter
    episode_num = 500
    alpha = 0.7
    gamma = 0.9
    dyna_rate = 20
    living_cost = -0.085

    #initialize the q_table and the number states has been explored
    q_table={}
    visit_time = {}
    for k in [True,False]:
        for i in range(6):
            for j in range(6):
                pos=(i,j,k)
                q_table[pos] = [0,0,0,0]
                visit_time[pos] = [0,0,0,0]

    #initialization for dyna process
    buff = {}

    #learning
    for episode in range(episode_num):
        # print("tbale: ",q_table)
        # print("buff: ",buff)
        s = env.reset()
        pos = (int((s[0] - 5) / 40), int((s[1] - 5) / 40),False)

        episode_reward = 0
        step=0
        while True:
            # if episode >100:
            #     env.render()
            a = agent.choose_action((pos,q_table,episode))
            s_, r, done = env.step(a)
            new_pos = (int((s_[0] - 5) / 40), int((s_[1] - 5) / 40),s_[4])
            buff[(pos, a)] = [new_pos, r]
            step+=1
            #update
            tmp = visit_time.copy()
            for k, v in tmp.items():
                visit_time[k] = np.array(visit_time[k]) * 0.9
            visit_time[pos][a] += 1
            episode_reward += r
            #update q table
            Vq = q_table[pos][a]  #initial q value
            max_action = []
            for i in range(4):
                max_action.append(q_table[new_pos][i] - visit_time[new_pos][i]/(5/0.99))
            max_next_Vq = max(max_action)
            new_Vq = Vq + alpha * (r+living_cost + gamma * max_next_Vq - Vq)
            q_table[pos][a] = new_Vq
            pos=new_pos

            #dyna update
            dyna_list=[i for i in buff.keys()]
            for i in range(dyna_rate):
                dyna_state = random.choice(dyna_list)
                dyna_pos = dyna_state[0]
                dyna_action = dyna_state[1]
                dyna_new_pos=buff[dyna_state][0]
                dyna_reward = buff[dyna_state][1]
                dyna_max_table = []
                for i in range(4):
                    dyna_max_table.append(q_table[dyna_new_pos][i] - visit_time[dyna_new_pos][i] / 5)
                dyna_max_next_v = max(dyna_max_table)
                q_table[dyna_pos][dyna_action]=(1-alpha)*q_table[dyna_pos][dyna_action]+alpha*(dyna_reward+gamma*dyna_max_next_v)


            if done:
                # if episode>100:
                #     env.render()
                #     time.sleep(0.5)
                break

        print('episode:', episode, 'episode_reward:', episode_reward)


    ### END CODE HERE ###

    print('\ntraining over\n')
