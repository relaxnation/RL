'''
part 2
'''
#You will implement an algorithm to iteratively calculate the optimal policies and the value of the game for each player in all three games.
import numpy as np


#constructor 
def initial_policy(val):
    initial_pol = []
    initial_pol.append(val1)
    return initial_pol 

#reward matrix constructor
def reward():
    reward_matrix = []
    return reward_matrix

def every_visit_update():
    #start w initial policy 
    #player 1 at time step 0
    p1 = np.array([0.5, 0.5])
    r1 = np.array([[5, 0],[10,1]])
    p2 = np.array([0.5, 0.5])
    p1T= p1.transpose()
    # calculate reward
    alpha = 0.001
    #p1 cooperate
    pjc_k1 = past_pjc + (alpha * reward * (1 - past_pjc))
    pjo_k1 = past_pjo - (alpha * reward *past_pjo)
    
    #reward =p1T * r1 * p2

    # update 
    # #if [if action c is taken at time t]

    #elif [for all other actions 0 != c]
#create graphs
#def graph_policies():

#simulate game play
def main():
    step = 50000
    every_visit_update()
    #prisoners dilemma 


if __name__ == "__main__":
    main()