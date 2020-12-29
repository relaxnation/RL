'''
Course: CISC 453: Reinforcement Learning
Windy Grid World with King's moves and stochastic wind
Student Number: 20062694

'''

#resolving with the king's moves and stochastic wind 
import numpy as np
import random

reward = -1

# exploration rate
epsilon = 0.1

# step-size
alpha = 0.5

# environment dimensions
row, col = 7, 10

# start-state
start = (3,0)

# goal-state
goal = (3,7)


# for each action you have q_value, next state
class Action:
    def __init__(self, q_value, next_state):
        self.q_value = q_value
        #given a state and an action and wind, where would the 3 possible next states be
        self.next_state = next_state


def create_grid():
    # grid is a 3D matrix where it maps a 2D space representing all the states where each one has a list of Action objects
    grid = [ [[0]*4] * col for _ in range(row) ]
    for r in range(row):
        for c in range(col):
            actions = []
            initial_q = 0

            # setting the wind parameter
            wind = [0,0,0,1,1,1,2,2,1,0]
            # North
            actions.append(Action(initial_q, (max(r - 1 - wind[c], 0), c)))
            # East
            actions.append(Action(initial_q, (max(r - wind[c], 0),min(c + 1, row - 1))))
            # South
            actions.append(Action(initial_q, (max(min(r + 1 - wind[c], col - 1), 0), c)))
            # West
            actions.append(Action(initial_q, (max(r - wind[c], 0),max(c - 1, 0))))
            #Northeast
            actions.append(Action(initial_q, (max(r - 1 - wind[c], 0), min(c + 1, row - 1))))
            #Northwest
            actions.append(Action(initial_q, (max(r - 1 - wind[c], 0), max(c - 1, 0))))
            #Southeast
            actions.append(Action(initial_q, (max(min(r + 1 - wind[c], col - 1), 0), min(c + 1, row - 1))))
            #Southwest
            actions.append(Action(initial_q, (max(min(r + 1 - wind[c], col - 1), 0), max(c - 1, 0))))
            grid[r][c] = actions

    return grid


def sarsa(grid):
    max_episodes = 5000
    episode = 0

    # loop for each episode
    while episode < max_episodes:
        # initialize S
        state = start
        # extract policies from grid
        current_policy = [ action.q_value for action in grid[state[0]][state[1]] ]

        #epsilon-greedy
        if np.random.rand() <= epsilon:
            # explore
            action = np.random.randint(0,4)
        else:
            # exploit
            action = current_policy.index(max(current_policy))
        
        # loop for each step of episode
        while state != goal:
            # take action and move to next state
            #STOCHASTIC WIND
            stochastic_var = random.randint(0,2)
            
            # extract policies from grid
            current_policy = [ action.q_value for action in grid[next_state[0]][next_state[1]] ]

            # epsilon-greedy
            if np.random.rand() <= epsilon:
                # explore
                next_action = np.random.randint(0,4)
            else:
                # exploit
                next_action = current_policy.index(max(current_policy))

            # SARSA
            grid[state[0]][state[1]][action].q_value += \
                alpha * (reward + grid[next_state[0]][next_state[1]][next_action].q_value - 
                grid[state[0]][state[1]][action].q_value)
            
            state = next_state
            action = next_action

        episode += 1


def qlearning(grid):
    max_episodes = 5000
    episode = 0

    # loop for each episode
    while episode < max_episodes:
        # initialize S
        state = start
        
        # loop for each step of episode
        while state != goal:
            # extract policies from grid
            current_policy = [ action.q_value for action in grid[state[0]][state[1]] ]
            #epsilon-greedy
            if np.random.rand() <= epsilon:
                # explore
                action = np.random.randint(0,4)
            else:
                # exploit
                action = current_policy.index(max(current_policy))

            # take action and move to next state
            #STOCHASTIC WIND
            next_state = grid[state[0]][state[1]][action].next_state
            
            # get q_values from next state
            next_q = [ action.q_value for action in grid[next_state[0]][next_state[1]] ]
            # Q-Learning
            grid[state[0]][state[1]][action].q_value += \
                alpha * (reward + max(next_q) - 
                grid[state[0]][state[1]][action].q_value)
            
            state = next_state

        episode += 1


def print_policy(grid):
    policy = [[0] * col for _ in range(row)]
    actions = ['N','E','S','W','NE','NW','SE','SW']

    for r in range(row):
        for c in range(col):
            state_policy = [ action.q_value for action in grid[r][c] ]
            best_action = actions[state_policy.index(max(state_policy))]
            policy[r][c] = best_action
    
    print(policy)


def main():
    print('SARSA')
    grid = create_grid()
    sarsa(grid)
    print_policy(grid)

    print('\nQ-Learning')
    grid = create_grid()
    qlearning(grid)
    print_policy(grid)

if __name__ == "__main__":
    main()