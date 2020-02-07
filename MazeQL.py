import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# The matrix of maze
maze = np.array(
    [[0, 0, 0, 0, 0, 0, ],
     [1, 0, 1, 1, 1, 1, ],
     [1, 0, 1, 0, 0, 0, ],
     [1, 0, 0, 0, 1, 1, ],
     [0, 1, 0, 0, 0, 0, ]]
)
print(pd.DataFrame(maze))

# Starting point
start_state = (0, 0)
# Ending point
target_state = (2, 5)

# Parameters of Q learning
alpha, gamma = .01, .9
# Reward
rewards = [-1, -.1, 10]
# Actions
actions = ['up', 'down', 'left', 'right']
# Q table
q_table = pd.DataFrame(columns=actions)

def get_next_state_reward(current_state, action):
    done = False
    if action == 'up':
        next_state = (current_state[0] - 1, current_state[1])
    elif action == 'down':
        next_state = (current_state[0] + 1, current_state[1])
    elif action == 'left':
        next_state = (current_state[0], current_state[1] - 1)
    else:
        next_state = (current_state[0], current_state[1] + 1)
    if next_state[0] < 0 or next_state[0] >= maze.shape[0] or next_state[1] < 0 or next_state[1] >= maze.shape[1] \
            or maze[next_state[0], next_state[1]] == 1:
        next_state = current_state
        reward = rewards[0]
    elif next_state == target_state:
        reward = rewards[2]
        done = True
    else:
        reward = rewards[1]
    return next_state, reward, done

def learn(current_state, action, reward, next_state):
    check_state_exist(next_state)
    q_sa = q_table.loc[current_state, action]
    max_next_q_sa = q_table.loc[next_state, :].max()
    #Q learning equation
    new_q_sa = (1 - alpha) * q_sa + alpha*(reward + gamma*max_next_q_sa)
    #Renew Q table
    q_table.loc[current_state, action] = new_q_sa

def check_state_exist(state):
    if state not in q_table.index:
        q_table.loc[state] = pd.Series(np.zeros(len(actions)), index=actions)

def choose_action(state, random_num=.9):
    series = pd.Series(q_table.loc[state])

    if random.random() > random_num:
        action = random.choice(actions)
    else:
        ss = shuffle(series)
        action = ss.idxmax()
    return action
# Max steps
max_steps = 1000

# Training the model
total_steps = []
for _ in range(max_steps):
    current_state = start_state
    step = 0
    while True:
        step += 1
        check_state_exist(str(current_state))
        action = choose_action(str(current_state))
        next_state, reward, done = get_next_state_reward(current_state, action)
        learn(str(current_state), action, reward, str(next_state))
        if done:
            total_steps.append(step)
            break
        current_state = next_state
print(q_table)
plt.plot(total_steps)
plt.xlabel('Learning process')
plt.ylabel('Total steps')
plt.show()

# Prediction
current_state = start_state
print('start_state:{}'.format(start_state))
step = 0
while True:
    step += 1
    check_state_exist(str(current_state))
    action = choose_action(str(current_state), random_num=1)
    next_state, reward, done = get_next_state_reward(current_state, action)
    print('step:{step}, action: {action}, state: {state}'.format(step=step, action=action, state=next_state))
    if done or step > 100:
        if next_state == target_state:
            print('success')
        else:
            print('fail')
        break

    else:
        current_state = next_state