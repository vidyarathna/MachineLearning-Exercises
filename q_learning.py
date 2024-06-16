import numpy as np

# Define the environment (example: Gridworld)
env = np.array([
    [-1, -1, -1,  1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1]
])

# Q-learning parameters
gamma = 0.8  # Discount factor
lr = 0.8     # Learning rate
num_episodes = 1000

# Q-table initialization
Q = np.zeros_like(env)

# Q-learning algorithm
for episode in range(num_episodes):
    state = np.random.randint(0, env.shape[0])  # Start state (random)
    while True:
        action = np.random.choice(np.flatnonzero(env[state] >= 0))  # Choose a valid action
        next_state = action
        reward = env[state, action]

        # Update Q-table
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state
        if env[state, action] == 1:  # Terminal state
            break

print("Q-table:")
print(Q) 
