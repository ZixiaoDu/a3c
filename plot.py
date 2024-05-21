import os
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def read_rewards_from_file(filename):
    rewards = []
    with open(filename, 'r') as file:
        for line in file:
            if "return" in line:
                return_value = float(line.split("return:")[1].strip())
                rewards.append(return_value)
    return rewards

def read_all_rewards(num_workers):
    all_rewards = {}
    for worker_id in range(num_workers):
        filename = os.path.join('./', f'output_{worker_id}.txt')
        if os.path.exists(filename):
            all_rewards[worker_id] = read_rewards_from_file(filename)
    return all_rewards

def calculate_average_rewards(all_rewards):
    max_length = max(len(rewards) for rewards in all_rewards.values())
    average_rewards = []
    for i in range(max_length):
        rewards_at_i = [rewards[i] for rewards in all_rewards.values() if i < len(rewards)]
        average_rewards.append(np.mean(rewards_at_i))
    return average_rewards

all_rewards = read_all_rewards(8)
average_rewards = calculate_average_rewards(all_rewards)
smoothed_rewards = moving_average(average_rewards, window_size=50)

plt.figure(figsize=(10, 5))
plt.plot(smoothed_rewards, label='Average Return')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Model Training')
plt.legend()
plt.grid()
plt.savefig("Results1.svg", dpi=300, format="svg")
plt.show()
