import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import os
import sys

# 构建策略网络--actor
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


# 构建价值网络--critic
class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class bg:
    def __init__(self, user_info, movie_info, movie_type, ratings, n_users):
        self.user_info = user_info
        self.movie_info = movie_info
        self.movie_type = movie_type
        self.ratings = ratings
        self.n_users = n_users

    def get_random_state(self):
        userid = np.random.randint(1, self.n_users + 1)
        res = self.user_info[userid]
        res = torch.tensor(res, dtype=torch.float)
        layer_norm = nn.LayerNorm(4, eps=1e-6)
        res = layer_norm(res)
        return res.detach().numpy(), userid

    def get_state(self, userid):
        res = self.user_info[userid]
        res = torch.tensor(res, dtype=torch.float)
        layer_norm = nn.LayerNorm(4, eps=1e-6)
        res = layer_norm(res)
        return res.detach().numpy()

    def get_reward(self, userid, movieid):
        reward = 0.0
        for movie, score in self.ratings[userid].items():
            set1 = self.movie_info[movieid]
            set2 = self.movie_info[movie]
            sameset = set1.intersection(set2)
            similar_ratio = len(sameset) / (len(set1) + len(set2) - len(sameset))
            reward += score * similar_ratio
        return reward


class A3CWorker(mp.Process):
    def __init__(self, global_actor, global_critic, global_optimizer, gamma, lmbda, worker_id, device, user_info,
                 movie_info, movie_type, ratings, n_users, n_movies, index2movieid):
        super(A3CWorker, self).__init__()
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.global_optimizer = global_optimizer
        self.gamma = gamma
        self.lmbda = lmbda
        self.worker_id = worker_id
        self.device = device
        self.env = bg(user_info, movie_info, movie_type, ratings, n_users)
        self.n_movies = n_movies
        self.index2movieid = index2movieid
        self.local_actor = PolicyNet(global_actor.fc1.in_features, global_actor.fc1.out_features,
                                     global_actor.fc2.out_features).to(device)
        self.local_critic = ValueNet(global_critic.fc1.in_features, global_critic.fc1.out_features).to(device)
        self.sync_with_global()

    def sync_with_global(self):
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

    def top_10(self, state):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state).detach().numpy()[0].tolist()
        map = {}
        list = []
        index = 0
        for p in probs:
            smalllist = [p, index]
            list.append(smalllist)
            index = index + 1
        list.sort(key=lambda x: x[0], reverse=True)
        returnlist = []
        for i in range(10):
            returnlist.append(self.index2movieid[list[i][1] + 1])

        return returnlist

    def run(self):
        rewards_log = []
        num_episodes = 300
        for episode in range(num_episodes):
            state, userid = self.env.get_random_state()
            episode_return = 0.0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }
            for t in range(200):
                action = self.take_action(state, userid)
                reward = self.env.get_reward(userid, self.index2movieid[action + 1])
                next_state, next_userid = self.env.get_random_state()
                done = t == 199
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                userid = next_userid
                episode_return += reward
                if done:
                    break
            self.learn(transition_dict)
            rewards_log.append(episode_return)
            with open(f'output_{self.worker_id}.txt', 'a') as f1:
                f1.write(f"iter: {episode}, return: {episode_return}\n")
            print(f'worker:{self.worker_id}, iter:{episode}, return:{episode_return}')
            self.sync_with_global()

        with open(f'rewards_log_{self.worker_id}.txt', 'w') as f:
            for reward in rewards_log:
                f.write(f"{reward}\n")

    def take_action(self, state, userid):
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        probs = self.local_actor(state)
        canuse_action = [action for action in range(self.n_movies) if self.index2movieid[action + 1] not in self.env.ratings[userid].keys()]
        canuse_action_pro = [probs[0][action] for action in canuse_action]
        action = random.choices(canuse_action, weights=canuse_action_pro, k=1)[0]
        return action

    def learn(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)
        next_q_target = self.local_critic(next_states)
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        td_value = self.local_critic(states)
        td_delta = td_target - td_value
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)
        old_log_probs = torch.log(self.local_actor(states).gather(1, actions)).detach()
        log_probs = torch.log(self.local_actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = torch.mean(F.mse_loss(self.local_critic(states), td_target.detach()))
        self.global_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        for global_param, local_param in zip(self.global_actor.parameters(), self.local_actor.parameters()):
            global_param._grad = local_param.grad
        for global_param, local_param in zip(self.global_critic.parameters(), self.local_critic.parameters()):
            global_param._grad = local_param.grad
        self.global_optimizer.step()
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())


def recommend_for_user(user_id, model, env, n_movies, index2movieid, device, top_k=10):
    state = env.get_state(user_id)
    state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(device)
    probs = model(state)
    canuse_action = [action for action in range(n_movies) if index2movieid[action + 1] not in env.ratings[user_id].keys()]
    canuse_action_pro = [probs[0][action] for action in canuse_action]
    sorted_actions = sorted(zip(canuse_action, canuse_action_pro), key=lambda x: x[1], reverse=True)
    recommendations = [index2movieid[action + 1] for action, _ in sorted_actions[:top_k]]
    return recommendations


def hits_at_10(recommended_list, actual_items):
    hits = 0
    for item in actual_items:
        if item in recommended_list:
            hits += 1
    return hits / len(actual_items)


def evaluate_recommendations(model, env, n_movies, index2movieid, device, num_users=10, top_k=10):
    hits = []
    for _ in range(num_users):
        user_id = random.choice(env.user_info.keys())
        recommendations = recommend_for_user(user_id, model, env, n_movies, index2movieid, device, top_k=top_k)
        actual_items = env.ratings[user_id].keys()
        if actual_items:
            hits.append(hits_at_10(recommendations, actual_items))
    return np.mean(hits)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_episodes = 300
    gamma = 0.9
    actor_lr = 1e-3
    critic_lr = 1e-3
    n_hiddens = 16
    user_info = {}
    user_path = ".\\data\\ml-1m\\users.txt"
    with open(user_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('::')
            if len(line) != 5:
                continue
            list = [0 if line[1] == 'F' else 1, int(line[2]), int(line[3]), int(line[4].split('-')[0])]
            user_info[int(line[0])] = list
    index2movieid = {}
    movie_info = {}
    movie_type = set()
    movie_path = ".\\data\\ml-1m\\movies.txt"
    with open(movie_path, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('::')
            if len(line) != 3:
                continue
            index2movieid[len(index2movieid) + 1] = int(line[0])
            movie_info[int(line[0])] = set(line[2].split('|'))
            for type in line[2].split('|'):
                movie_type.add(type)
    ratings = {}
    rating_path = ".\\data\\ml-1m\\ratings.txt"
    with open(rating_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('::')
            if len(line) != 4:
                continue
            if int(line[0]) not in ratings.keys():
                ratings[int(line[0])] = {}
            ratings[int(line[0])][int(line[1])] = float(line[2])
    n_users = len(user_info.keys())
    n_movies = len(index2movieid.keys())
    global_actor = PolicyNet(4, n_hiddens, n_movies).to(device)
    global_critic = ValueNet(4, n_hiddens).to(device)
    global_actor.share_memory()
    global_critic.share_memory()
    global_optimizer = torch.optim.Adam([
        {'params': global_actor.parameters(), 'lr': actor_lr},
        {'params': global_critic.parameters(), 'lr': critic_lr}
    ])
    workers = [A3CWorker(global_actor, global_critic, global_optimizer, gamma, 0.95, i, device, user_info, movie_info, movie_type, ratings, n_users, n_movies, index2movieid) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]

    all_rewards = []
    for i in range(mp.cpu_count()):
        rewards_log_file = f'rewards_log_{i}.txt'
        if os.path.exists(rewards_log_file):
            with open(rewards_log_file, 'r') as f:
                worker_rewards = [float(line.strip()) for line in f.readlines()]
                all_rewards.append(worker_rewards)

    all_rewards = np.array(all_rewards)
    average_rewards = np.mean(all_rewards, axis=0)
    smoothed_rewards = moving_average(average_rewards, window_size=50)

    plt.plot(smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Return')
    plt.savefig("Results1.svg", dpi=300, format="svg")
    plt.show()

    plt.plot(average_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Return')
    plt.savefig("Results2.svg", dpi=300, format="svg")
    plt.show()

    # accuracy = evaluate_recommendations(global_actor, env, n_movies, index2movieid, device, num_users=10, top_k=10)
    # print(f"hits@10 accuracy: {accuracy * 100:.2f}%")
    '''
    for i in range(1, n_users + 1):
        state = workers.env.get_state(i)
        list = workers.top_10(state)
        with open('recommendation.txt', 'a') as f:
            sys.stdout = f
            print(f'用户:{i}, 推荐的电影:{list}')
            sys.stdout = sys.__stdout__
    '''