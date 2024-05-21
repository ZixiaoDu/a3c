# 代码用于离散环境的模型
import sys
import time
import numpy as np
import torch
from torch import nn  #所有网络基类，管理网络属性
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import gym

# 构建策略网络--actor
# 在 PolicyNet 类中，通过输入状态 x 经过神经网络计算得到各个动作的概率分布，
# 而在 ValueNet 类中，通过输入状态 x 经过神经网络计算得到该状态的价值（即给定状态下的预期累积奖励）。

class PolicyNet(nn.Module):  #继承父类
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        # 输入特征数，输出特征数
        self.fc1 = nn.Linear(n_states, n_hiddens)  #状态数，hidden数
        self.fc2 = nn.Linear(n_hiddens, n_actions)  #hidden数，动作数

    def forward(self, x):  #这个状态每个动作的概率
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)  #小于0的变为0
        x = self.fc2(x)  # [b, n_actions]
        #归一化指数函数
        # Softmax
        # 可以使正样本（正数）的结果趋近于1
        # 使负样本（负数）的结果趋近于0
        # 且样本的绝对值越大，两极化越明显。
        x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        #dim=0固定列  dim=1固定行
        return x


# 构建价值网络--critic
class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x


class bg:
    def __init__(self, user_info, movie_info, movie_type, ratings, n_users):
        #只存id不存名字
        self.user_info = user_info
        self.movie_info = movie_info
        self.movie_type = movie_type
        self.ratings = ratings
        self.n_users = n_users

    def get_random_state(self):
        userid = np.random.randint(1, self.n_users + 1)
        res = self.user_info[userid]
        res = torch.tensor(res, dtype=torch.float)
        layer_norm = nn.LayerNorm(4, eps=1e-6)  #标准化
        res = layer_norm(res)
        return res.detach().numpy(), userid

    def get_state(self, userid):
        res = self.user_info[userid]
        res = torch.tensor(res, dtype=torch.float)
        layer_norm = nn.LayerNorm(4, eps=1e-6)  # 标准化
        res = layer_norm(res)
        return res.detach().numpy()

    def get_reward(self, userid, movieid):
        reward = 0.0
        for movie, score in self.ratings[userid].items():  #枚举用户对看过电影的打分

            set1 = self.movie_info[movieid]
            set2 = self.movie_info[movie]

            sameset = set1.intersection(set2)
            similar_ratio = len(sameset) / (len(set1) + len(set2) - len(sameset))
            reward += score * similar_ratio
        return reward


# 构建模型
#Agent是基于Actor-Critic架构和PPO算法的强化学习智能体。
# torch.optim.Adam是PyTorch中用于训练神经网络的优化器之一。
# 它实现了Adam算法，这是一种对比梯度下降算法更高效的优化算法。
#
# Adam算法有三个主要参数:
#
# lr(learning rate): 学习率。表示每次参数更新时步长的大小。默认值为0.001。
# betas(beta1, beta2): 表示Adam算法中两个动量参数。默认值为(0.9, 0.999)。
# eps(epsilon): 一个很小的值，用来维持数值稳定性。默认值为1e-8。
class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device,
                 ratings, n_movies, index2movieid):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)  #转化为指定设备可用格式
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子，用于计算未来奖励的折现值
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数，每次更新策略的迭代次数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.ratings = ratings
        self.n_movies = n_movies
        self.index2movieid = index2movieid

    # 动作选择
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

    def take_action(self, state, userid):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state)
        canuse_action = []
        for action in range(0, n_movies):
            movieid = index2movieid[action + 1]
            if (movieid not in self.ratings[userid].keys()):  #找没看过的电影
                canuse_action.append(action)
        canuse_action_pro = []
        for action in canuse_action:
            canuse_action_pro.append(probs[0][action])
        action = random.choices(canuse_action, weights=canuse_action_pro, k=1)[0]
        return action

    # 训练
    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        #print(1-dones)
        td_target = rewards + self.gamma * next_q_target * (1 - dones)  #点乘，最后一个不用加
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式 :计算优势函数估计，使用时间差分误差和上一个时间步的优势函数估计
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    # 参数设置
    num_episodes = 2050  # 总迭代次数
    gamma = 0.9  # 折扣因子
    actor_lr = 1e-3  # 策略网络的学习率
    critic_lr = 1e-3  # 价值网络的学习率
    n_hiddens = 16  # 隐含层神经元个数

    return_list = []  # 保存每个回合的return

    # 数据读取 用户 电影 打分
    user_info = {}  #F0 M1
    user_path = ".\\data\\ml-1m\\users.txt"
    with open(user_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('::')
            if len(line) != 5:
                continue
            list = []
            if (line[1] == 'F'):
                list.append(0)
            else:
                list.append(1)
            list.append(int(line[2]))
            list.append(int(line[3]))

            list.append(int(line[4].split('-')[0]))
            user_info[int(line[0])] = list
    index2movieid = {}
    movie_info = {}
    movie_type = set()
    movie_path = ".\\data\\ml-1m\\movies.txt"
    index = 1
    with open(movie_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('::')
            if len(line) != 3:
                continue
            typelist = line[2].split('|')
            tempset = set()
            for type in typelist:
                movie_type.add(type)
                tempset.add(type)
            index2movieid[index] = int(line[0])
            index = index + 1
            movie_info[int(line[0])] = tempset
    ratings = {}
    rating_path = ".\\data\\ml-1m\\ratings.txt"
    with open(rating_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('::')
            if len(line) != 4:
                continue
            if (int(line[0]) not in ratings.keys()):
                ratings[int(line[0])] = {}
            ratings[int(line[0])][int(line[1])] = int(line[2])

    # 环境加载
    n_users = 6040
    n_movies = 3883
    env = bg(user_info=user_info,
             movie_info=movie_info,
             movie_type=movie_type,
             ratings=ratings,
             n_users=n_users
             )
    n_states = 4  # 状态数 4
    n_actions = n_movies  # 动作数 =电影数

    # 模型构建
    agent = PPO(n_states=n_states,  # 状态数
                n_hiddens=n_hiddens,  # 隐含层数
                n_actions=n_actions,  # 动作数
                actor_lr=actor_lr,  # 策略网络学习率
                critic_lr=critic_lr,  # 价值网络学习率
                lmbda=0.95,  # 优势函数的缩放因子
                epochs=20,  # 一组序列训练的轮次
                eps=0.2,  # PPO中截断范围的参数
                gamma=gamma,  # 折扣因子
                device=device,
                ratings=ratings,
                n_movies=n_movies,
                index2movieid=index2movieid
                )

    # 训练--回合更新 on_policy
    for i in range(num_episodes):
        start_time=time.time()
        state, userid = env.get_random_state()  #

        done = False  # 任务完成的标记
        episode_return = 0.0  # 累计每回合的reward

        # 构造数据集，保存每个回合的状态数据
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        for j in range(200):  #模拟推荐200个用户

            action = agent.take_action(state, userid)  # 动作选择

            reward = env.get_reward(userid, index2movieid[action + 1])
            next_state, next_userid = env.get_random_state()
            done = False
            if (j == 199):
                done = True
            # 保存每个时刻的状态\动作\...
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            # 更新状态
            state = next_state
            userid = next_userid
            # 累计回合奖励
            #
            episode_return += reward

        # 保存每个回合的return
        return_list.append(episode_return)
        # 模型训练
        agent.learn(transition_dict)

        end_time=time.time()
        elapsdtime=end_time-start_time

        # 打印回合信息
        #print(f'iter:{i}, return:{np.mean(return_list[-10:])}')
        with open('output.txt', 'a') as f1:
            f1.write("iter: {}, return: {}, time: {:.{}f}\n".format(i, episode_return, elapsdtime, 3))
        print(f'iter:{i}, return:{episode_return}, time:{round(elapsdtime, 3)}')


    for i in range(1, n_users + 1):
        state = env.get_state(i)
        list = agent.top_10(state)
        with open('recommendation.txt', 'a') as f:
            sys.stdout = f
            print(f'用户:{i}, 推荐的电影:{list}')
            sys.stdout = sys.__stdout__


    # 绘图
    plt.plot(return_list)
    plt.title('Model Training')
    plt.xlabel('Epoches')
    plt.ylabel('Rewards')
    plt.show()
