import numpy as np
import tensorflow as tf
import threading
import random
import collections
import pandas as pd
from sklearn.model_selection import train_test_split

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
ENTROPY_BETA = 0.01
MAX_GLOBAL_EP = 2000
UPDATE_GLOBAL_ITER = 10
N_WORKERS = 8

# TransE hyperparameters
EMBEDDING_DIM = 50
LEARNING_RATE_T = 0.01
EPOCHS_T = 100


# TransE implementation
class TransE:
    def __init__(self, num_entities, num_relations, embedding_dim, learning_rate):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        self.entity_embeddings = tf.Variable(tf.random.uniform([num_entities, embedding_dim], -1.0, 1.0))
        self.relation_embeddings = tf.Variable(tf.random.uniform([num_relations, embedding_dim], -1.0, 1.0))
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def train_step(self, head, relation, tail, negative_tail):
        with tf.GradientTape() as tape:
            pos_distance = tf.reduce_sum(tf.abs(
                self.entity_embeddings[head] + self.relation_embeddings[relation] - self.entity_embeddings[tail]),
                                         axis=1)
            neg_distance = tf.reduce_sum(tf.abs(
                self.entity_embeddings[head] + self.relation_embeddings[relation] - self.entity_embeddings[
                    negative_tail]), axis=1)

            loss = tf.reduce_mean(tf.maximum(pos_distance - neg_distance + 1, 0))

        gradients = tape.gradient(loss, [self.entity_embeddings, self.relation_embeddings])
        self.optimizer.apply_gradients(zip(gradients, [self.entity_embeddings, self.relation_embeddings]))
        return loss

    def train(self, triples, num_epochs):
        for epoch in range(num_epochs):
            np.random.shuffle(triples)
            losses = []
            for head, relation, tail, negative_tail in triples:
                loss = self.train_step(head, relation, tail, negative_tail)
                losses.append(loss)
            print(f"Epoch {epoch + 1}, Loss: {np.mean(losses)}")


# A3C model components
class ACNet(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ACNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(200, activation='relu')
        self.fc2 = tf.keras.layers.Dense(100, activation='relu')
        self.policy = tf.keras.layers.Dense(action_dim)
        self.value = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


class Worker(threading.Thread):
    def __init__(self, name, global_model, optimizer, global_ep, global_ep_r, res_queue, env):
        super(Worker, self).__init__()
        self.name = name
        self.global_model = global_model
        self.optimizer = optimizer
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        self.env = env
        self.local_model = ACNet(state_dim=EMBEDDING_DIM, action_dim=NUM_MOVIES)
        self.worker_steps = 0

    def run(self):
        total_step = 1
        while self.global_ep < MAX_GLOBAL_EP:
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while True:
                policy, _ = self.local_model(tf.convert_to_tensor([state], dtype=tf.float32))
                action = np.random.choice(range(policy.shape[1]), p=policy.numpy().ravel())
                state_, reward, done, _ = self.env.step(action)
                ep_r += reward
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    self.update_global(buffer_s, buffer_a, buffer_r, done)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:
                        with self.global_ep.get_lock():
                            self.global_ep.value += 1
                        with self.global_ep_r.get_lock():
                            if self.global_ep_r.value == 0.:
                                self.global_ep_r.value = ep_r
                            else:
                                self.global_ep_r.value = self.global_ep_r.value * 0.99 + ep_r * 0.01
                        self.res_queue.put(self.global_ep_r.value)
                        break
                state = state_
                total_step += 1
        self.res_queue.put(None)

    def update_global(self, buffer_s, buffer_a, buffer_r, done):
        buffer_v_target = []
        if done:
            v_s_ = 0
        else:
            _, v_s_ = self.local_model(tf.convert_to_tensor([buffer_s[-1]], dtype=tf.float32))
            v_s_ = v_s_.numpy()[0, 0]

        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        buffer_s = tf.convert_to_tensor(np.vstack(buffer_s), dtype=tf.float32)
        buffer_a = tf.convert_to_tensor(buffer_a, dtype=tf.int32)
        buffer_v_target = tf.convert_to_tensor(np.vstack(buffer_v_target), dtype=tf.float32)

        with tf.GradientTape() as tape:
            policy, values = self.local_model(buffer_s)
            values = tf.squeeze(values)
            advantage = buffer_v_target - values
            value_loss = tf.reduce_mean(tf.square(advantage))
            policy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=buffer_a, logits=policy) * tf.stop_gradient(
                    advantage))
            entropy = tf.reduce_mean(tf.nn.softmax(policy) * tf.nn.log_softmax(policy))
            total_loss = value_loss + policy_loss - ENTROPY_BETA * entropy

        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))
        self.local_model.set_weights(self.global_model.get_weights())


# Environment for movie recommendation
class MovieEnv:
    def __init__(self, user_embeddings, movie_embeddings):
        self.user_embeddings = user_embeddings
        self.movie_embeddings = movie_embeddings
        self.current_user = None

    def reset(self):
        self.current_user = random.choice(range(NUM_USERS))
        return self.user_embeddings[self.current_user]

    def step(self, action):
        recommended_movie_embedding = self.movie_embeddings[action]
        user_embedding = self.user_embeddings[self.current_user]
        reward = np.dot(user_embedding, recommended_movie_embedding)  # simplified reward
        done = True  # Each step is terminal in this simplified example
        return self.user_embeddings[self.current_user], reward, done, {}


# Data processing
def load_movielens_data():
    # Load the dataset
    # Assuming the dataset is a CSV file with columns: userId, movieId, rating
    ratings = pd.read_csv('./data/ml-1m/ratings.txt')
    ratings = ratings[['userId', '::', 'movieId', '::', 'rating']]

    # Create a mapping from userId and movieId to a continuous integer index
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_id_map = {id: idx for idx, id in enumerate(user_ids)}
    movie_id_map = {id: idx for idx, id in enumerate(movie_ids)}

    # Update the ratings dataframe to use these indices
    ratings['userId'] = ratings['userId'].map(user_id_map)
    ratings['movieId'] = ratings['movieId'].map(movie_id_map)

    return ratings, len(user_ids), len(movie_ids)


def prepare_triples(ratings, num_users, num_movies):
    triples = []
    for user, movie, _ in ratings.itertuples(index=False):
        negative_movie = random.choice([m for m in range(num_movies) if m != movie])
        triples.append((user, 0, num_users + movie, num_users + negative_movie))
    return triples


# Main A3C implementation
def main():
    ratings, num_users, num_movies = load_movielens_data()
    global NUM_USERS, NUM_MOVIES
    NUM_USERS = num_users
    NUM_MOVIES = num_movies

    triples = prepare_triples(ratings, NUM_USERS, NUM_MOVIES)
    transE_model = TransE(num_entities=NUM_USERS + NUM_MOVIES, num_relations=1, embedding_dim=EMBEDDING_DIM,
                          learning_rate=LEARNING_RATE_T)
    transE_model.train(triples, EPOCHS_T)

    user_embeddings = transE_model.entity_embeddings[:NUM_USERS].numpy()
    movie_embeddings = transE_model.entity_embeddings[NUM_USERS:].numpy()

    env = MovieEnv(user_embeddings, movie_embeddings)
    global_model = ACNet(state_dim=EMBEDDING_DIM, action_dim=NUM_MOVIES)
    global_model(tf.convert_to_tensor(np.random.random((1, EMBEDDING_DIM)), dtype=tf.float32))  # Build the model
    optimizer = tf.optimizers.Adam(LEARNING_RATE)
    global_ep, global_ep_r = tf.Variable(0), tf.Variable(0.0)
    res_queue = collections.deque()

    workers = [Worker(f'Worker_{i}', global_model, optimizer, global_ep, global_ep_r, res_queue, env) for i in
               range(N_WORKERS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    rewards = []
    while not res_queue.empty():
        reward = res_queue.popleft()
        if reward is not None:
            rewards.append(reward)
    print(f'Final Global Episode Reward: {global_ep_r.numpy()}')

    # Evaluation using hit@10
    hit_at_10 = evaluate_hit_at_10(global_model, env, user_embeddings, movie_embeddings)
    print(f'hit@10: {hit_at_10}')


def evaluate_hit_at_10(model, env, user_embeddings, movie_embeddings):
    hits = 0
    for user_embedding in user_embeddings:
        state = user_embedding
        policy, _ = model(tf.convert_to_tensor([state], dtype=tf.float32))
        recommendations = np.argsort(-policy.numpy().ravel())[:10]
        if np.random.choice(range(NUM_MOVIES)) in recommendations:
            hits += 1
    return hits / NUM_USERS


if __name__ == '__main__':
    main()
