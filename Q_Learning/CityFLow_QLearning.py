import gym
import gym_cityflow
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

EPISODES =100

alpha = 0.1
discount_factor = 0.05
epsilon_decay_factor = 0.9

epsilon = 0.05


def get_action_probabilities(Q, chosen_state, n, e):
    all_ones = np.ones(n, dtype=float)
    probs = (all_ones * e) / n
    best_action = np.argmax(Q[chosen_state])
    probs[best_action] += 1.0 - e
    return probs


def update_epsilon(e):
    return e * epsilon_decay_factor


def get_chosen_state(state):
    return np.argmax(state)


if __name__ == "__main__":
    env = gym.make('gym_cityflow:CityFlow-1x1-LowTraffic-v0')
    rewards = []
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for _ in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        turns = 0
        chosen_state = get_chosen_state(state)
        e = epsilon
        while not done:
            action_probabilities = get_action_probabilities(Q, chosen_state, env.action_space.n, e)
            action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)

            next_state, reward, done, _ = env.step(action)

            next_chosen_state = get_chosen_state(next_state)
            best_next_action = np.argmax(Q[next_chosen_state])

            second_term = reward + (discount_factor * Q[next_chosen_state][best_next_action])
            delta = second_term - Q[chosen_state][action]
            Q[chosen_state][action] += alpha * delta

            chosen_state = get_chosen_state(next_state)
            total_reward += reward

            if turns % 10 == 0:
                e = update_epsilon(e)

            turns += 1
        rewards.append(total_reward)

    print(np.mean(rewards))
    plt.plot(rewards, linewidth=1.0)
    plt.savefig("plot_real.png")
