import numpy as np
import pickle
import random
import gym
import time

q_table = {}

def get_state(obs, target_pos, has_picked_up):
    agent_pos = (obs[0], obs[1])
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs[10], obs[11], obs[12], obs[13], obs[14], obs[15]
    return (agent_pos, target_pos, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, has_picked_up)

def calculate_manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def train_agent(env, episodes=5000000, alpha=0.001, gamma=0.99):
    global q_table
    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(episodes):
        obs, _ = env.reset()

        stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        target_pos = stations[0]

        for i in range(1, len(stations)):
            prev_dist_to_target = calculate_manhattan_distance((obs[0], obs[1]), target_pos)
            curr_dist_to_target = calculate_manhattan_distance((obs[0], obs[1]), stations[i])

            if curr_dist_to_target < prev_dist_to_target:
                target_pos = stations[i]

        state = get_state(obs, target_pos, False)
        done = False
        total_reward = 0
        episode_step = 0
        previous_pos = None
        
        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(6)
            
            action_probs = softmax(q_table[state])
            action = np.random.choice(len(action_probs), p=action_probs)

            obs, reward, done, _ = env.step(action)
            next_state = get_state(obs, state[1], state[8])
            episode_step += 1

            shaped_reward = 0
            if done and episode_step < 200:
                shaped_reward += 100
            elif done and episode_step >= 200:
                shaped_reward -= 200

            prev_dist_to_target = calculate_manhattan_distance(state[0], state[1])
            curr_dist_to_target = calculate_manhattan_distance(next_state[0], next_state[1])

            if curr_dist_to_target < prev_dist_to_target:
                shaped_reward += 10
            if next_state[0] == previous_pos and action in [0, 1, 2, 3]:
                shaped_reward -= 10
            if state[0] == next_state[0] and action in [0, 1, 2, 3]:
                shaped_reward -= 10000
            if state[0] == next_state[0] and ((action == 4 and (state[8] or state[0] != state[1] or not state[6])) or (action == 5 and (not state[8] or state[0] != state[1] or not state[7]))):
                shaped_reward -= 10000
            if next_state[0] == next_state[1] and ((not next_state[6] or (not next_state[7] and next_state[8])) or (next_state[7] and not next_state[8])):
                next_state = get_state(obs, stations[(stations.index(next_state[1]) + 1) % 4], next_state[8])
            if state[0] != next_state[0] and ((not state[6] and next_state[6] and not next_state[8]) or (not state[7] and next_state[7] and next_state[8])):
                shaped_reward += 10
            if state[0] != next_state[0] and ((state[6] and not next_state[6] and not next_state[8]) or (state[7] and not next_state[7] and next_state[8])):
                shaped_reward -= 10
            if state[0] != next_state[0] and next_state[0] == next_state[1] and ((next_state[6] and not next_state[8]) or (next_state[7] and next_state[8])):
                shaped_reward += 10
            if state[0] != next_state[0] and state[0] == state[1]:
                shaped_reward -= 10 
            if state[0] == next_state[0] and state[0] == state[1] and state[6] and not state[8] and action == 4:
                shaped_reward += 100
                next_state = get_state(obs, stations[(stations.index(next_state[1]) + 1) % 4], True)
            if not next_state[7] and next_state[0] != next_state[1] and next_state[8] and action == 5:
                shaped_reward -= 10000
                episode_step = 200
                done = True

            reward += shaped_reward
            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)

            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            previous_pos = state[0]
            state = next_state

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(episode_step)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_step = np.mean(steps_per_episode[-100:])
            print(f"🚀 Episode {ep + 1}/{episodes}, Average Reward: {avg_reward:.2f}, Average Step: {avg_step:.2f}")

    with open("qtable.pkl", "wb") as f:
        pickle.dump(q_table, f)
    return q_table, rewards_per_episode

if __name__ == '__main__':
    try:
        from training_env import SimpleTaxiEnv
        print("Using SimpleTaxiEnv for training.")
        env = SimpleTaxiEnv(fuel_limit=200)
        print("Training agent...")
        q_table, rewards = train_agent(env)
        print("Training finished.")
    except ImportError:
        print("SimpleTaxiEnv not found.")