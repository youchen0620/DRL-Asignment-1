import numpy as np
import pickle
import random
import gym

try:
    with open("qtable.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("Loaded Q-table from qtable.pkl")
except FileNotFoundError:
    q_table = {}

def get_state(obs, target_pos, has_picked_up):
    agent_pos = (obs[0], obs[1])
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs[10], obs[11], obs[12], obs[13], obs[14], obs[15]
    return (agent_pos, target_pos, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, has_picked_up)

def calculate_manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

initialized = False
previous_state = None
previous_action = None

def get_action(obs):
    global initialized
    global previous_state
    global previous_action

    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]

    if not initialized:
        initialized = True
        state = get_state(obs, stations[0], False)

        action = None
        if state not in q_table:
            action = random.choice(range(6))
        else:
            action = int(np.argmax(q_table[state]))
        
        previous_state = state

        return action
    
    state = get_state(obs, previous_state[1], previous_state[8])

    if state[7] and not state[8]:
        state = get_state(obs, stations[(stations.index(state[1]) + 1) % 4], False)
    if previous_state[0] == state[0] and state[0] == state[1]:
        if state[6] and not state[8] and previous_action == 4:
            state = get_state(obs, stations[(stations.index(state[1]) + 1) % 4], True)
    if not state[7] and state[0] not in stations and state[8] and previous_action == 5:
        state = get_state(obs, state[1], False)

    action = None
    if state not in q_table:
        action = random.choice(range(6))
    else:
        action = int(np.argmax(q_table[state]))

    previous_state = state
    previous_action = action

    return action

def train_agent(env, episodes=500000, alpha=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.0, decay_rate=0.99995):
    global q_table
    rewards_per_episode = []
    steps_per_episode = []
    epsilon = epsilon_start

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
        
        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(6)

            if random.random() < epsilon:
                action = random.choice(range(6))
            else:
                action = int(np.argmax(q_table[state]))

            obs, reward, done, _ = env.step(action)
            next_state = get_state(obs, state[1], state[8])
            episode_step += 1

            shaped_reward = 0
            if done:
                shaped_reward += 100

            prev_dist_to_target = calculate_manhattan_distance(state[0], state[1])
            curr_dist_to_target = calculate_manhattan_distance(next_state[0], next_state[1])

            if curr_dist_to_target < prev_dist_to_target:
                shaped_reward += 0.2
            if state[0] == next_state[0] and action in [0, 1, 2, 3]:
                shaped_reward -= 10
            if state[0] == next_state[0] and ((action == 4 and not state[6] and state[0] != state[1]) or (action == 5 and not state[7] and state[0] != state[1])):
                shaped_reward -= 10000
            if next_state[7] and not next_state[8]:
                next_state = get_state(obs, stations[(stations.index(next_state[1]) + 1) % 4], False)
            if state[0] != next_state[0] and ((not state[6] and next_state[6] and not next_state[8]) or (not state[7] and next_state[7] and next_state[8])):
                shaped_reward += 10
            if state[0] != next_state[0] and ((state[6] and not next_state[6] and not next_state[8]) or (state[7] and not next_state[7] and next_state[8])):
                shaped_reward -= 10
            if state[0] != next_state[0] and next_state[0] == next_state[1] and ((next_state[6] and not next_state[8]) or (next_state[7] and next_state[8])):
                shaped_reward += 10
            if state[0] != next_state[0] and state[0] == state[1]:
                shaped_reward -= 10 
            if state[0] == next_state[0] and state[0] == state[1]:
                if state[6] and not state[8] and action == 4:
                    shaped_reward += 100
                    next_state = get_state(obs, stations[(stations.index(next_state[1]) + 1) % 4], True)
            if not next_state[7] and next_state[0] != next_state[1] and next_state[8] and action == 5:
                shaped_reward -= 10000
                episode_step = 5000
                done = True

            reward += shaped_reward
            total_reward += reward

            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)

            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            state = next_state

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(episode_step)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_step = np.mean(steps_per_episode[-100:])
            print(f"🚀 Episode {ep + 1}/{episodes}, Average Reward: {avg_reward:.2f}, Average Step: {avg_step:.2f}, Epsilon: {epsilon:.3f}")

    with open("qtable.pkl", "wb") as f:
        pickle.dump(q_table, f)
    return q_table, rewards_per_episode

if __name__ == '__main__':
    try:
        from simple_custom_taxi_env import SimpleTaxiEnv
        print("Using SimpleTaxiEnv for training.")
        env = SimpleTaxiEnv(fuel_limit=5000)
    except ImportError:
        print("SimpleTaxiEnv not found, using gym.make('Taxi-v3') instead.")
        env = gym.make("Taxi-v3", render_mode="ansi")
    
    print("Training agent...")
    q_table, rewards = train_agent(env)
    print("Training finished.")